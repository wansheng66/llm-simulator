#!/usr/bin/env python3
"""Compare two strict offline fixed-batch suites on identical workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple


RUNTIME_KEYS = (
    "dtype", "max_model_len", "max_num_batched_tokens", "max_num_seqs",
    "enforce_eager", "enable_prefix_caching", "block_size_tokens",
)


def geometric_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def artifact_path(manifest_path: Path, stored: str) -> Path:
    path = Path(stored)
    if path.exists():
        return path
    local = manifest_path.parent / path.name
    if local.exists():
        return local
    raise FileNotFoundError(f"artifact not found: {stored} or {local}")


def normalize_runtime(config: Dict) -> Dict:
    result = {key: config.get(key) for key in RUNTIME_KEYS}
    result["tp_size"] = config.get(
        "tensor_parallel_size", config.get("tp_size"))
    result["block_size_tokens"] = config.get(
        "block_size_tokens", config.get("block_size"))
    return result


def hardware_id(payload: Dict) -> str:
    snapshot = (payload.get("serving_host_hardware") or {}).get("snapshot") or {}
    identity = snapshot.get("hardware_identity") or {}
    return str(identity.get("hardware_id") or "unknown")


def load_suite(manifest_path: Path) -> Dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "fixed_batch_suite_manifest":
        raise ValueError(f"not a fixed-batch suite manifest: {manifest_path}")
    if manifest.get("collector_mode") != "offline":
        raise ValueError("relative hardware truth requires collector_mode=offline")
    if manifest.get("submission_mode") != "offline_enqueue_barrier":
        raise ValueError("relative hardware truth requires the enqueue barrier")

    points: Dict[Tuple[str, int, int], Dict] = {}
    identity = None
    runtime = None
    policy = None
    for stored in manifest.get("files", []):
        path = artifact_path(manifest_path, stored)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("schema_version") != 2
                or payload.get("collector") != "vllm_fixed_batch_offline"
                or payload.get("valid") is not True
                or payload.get("fixed_batch_valid") is not True):
            raise ValueError(f"invalid strict fixed-batch artifact: {path}")
        config = payload["configuration"]
        stage = config["stage"]
        length_key = "prompt_length" if stage == "prefill" else "initial_kv_length"
        key = (stage, int(config["batch_size"]), int(config[length_key]))
        if key in points:
            raise ValueError(f"duplicate fixed-batch shape: {key}")
        summary = payload.get("summary") or {}
        mean_ms = float(summary.get("mean_ms", 0))
        std_ms = float(summary.get("std_ms", 0))
        count = int(summary.get("count", 0))
        if mean_ms <= 0 or count <= 0:
            raise ValueError(f"missing timing summary: {path}")
        model_hash = (payload.get("metadata", {}).get("model", {})
                      .get("config_sha256"))
        current_identity = {
            "model_config_sha256": model_hash,
            "tp_size": int(config["tp_size"]),
            "dtype": config.get("dtype"),
        }
        current_runtime = normalize_runtime(
            payload.get("deployment", {}).get("config", {}))
        current_policy = {
            "submission_mode": config.get("submission_mode"),
            "release_policy": config.get("release_policy"),
            "v1_engine_core_multiprocessing": config.get(
                "v1_engine_core_multiprocessing"),
            "async_scheduling": config.get("async_scheduling"),
            "enable_chunked_prefill": config.get("enable_chunked_prefill"),
            "prompt_policy": config.get("prompt_policy"),
            "warmup_batches": int(config.get("warmup_batches", 0)),
            "measured_batches": int(config.get("measured_batches", 0)),
        }
        identity = identity or current_identity
        runtime = runtime or current_runtime
        policy = policy or current_policy
        if current_identity != identity or current_runtime != runtime or current_policy != policy:
            raise ValueError(f"suite contains mixed identities or policies: {path}")
        points[key] = {
            "path": str(path.resolve()),
            "payload": payload,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "count": count,
            "cv_pct": std_ms / mean_ms * 100.0,
        }
    if not points:
        raise ValueError(f"suite contains no points: {manifest_path}")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "hardware_id": hardware_id(next(iter(points.values()))["payload"]),
        "identity": identity,
        "runtime": runtime,
        "policy": policy,
        "points": points,
    }


def compare_suites(reference: Dict, candidate: Dict, max_cv_pct: float) -> Dict:
    if reference["identity"] != candidate["identity"]:
        raise ValueError("suite model, TP or dtype identity differs")
    if reference["runtime"] != candidate["runtime"]:
        raise ValueError("suite scheduler/runtime configuration differs")
    if reference["policy"] != candidate["policy"]:
        raise ValueError("suite measurement policy differs")
    if set(reference["points"]) != set(candidate["points"]):
        raise ValueError("suites do not contain exactly the same workload shapes")

    points = []
    stage_ratios = {"prefill": [], "decode": []}
    winner_counts = {
        "prefill": {"reference": 0, "candidate": 0, "tie": 0},
        "decode": {"reference": 0, "candidate": 0, "tie": 0},
    }
    repeatability_ok = True
    for stage, batch, length in sorted(reference["points"]):
        ref = reference["points"][(stage, batch, length)]
        cand = candidate["points"][(stage, batch, length)]
        token_count = batch * length if stage == "prefill" else batch
        ref_rate = token_count * 1000.0 / ref["mean_ms"]
        cand_rate = token_count * 1000.0 / cand["mean_ms"]
        speedup = cand_rate / ref_rate
        cv_ok = max(ref["cv_pct"], cand["cv_pct"]) <= max_cv_pct
        repeatability_ok = repeatability_ok and cv_ok
        if math.isclose(speedup, 1.0, rel_tol=0.01):
            winner = "tie"
        elif speedup > 1.0:
            winner = "candidate"
        else:
            winner = "reference"
        winner_counts[stage][winner] += 1
        stage_ratios[stage].append(speedup)
        points.append({
            "stage": stage,
            "batch_size": batch,
            "representative_length": length,
            "reference_mean_ms": ref["mean_ms"],
            "candidate_mean_ms": cand["mean_ms"],
            "reference_throughput_tokens_per_s": ref_rate,
            "candidate_throughput_tokens_per_s": cand_rate,
            "candidate_speedup_vs_reference": speedup,
            "winner": winner,
            "reference_cv_pct": ref["cv_pct"],
            "candidate_cv_pct": cand["cv_pct"],
            "repeatability_ok": cv_ok,
        })

    return {
        "schema_version": 1,
        "experiment_type": "fixed_batch_relative_ground_truth",
        "score_direction": (
            "candidate throughput / reference throughput; values above 1 mean "
            "the candidate is faster"),
        "reference": {
            "hardware_id": reference["hardware_id"],
            "manifest": reference["manifest_path"],
        },
        "candidate": {
            "hardware_id": candidate["hardware_id"],
            "manifest": candidate["manifest_path"],
        },
        "identity": reference["identity"],
        "runtime": reference["runtime"],
        "measurement_policy": reference["policy"],
        "acceptance": {
            "max_cv_pct": max_cv_pct,
            "all_points_repeatable": repeatability_ok,
            "paired_point_count": len(points),
            "passed": repeatability_ok and len(points) == 18,
        },
        "scores": {
            "p_score": geometric_mean(stage_ratios["prefill"]),
            "d_score": geometric_mean(stage_ratios["decode"]),
            "combined_score": None,
            "combined_score_status": (
                "undefined until an explicit P/D workload mix is selected"),
        },
        "winner_counts": winner_counts,
        "points": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--max-cv-pct", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_cv_pct <= 0:
        raise SystemExit("--max-cv-pct must be positive")

    output = compare_suites(
        load_suite(args.reference_manifest),
        load_suite(args.candidate_manifest),
        args.max_cv_pct,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "fixed_batch_relative_report.json"
    csv_path = args.output_dir / "fixed_batch_relative_cases.csv"
    report_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output["points"][0]))
        writer.writeheader()
        writer.writerows(output["points"])

    print(f"P-Score={output['scores']['p_score']:.4f}x")
    print(f"D-Score={output['scores']['d_score']:.4f}x")
    print(f"accepted={output['acceptance']['passed']}")
    print(f"report: {report_path}")
    return 0 if output["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
