#!/usr/bin/env python3
"""Predict the versioned fixed-batch benchmark from lightweight profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark_protocol import load_benchmark_spec  # noqa: E402
from core.qwen3_cost_model import LLMCostModel  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(spec: dict, model: LLMCostModel, hardware_id: str,
            tp_size: int, spec_path: Path) -> dict:
    expected_tp = int(spec.get("deployment", {}).get("tensor_parallel_size", tp_size))
    if expected_tp != tp_size:
        raise ValueError(
            f"benchmark spec requires TP{expected_tp}, received TP{tp_size}")
    stages = {"prefill": [], "decode": []}
    extrapolated = 0
    for stage in ("prefill", "decode"):
        for case in spec["workloads"][stage]:
            batch = int(case["batch_size"])
            length_key = "prompt_length" if stage == "prefill" else "kv_length"
            length = int(case[length_key])
            estimate = (model.predict_prefill(batch, length, tp_size)
                        if stage == "prefill"
                        else model.predict_decode(batch, length, tp_size))
            total_ms = float(estimate["total_time_ms"])
            tokens = batch * length if stage == "prefill" else batch
            extrapolated += int(bool(estimate.get("extrapolated")))
            row = {
                "case": case["case"],
                "batch": batch,
                "original_batch": batch,
                "seq" if stage == "prefill" else "kv": length,
                "total_ms": total_ms,
                "throughput_tok_s": tokens * 1000.0 / total_ms,
                "memory_oom": False,
                "extrapolated": bool(estimate.get("extrapolated")),
                "profile_source": estimate.get("profile_source"),
                "communication_profile_source": estimate.get(
                    "communication_profile_source"),
                "estimated_bottleneck": estimate.get("estimated_bottleneck"),
                "operator_breakdown": estimate.get("operator_breakdown"),
                "warnings": estimate.get("warnings", []),
            }
            stages[stage].append(row)
    return {
        "schema_version": 1,
        "experiment_type": "fixed_batch_runtime_prediction",
        "meta": {
            "protocol_version": spec["protocol_version"],
            "benchmark_spec": str(spec_path.resolve()),
            "benchmark_spec_sha256": sha256(spec_path),
            "model_id": spec["model"]["model_id"],
            "model_name": spec["model"]["name"],
            "model_config_sha256": spec["model"].get("config_sha256"),
            "dtype": spec["model"]["dtype"],
            "gpu_type": hardware_id,
            "tp": tp_size,
            "comparison_status": "prediction_requires_ground_truth_validation",
        },
        "prediction_policy": {
            "source": "operator and collective profiles",
            "profile_timing_statistic": model.profile_statistic,
            "full_fixed_batch_ground_truth_used_as_input": False,
            "end_to_end_calibration_enabled": model.enable_e2e_compensation,
        },
        "coverage": {
            "point_count": sum(len(value) for value in stages.values()),
            "extrapolated_point_count": extrapolated,
        },
        **stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--benchmark-spec", type=Path, default=(
        PROJECT_ROOT / "configs" / "benchmark_specs" /
        "qwen3_32b_fixed_batch_tp4_v1.json"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path, required=True)
    parser.add_argument("--collective-profile-dir", type=Path, required=True)
    parser.add_argument("--profile-statistic", choices=("mean", "p50"),
                        default="p50")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.enable_calibration and not args.calibration_file:
        raise SystemExit("--enable-calibration requires --calibration-file")

    spec = load_benchmark_spec(args.benchmark_spec)
    cost_model = LLMCostModel(
        str(args.data_dir),
        operator_profile_dir=str(args.operator_profile_dir),
        collective_profile_dir=str(args.collective_profile_dir),
        profile_statistic=args.profile_statistic,
        calibration_file=(str(args.calibration_file)
                          if args.calibration_file else None),
        enable_e2e_compensation=args.enable_calibration,
    )
    result = predict(spec, cost_model, args.hardware_id,
                     args.tp_size, args.benchmark_spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"predicted {result['coverage']['point_count']} points")
    print(f"extrapolated points: {result['coverage']['extrapolated_point_count']}")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
