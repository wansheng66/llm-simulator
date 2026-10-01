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
from core.runtime_backend import (  # noqa: E402
    load_runtime_backend,
    validate_backend_identity,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a repository-relative path in portable report artifacts."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def predict(spec: dict, model: LLMCostModel, hardware_id: str,
            tp_size: int, spec_path: Path,
            runtime_backend: dict | None = None,
            physical_accelerators: int | None = None) -> dict:
    deployment = spec.get("deployment", {})
    expected_tp = deployment.get("tensor_parallel_size")
    if expected_tp is not None and int(expected_tp) != tp_size:
        raise ValueError(
            f"benchmark spec requires TP{expected_tp}, received TP{tp_size}")
    expected_physical = deployment.get("physical_accelerator_count")
    if expected_physical is not None:
        if physical_accelerators is None:
            raise ValueError(
                "benchmark spec requires --physical-accelerators")
        if int(expected_physical) != int(physical_accelerators):
            raise ValueError(
                "benchmark spec requires "
                f"{expected_physical} physical accelerators, received "
                f"{physical_accelerators}")
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
            "benchmark_spec": portable_path(spec_path),
            "benchmark_spec_sha256": sha256(spec_path),
            "model_id": spec["model"]["model_id"],
            "model_name": spec["model"]["name"],
            "model_config_sha256": spec["model"].get("config_sha256"),
            "dtype": spec["model"]["dtype"],
            "gpu_type": hardware_id,
            "tp": tp_size,
            "physical_accelerators": physical_accelerators,
            "comparison_track": spec.get(
                "benchmark_track", "iso_logical_tp"),
            "comparison_status": "prediction_requires_ground_truth_validation",
            "runtime_backend": runtime_backend,
        },
        "prediction_policy": {
            "source": "operator and collective profiles",
            "implementation": {
                "cost_model": "core/qwen3_cost_model.py",
                "cost_model_sha256": sha256(
                    PROJECT_ROOT / "core" / "qwen3_cost_model.py"),
                "predictor": "scripts/predict_fixed_batch_benchmark.py",
                "predictor_sha256": sha256(Path(__file__)),
            },
            "profile_timing_statistic": model.profile_statistic,
            "operator_profile_timing_statistic": (
                getattr(model, "operator_profile_statistic",
                        model.profile_statistic)),
            "collective_profile_timing_statistic": (
                getattr(model, "collective_profile_statistic",
                        model.profile_statistic)),
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
    parser.add_argument("--hardware-id")
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--physical-accelerators", type=int)
    parser.add_argument(
        "--runtime-backend-config", type=Path,
        help=("Versioned hardware backend manifest. When supplied, profile "
              "paths, statistic and calibration cannot be overridden."))
    parser.add_argument("--benchmark-spec", type=Path, default=(
        PROJECT_ROOT / "configs" / "benchmark_specs" /
        "qwen3_32b_fixed_batch_tp4_v1.json"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path)
    parser.add_argument("--collective-profile-dir", type=Path)
    parser.add_argument("--profile-statistic", choices=("mean", "p50"),
                        default=None)
    parser.add_argument("--operator-profile-statistic",
                        choices=("mean", "p50"))
    parser.add_argument("--collective-profile-statistic",
                        choices=("mean", "p50"))
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_benchmark_spec(args.benchmark_spec)
    backend_metadata = None
    if args.runtime_backend_config:
        manual = (
            args.operator_profile_dir, args.collective_profile_dir,
            args.profile_statistic, args.operator_profile_statistic,
            args.collective_profile_statistic, args.calibration_file,
            args.enable_calibration,
        )
        if any(manual):
            raise SystemExit(
                "--runtime-backend-config cannot be combined with manual "
                "profile/calibration arguments")
        backend = load_runtime_backend(
            args.runtime_backend_config, PROJECT_ROOT)
        manifest = backend["manifest"]
        hardware_id = args.hardware_id or manifest["hardware_id"]
        tp_size = args.tp_size or int(manifest["tp_size"])
        validate_backend_identity(backend, spec, hardware_id, tp_size)
        operator_profile_dir = backend["operator_profile_dir"]
        collective_profile_dir = backend["collective_profile_dir"]
        profile_statistic = backend["profile_statistic"]
        operator_profile_statistic = backend["operator_profile_statistic"]
        collective_profile_statistic = backend[
            "collective_profile_statistic"]
        calibration_file = backend["calibration_file"]
        enable_calibration = backend["calibration_enabled"]
        backend_metadata = {
            "backend_id": manifest["backend_id"],
            "platform_family": manifest["platform_family"],
            "manifest": portable_path(backend["manifest_path"]),
            "fingerprints": backend["fingerprints"],
            "validation": manifest["validation"],
        }
    else:
        if not args.hardware_id or args.tp_size is None:
            raise SystemExit(
                "manual mode requires --hardware-id and --tp-size")
        if not args.operator_profile_dir or not args.collective_profile_dir:
            raise SystemExit(
                "manual mode requires both profile directories")
        if args.enable_calibration and not args.calibration_file:
            raise SystemExit("--enable-calibration requires --calibration-file")
        hardware_id = args.hardware_id
        tp_size = args.tp_size
        operator_profile_dir = args.operator_profile_dir
        collective_profile_dir = args.collective_profile_dir
        profile_statistic = args.profile_statistic or "p50"
        operator_profile_statistic = (
            args.operator_profile_statistic or profile_statistic)
        collective_profile_statistic = (
            args.collective_profile_statistic or profile_statistic)
        calibration_file = args.calibration_file
        enable_calibration = args.enable_calibration

    cost_model = LLMCostModel(
        str(args.data_dir),
        operator_profile_dir=str(operator_profile_dir),
        collective_profile_dir=str(collective_profile_dir),
        profile_statistic=profile_statistic,
        operator_profile_statistic=operator_profile_statistic,
        collective_profile_statistic=collective_profile_statistic,
        calibration_file=(str(calibration_file)
                          if calibration_file else None),
        enable_e2e_compensation=enable_calibration,
    )
    result = predict(spec, cost_model, hardware_id,
                     tp_size, args.benchmark_spec, backend_metadata,
                     args.physical_accelerators)
    point_count = int(result["coverage"]["point_count"])
    extrapolated_count = int(
        result["coverage"]["extrapolated_point_count"])
    requires_all_extrapolated = bool(
        spec.get("acceptance", {}).get(
            "all_predictions_must_be_marked_extrapolated", False
        )
    )
    result["coverage"]["all_extrapolated_required"] = (
        requires_all_extrapolated
    )
    result["coverage"]["coverage_requirement_satisfied"] = (
        not requires_all_extrapolated
        or extrapolated_count == point_count
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"predicted {result['coverage']['point_count']} points")
    print(f"extrapolated points: {result['coverage']['extrapolated_point_count']}")
    print(f"report: {args.output}")
    if not result["coverage"]["coverage_requirement_satisfied"]:
        print(
            "ERROR: BenchmarkSpec requires every prediction to be "
            "extrapolated, but the loaded profiling support covers "
            f"{point_count - extrapolated_count}/{point_count} cases."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
