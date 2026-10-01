#!/usr/bin/env python3
"""Validate a frozen runtime-extrapolation prediction against strict truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Tuple


Shape = Tuple[str, int, int]
DATASET_ROLE = "runtime_extrapolation_validation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_path(manifest_path: Path, stored: str) -> Path:
    path = Path(stored)
    if path.exists():
        return path
    local = manifest_path.parent / path.name
    if local.exists():
        return local
    raise FileNotFoundError(f"artifact not found: {stored} or {local}")


def prediction_points(payload: Dict[str, Any]) -> Dict[Shape, Dict[str, Any]]:
    result: Dict[Shape, Dict[str, Any]] = {}
    for stage in ("prefill", "decode"):
        length_key = "seq" if stage == "prefill" else "kv"
        for row in payload.get(stage, []):
            shape = (stage, int(row["batch"]), int(row[length_key]))
            if shape in result:
                raise ValueError(f"duplicate prediction shape: {shape}")
            result[shape] = row
    return result


def specification_points(spec: Dict[str, Any]) -> Dict[Shape, Dict[str, Any]]:
    result: Dict[Shape, Dict[str, Any]] = {}
    for stage in ("prefill", "decode"):
        length_key = "prompt_length" if stage == "prefill" else "kv_length"
        for row in spec.get("workloads", {}).get(stage, []):
            shape = (stage, int(row["batch_size"]), int(row[length_key]))
            if shape in result:
                raise ValueError(f"duplicate benchmark shape: {shape}")
            result[shape] = row
    return result


def truth_points(manifest_path: Path) -> tuple[Dict[Shape, Dict[str, Any]], Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "fixed_batch_suite_manifest":
        raise ValueError("ground truth is not a fixed-batch suite manifest")
    if manifest.get("dataset_role") != DATASET_ROLE:
        raise ValueError(
            f"ground-truth manifest must declare dataset_role={DATASET_ROLE}"
        )
    result: Dict[Shape, Dict[str, Any]] = {}
    for stored in manifest.get("files", []):
        path = artifact_path(manifest_path, stored)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("valid") is not True or
                payload.get("fixed_batch_valid") is not True):
            raise ValueError(f"invalid fixed-batch ground truth: {path}")
        config = payload["configuration"]
        stage = config["stage"]
        length_key = (
            "prompt_length" if stage == "prefill" else "initial_kv_length"
        )
        shape = (stage, int(config["batch_size"]), int(config[length_key]))
        if shape in result:
            raise ValueError(f"duplicate ground-truth shape: {shape}")
        summary = payload.get("summary") or {}
        result[shape] = {
            "path": str(path.resolve()),
            "mean_ms": float(summary["mean_ms"]),
            "std_ms": float(summary["std_ms"]),
            "count": int(summary["count"]),
            "deployment_config_sha256": payload.get(
                "deployment", {}).get("config_sha256"),
            "max_num_batched_tokens": payload.get(
                "deployment", {}).get("config", {}).get(
                    "max_num_batched_tokens"),
        }
    return result, manifest


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty validation group")
    absolute = [abs(float(row["error_pct"])) for row in rows]
    signed = [float(row["error_pct"]) for row in rows]
    return {
        "count": len(rows),
        "mape_pct": statistics.fmean(absolute),
        "bias_pct": statistics.fmean(signed),
        "p90_absolute_error_pct": percentile(absolute, 0.9),
        "max_absolute_error_pct": max(absolute),
    }


def require_same_shapes(
    expected: set[Shape], predicted: set[Shape], truth: set[Shape],
) -> None:
    if expected == predicted == truth:
        return
    raise ValueError(
        "benchmark, prediction and ground-truth shapes differ: "
        f"prediction_missing={sorted(expected - predicted)}, "
        f"prediction_extra={sorted(predicted - expected)}, "
        f"truth_missing={sorted(expected - truth)}, "
        f"truth_extra={sorted(truth - expected)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument(
        "--mape-threshold-pct", type=float,
        help="override the per-stage threshold declared by the BenchmarkSpec",
    )
    parser.add_argument(
        "--max-cv-pct", type=float,
        help="override the ground-truth CV limit declared by the BenchmarkSpec",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.benchmark_spec.read_text(encoding="utf-8"))
    if spec.get("dataset_role") != DATASET_ROLE:
        raise SystemExit("benchmark spec has the wrong dataset_role")
    acceptance = spec.get("acceptance", {})
    mape_threshold = float(
        args.mape_threshold_pct
        if args.mape_threshold_pct is not None
        else acceptance.get("stage_mape_threshold_pct", 20.0)
    )
    max_cv = float(
        args.max_cv_pct
        if args.max_cv_pct is not None
        else acceptance.get("maximum_ground_truth_cv_pct", 5.0)
    )
    if mape_threshold <= 0 or max_cv <= 0:
        raise SystemExit("MAPE and CV thresholds must be positive")

    expected_hash = sha256(args.benchmark_spec)
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    if prediction.get("meta", {}).get("benchmark_spec_sha256") != expected_hash:
        raise ValueError("prediction was not frozen from this benchmark spec")
    predicted = prediction_points(prediction)
    expected = specification_points(spec)
    truth, manifest = truth_points(args.ground_truth_manifest)
    if manifest.get("benchmark_spec_sha256") != expected_hash:
        raise ValueError("ground truth was not collected from this benchmark spec")
    require_same_shapes(set(expected), set(predicted), set(truth))

    rows: list[Dict[str, Any]] = []
    for shape in sorted(expected):
        stage, batch, length = shape
        specification = expected[shape]
        estimate = predicted[shape]
        observed = truth[shape]
        predicted_ms = float(estimate["total_ms"])
        observed_ms = float(observed["mean_ms"])
        error_pct = 100.0 * (predicted_ms - observed_ms) / observed_ms
        cv_pct = 100.0 * observed["std_ms"] / observed_ms
        rows.append({
            "case": specification["case"],
            "stage": stage,
            "extrapolation_axis": specification.get(
                "extrapolation_axis", "unspecified"),
            "batch_size": batch,
            "representative_length": length,
            "observed_ms": observed_ms,
            "predicted_ms": predicted_ms,
            "error_pct": error_pct,
            "absolute_error_pct": abs(error_pct),
            "ground_truth_cv_pct": cv_pct,
            "ground_truth_count": observed["count"],
            "prediction_marked_extrapolated": bool(
                estimate.get("extrapolated")),
            "ground_truth_max_num_batched_tokens": observed.get(
                "max_num_batched_tokens"),
            "ground_truth_file": observed["path"],
        })

    stage_rows = {
        stage: [row for row in rows if row["stage"] == stage]
        for stage in ("prefill", "decode")
    }
    stage_summary = {
        stage: summarize(items) for stage, items in stage_rows.items()
    }
    axes = sorted({str(row["extrapolation_axis"]) for row in rows})
    axis_summary = {
        axis: summarize([
            row for row in rows if row["extrapolation_axis"] == axis
        ])
        for axis in axes
    }
    all_extrapolated = all(
        row["prediction_marked_extrapolated"] for row in rows)
    cv_passed = all(
        float(row["ground_truth_cv_pct"]) <= max_cv for row in rows)
    stage_mape_passed = all(
        value["mape_pct"] <= mape_threshold
        for value in stage_summary.values()
    )
    passed = all_extrapolated and cv_passed and stage_mape_passed

    report = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_runtime_extrapolation_validation",
        "dataset_role": DATASET_ROLE,
        "benchmark_spec": str(args.benchmark_spec.resolve()),
        "benchmark_spec_sha256": expected_hash,
        "frozen_prediction": str(args.prediction.resolve()),
        "frozen_prediction_sha256": sha256(args.prediction),
        "ground_truth_manifest": str(args.ground_truth_manifest.resolve()),
        "ground_truth_manifest_sha256": sha256(args.ground_truth_manifest),
        "ground_truth_collection_policy": {
            "collector_mode": manifest.get("collector_mode"),
            "offline_engine_lifecycle": manifest.get(
                "offline_engine_lifecycle"),
            "submission_mode": manifest.get("submission_mode"),
            "reuse_policy": manifest.get("reuse_policy"),
            "mixed_deployment_configs": manifest.get(
                "mixed_deployment_configs"),
        },
        "acceptance": {
            "stage_mape_threshold_pct": mape_threshold,
            "maximum_ground_truth_cv_pct": max_cv,
            "all_predictions_marked_extrapolated": all_extrapolated,
            "ground_truth_cv_passed": cv_passed,
            "stage_mape_passed": stage_mape_passed,
            "passed": passed,
        },
        "summary": {
            "by_stage": stage_summary,
            "by_extrapolation_axis": axis_summary,
        },
        "points": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "runtime_extrapolation_validation.json"
    csv_path = args.output_dir / "runtime_extrapolation_points.csv"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for stage in ("prefill", "decode"):
        value = stage_summary[stage]
        print(
            f"{stage}: MAPE={value['mape_pct']:.2f}%, "
            f"bias={value['bias_pct']:.2f}%"
        )
    for axis in axes:
        print(
            f"axis {axis}: MAPE={axis_summary[axis]['mape_pct']:.2f}%"
        )
    print(f"all predictions extrapolated: {all_extrapolated}")
    print(f"ground-truth CV passed: {cv_passed}")
    print(f"passed={passed}")
    print(f"report: {report_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
