#!/usr/bin/env python3
"""Validate a frozen runtime-interpolation prediction against strict truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Dict, Tuple


Shape = Tuple[str, int, int]


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


def prediction_points(payload: Dict) -> Dict[Shape, Dict]:
    result = {}
    for stage in ("prefill", "decode"):
        length_key = "seq" if stage == "prefill" else "kv"
        for row in payload.get(stage, []):
            shape = (stage, int(row["batch"]), int(row[length_key]))
            if shape in result:
                raise ValueError(f"duplicate prediction shape: {shape}")
            result[shape] = row
    return result


def truth_points(manifest_path: Path) -> tuple[Dict[Shape, Dict], Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_type") != "fixed_batch_suite_manifest":
        raise ValueError("ground truth is not a fixed-batch suite manifest")
    if manifest.get("dataset_role") != "runtime_interpolation_validation":
        raise ValueError(
            "ground-truth manifest must declare dataset_role="
            "runtime_interpolation_validation"
        )
    result = {}
    for stored in manifest.get("files", []):
        path = artifact_path(manifest_path, stored)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("valid") is not True
                or payload.get("fixed_batch_valid") is not True):
            raise ValueError(f"invalid fixed-batch ground truth: {path}")
        config = payload["configuration"]
        stage = config["stage"]
        length_key = "prompt_length" if stage == "prefill" else "initial_kv_length"
        shape = (stage, int(config["batch_size"]), int(config[length_key]))
        if shape in result:
            raise ValueError(f"duplicate ground-truth shape: {shape}")
        result[shape] = {
            "path": str(path.resolve()),
            "mean_ms": float(payload["summary"]["mean_ms"]),
            "std_ms": float(payload["summary"]["std_ms"]),
            "count": int(payload["summary"]["count"]),
        }
    return result, manifest


def summarize(rows: list[Dict]) -> Dict:
    absolute = [abs(float(row["error_pct"])) for row in rows]
    signed = [float(row["error_pct"]) for row in rows]
    return {
        "count": len(rows),
        "mape_pct": statistics.fmean(absolute),
        "bias_pct": statistics.fmean(signed),
        "max_absolute_error_pct": max(absolute),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--mape-threshold-pct", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.mape_threshold_pct <= 0:
        raise SystemExit("--mape-threshold-pct must be positive")
    spec = json.loads(args.benchmark_spec.read_text(encoding="utf-8"))
    if spec.get("dataset_role") != "runtime_interpolation_validation":
        raise SystemExit("benchmark spec has the wrong dataset_role")
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    expected_hash = sha256(args.benchmark_spec)
    if prediction.get("meta", {}).get("benchmark_spec_sha256") != expected_hash:
        raise ValueError("prediction was not frozen from this benchmark spec")
    predicted = prediction_points(prediction)
    truth, manifest = truth_points(args.ground_truth_manifest)
    if set(predicted) != set(truth):
        missing_truth = sorted(set(predicted) - set(truth))
        missing_prediction = sorted(set(truth) - set(predicted))
        raise ValueError(
            "prediction and ground truth shapes differ: "
            f"missing_truth={missing_truth}, "
            f"missing_prediction={missing_prediction}"
        )

    rows = []
    for stage, batch, length in sorted(predicted):
        estimate = predicted[(stage, batch, length)]
        observed = truth[(stage, batch, length)]
        predicted_ms = float(estimate["total_ms"])
        observed_ms = float(observed["mean_ms"])
        error_pct = 100.0 * (predicted_ms - observed_ms) / observed_ms
        cv_pct = 100.0 * observed["std_ms"] / observed_ms
        rows.append({
            "stage": stage,
            "batch_size": batch,
            "representative_length": length,
            "observed_ms": observed_ms,
            "predicted_ms": predicted_ms,
            "error_pct": error_pct,
            "absolute_error_pct": abs(error_pct),
            "ground_truth_cv_pct": cv_pct,
            "prediction_marked_extrapolated": bool(
                estimate.get("extrapolated")),
            "ground_truth_file": observed["path"],
        })

    stage_rows = {
        stage: [row for row in rows if row["stage"] == stage]
        for stage in ("prefill", "decode")
    }
    summary = {stage: summarize(items) for stage, items in stage_rows.items()}
    no_extrapolation = not any(
        row["prediction_marked_extrapolated"] for row in rows
    )
    passed = (
        no_extrapolation
        and all(value["mape_pct"] <= args.mape_threshold_pct
                for value in summary.values())
    )
    report = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_runtime_interpolation_validation",
        "dataset_role": "runtime_interpolation_validation",
        "benchmark_spec": str(args.benchmark_spec.resolve()),
        "benchmark_spec_sha256": expected_hash,
        "frozen_prediction": str(args.prediction.resolve()),
        "ground_truth_manifest": str(args.ground_truth_manifest.resolve()),
        "ground_truth_collection_policy": {
            "collector_mode": manifest.get("collector_mode"),
            "offline_engine_lifecycle": manifest.get(
                "offline_engine_lifecycle"),
            "submission_mode": manifest.get("submission_mode"),
        },
        "acceptance": {
            "mape_threshold_pct": args.mape_threshold_pct,
            "all_predictions_are_interpolation": no_extrapolation,
            "passed": passed,
        },
        "summary": summary,
        "points": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "runtime_interpolation_validation.json"
    csv_path = args.output_dir / "runtime_interpolation_points.csv"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for stage in ("prefill", "decode"):
        print(f"{stage}: MAPE={summary[stage]['mape_pct']:.2f}%")
    print(f"passed={passed}")
    print(f"report: {report_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
