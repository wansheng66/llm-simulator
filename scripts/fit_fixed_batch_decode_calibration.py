#!/usr/bin/env python3
"""Fit Decode shape compensation from independent fixed-batch measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.qwen3_cost_model import LLMCostModel  # noqa: E402
from scripts.profiling_common import sha256_file, write_profile  # noqa: E402


def csv_ints(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path, required=True)
    parser.add_argument("--collective-profile-dir", type=Path, required=True)
    parser.add_argument("--reference-batch-size", type=float, default=4.0)
    parser.add_argument("--reference-kv-len", type=float, default=512.0)
    parser.add_argument("--forbid-batches", type=csv_ints,
                        default=csv_ints("1,4,8"))
    parser.add_argument("--forbid-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024"))
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def solve_linear_system(matrix: Sequence[Sequence[float]],
                        values: Sequence[float]) -> List[float]:
    augmented = [list(row) + [float(value)]
                 for row, value in zip(matrix, values)]
    size = len(augmented)
    for column in range(size):
        pivot = max(range(column, size),
                    key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("calibration design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column])]
    return [augmented[index][-1] for index in range(size)]


def fit_log_shape(rows: Sequence[Dict], reference_batch: float,
                  reference_kv: float) -> Dict[str, float]:
    design = [[
        1.0,
        math.log(float(row["batch_size"]) / reference_batch),
        math.log(float(row["representative_kv_length"]) / reference_kv),
    ] for row in rows]
    target = [math.log(float(row["observed_ms"]) /
                       float(row["baseline_predicted_ms"])) for row in rows]
    normal = [[sum(a[i] * a[j] for a in design)
               for j in range(3)] for i in range(3)]
    rhs = [sum(a[i] * y for a, y in zip(design, target))
           for i in range(3)]
    intercept, batch_exponent, kv_exponent = solve_linear_system(normal, rhs)
    return {
        "base_factor": math.exp(intercept),
        "batch_exponent": batch_exponent,
        "kv_exponent": kv_exponent,
    }


def mape(rows: Iterable[Dict], key: str) -> float:
    errors = [abs(float(row[key]) - float(row["observed_ms"])) /
              float(row["observed_ms"]) * 100.0 for row in rows]
    return statistics.fmean(errors)


def main() -> int:
    args = parse_args()
    model = LLMCostModel(
        str(args.data_dir),
        operator_profile_dir=str(args.operator_profile_dir),
        collective_profile_dir=str(args.collective_profile_dir),
    )
    rows_by_tp: Dict[int, List[Dict]] = {}
    identities = set()
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("configuration", {})
        if (payload.get("collector") not in {
                "vllm_fixed_batch_streaming", "vllm_fixed_batch_offline"} or
                payload.get("valid") is not True or
                config.get("stage") != "decode"):
            raise ValueError(f"not a valid Decode calibration point: {path}")
        batch = int(config["batch_size"])
        initial_length = int(config["initial_kv_length"])
        if batch in args.forbid_batches and initial_length in args.forbid_lengths:
            raise ValueError(f"calibration point overlaps the validation grid: {path}")
        decode_tokens = int(config["decode_tokens"])
        representative = round(initial_length + (decode_tokens - 1) / 2)
        tp_size = int(config["tp_size"])
        baseline = model.predict_decode(batch, representative, tp_size)
        metadata = payload.get("metadata", {})
        identities.add((
            metadata.get("model", {}).get("config_sha256"),
            metadata.get("software", {}).get("vllm"),
            config.get("dtype"),
        ))
        rows_by_tp.setdefault(tp_size, []).append({
            "source": str(path.resolve()),
            "source_sha256": sha256_file(path),
            "tp_size": tp_size,
            "batch_size": batch,
            "initial_kv_length": initial_length,
            "representative_kv_length": representative,
            "observed_ms": float(payload["summary"]["mean_ms"]),
            "baseline_predicted_ms": float(baseline["total_time_ms"]),
        })
    if len(identities) != 1 or None in next(iter(identities), ()):
        raise ValueError("calibration files do not share complete model/runtime identity")

    decode = {}
    prefill = {}
    overhead = {}
    tp_reports = {}
    for tp_size, rows in sorted(rows_by_tp.items()):
        if len({row["batch_size"] for row in rows}) < 2 or len({
                row["initial_kv_length"] for row in rows}) < 2:
            raise ValueError(f"TP{tp_size} needs at least two Batch and KV values")
        fitted = fit_log_shape(
            rows, args.reference_batch_size, args.reference_kv_len)
        entry = {
            **fitted,
            "reference_batch_size": args.reference_batch_size,
            "reference_kv_len": args.reference_kv_len,
            "batch_size_min": min(row["batch_size"] for row in rows),
            "batch_size_max": max(row["batch_size"] for row in rows),
            "kv_len_min": min(row["representative_kv_length"] for row in rows),
            "kv_len_max": max(row["representative_kv_length"] for row in rows),
            "min_factor": 0.2,
            "max_factor": 2.0,
        }
        for row in rows:
            factor = fitted["base_factor"] * (
                row["batch_size"] / args.reference_batch_size
            ) ** fitted["batch_exponent"] * (
                row["representative_kv_length"] / args.reference_kv_len
            ) ** fitted["kv_exponent"]
            row["calibration_factor"] = min(max(factor, 0.2), 2.0)
            row["calibrated_predicted_ms"] = (
                row["baseline_predicted_ms"] * row["calibration_factor"])
        decode[str(tp_size)] = entry
        prefill[str(tp_size)] = 1.0
        overhead[str(tp_size)] = 0.0
        tp_reports[str(tp_size)] = {
            "point_count": len(rows),
            "baseline_mape_pct": mape(rows, "baseline_predicted_ms"),
            "fitted_mape_pct": mape(rows, "calibrated_predicted_ms"),
            "parameters": entry,
            "points": rows,
        }
        print(f"TP{tp_size}: baseline MAPE="
              f"{tp_reports[str(tp_size)]['baseline_mape_pct']:.2f}%, "
              f"fitted MAPE={tp_reports[str(tp_size)]['fitted_mape_pct']:.2f}%")

    calibration = {
        "schema_version": 2,
        "prefill": prefill,
        "decode": decode,
        "iteration_overhead_ms": overhead,
        "metadata": {
            "calibration_role": "fixed_batch_decode",
            "method": "log-linear least squares on observed/model Decode ratio",
            "fit_dataset": "independent fixed-batch Decode calibration grid",
            "validation_policy": (
                "B1/4/8 x KV128/512/1024 validation points are forbidden"),
        },
    }
    report = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_decode_shape_calibration",
        "configuration": {
            "reference_batch_size": args.reference_batch_size,
            "reference_kv_len": args.reference_kv_len,
            "forbidden_validation_batches": sorted(args.forbid_batches),
            "forbidden_validation_lengths": sorted(args.forbid_lengths),
        },
        "runtime_identity": {
            "model_config_sha256": next(iter(identities))[0],
            "vllm_version": next(iter(identities))[1],
            "dtype": next(iter(identities))[2],
        },
        "tp": tp_reports,
    }
    write_profile(calibration, args.output_config)
    write_profile(report, args.output_report)
    print(f"calibration config: {args.output_config}")
    print(f"fit report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
