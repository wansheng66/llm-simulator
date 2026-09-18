#!/usr/bin/env python3
"""Compare TP1/2/4/8 fixed-batch vLLM measurements with the cost model."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.qwen3_cost_model import LLMCostModel  # noqa: E402
from scripts.profiling_common import dominant_component, write_profile  # noqa: E402
from scripts.calibration_roles import (  # noqa: E402
    FIXED_BATCH_DECODE,
    require_calibration_role,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path)
    parser.add_argument("--collective-profile-dir", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    parser.add_argument("--expected-tp", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--infeasible-tp", action="append", default=[],
                        help="Document a justified skip as TP=reason")
    parser.add_argument("--mape-threshold-pct", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_measurement(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"unsupported fixed-batch schema: {path}")
    if payload.get("collector") not in {
            "vllm_fixed_batch_streaming", "vllm_fixed_batch_offline"}:
        raise ValueError(f"unsupported collector: {path}")
    if not payload.get("valid") or not payload.get("summary"):
        raise ValueError(f"invalid fixed-batch measurement: {path}")
    payload["_source"] = str(path.resolve())
    return payload


def parse_infeasible(values: Sequence[str]) -> Dict[int, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--infeasible-tp must use TP=reason")
        tp, reason = value.split("=", 1)
        result[int(tp)] = reason.strip()
    return result


def metadata_audit(payload: Dict) -> Dict:
    metadata = payload.get("metadata", {})
    software = metadata.get("software", {})
    benchmark = metadata.get("benchmark", {})
    required = {
        "gpu_inventory": metadata.get("cuda", {}).get("nvidia_smi_query"),
        "torch_version": software.get("torch"),
        "vllm_version": software.get("vllm"),
        "model_config_sha256": metadata.get("model", {}).get("config_sha256"),
        "dtype": benchmark.get("dtype") or payload.get("deployment", {}).get("config", {}).get("dtype"),
        "warmup_batches": benchmark.get("warmup_batches"),
        "measured_batches": benchmark.get("measured_batches"),
        "server_launch_configuration": (
            payload.get("deployment", {}).get("config") or
            payload.get("deployment", {}).get("server_command")),
    }
    missing = [key for key, value in required.items() if value in (None, "", {})]
    return {"complete": not missing, "missing": missing, "fields": required}


def summarize(rows: Sequence[Dict]) -> Dict:
    errors = [float(row["error_pct"]) for row in rows]
    absolute = [abs(value) for value in errors]
    return {
        "count": len(rows),
        "mape_pct": statistics.fmean(absolute),
        "bias_pct": statistics.fmean(errors),
        "max_absolute_error_pct": max(absolute),
        "bottleneck_counts": {
            name: sum(row["estimated_bottleneck"] == name for row in rows)
            for name in sorted({row["estimated_bottleneck"] for row in rows})
        },
    }


def main() -> int:
    args = parse_args()
    calibration_role = None
    if args.enable_calibration:
        if args.calibration_file is None:
            raise SystemExit(
                "--enable-calibration requires --calibration-file")
        calibration_role = require_calibration_role(
            args.calibration_file, FIXED_BATCH_DECODE)
    infeasible = parse_infeasible(args.infeasible_tp)
    payloads = [load_measurement(path) for path in args.inputs]
    model = LLMCostModel(
        str(args.data_dir),
        operator_profile_dir=(str(args.operator_profile_dir)
                              if args.operator_profile_dir else None),
        collective_profile_dir=(str(args.collective_profile_dir)
                                if args.collective_profile_dir else None),
        calibration_file=(str(args.calibration_file) if args.calibration_file else None),
        enable_e2e_compensation=args.enable_calibration,
    )
    points: List[Dict] = []
    audits = []
    for payload in payloads:
        config = payload["configuration"]
        stage = config["stage"]
        tp_size = int(config["tp_size"])
        batch = int(config["batch_size"])
        if stage == "prefill":
            length = int(config["prompt_length"])
            prediction = model.predict_prefill(batch, length, tp_size)
            representative_length = length
        else:
            initial = int(config["initial_kv_length"])
            decode_tokens = int(config["decode_tokens"])
            representative_length = round(initial + (decode_tokens - 1) / 2)
            prediction = model.predict_decode(batch, representative_length, tp_size)
        observed = float(payload["summary"]["mean_ms"])
        predicted = float(prediction["total_time_ms"])
        residual = observed - predicted
        breakdown = prediction["operator_breakdown"]
        attribution = {
            "attention_ms": float(breakdown["attention_ms"]),
            "ffn_ms": float(breakdown["ffn_ms"]),
            "collective_ms": float(breakdown["collective_ms"]),
            "unmodeled_runtime_residual_ms": residual,
        }
        nonnegative = {key.removesuffix("_ms"): max(value, 0.0)
                       for key, value in attribution.items()}
        point = {
            "source": payload["_source"],
            "stage": stage,
            "tp_size": tp_size,
            "batch_size": batch,
            "representative_length": representative_length,
            "observed_iteration_ms": observed,
            "predicted_iteration_ms": predicted,
            "absolute_error_ms": abs(residual),
            "error_pct": 100.0 * (predicted - observed) / observed if observed else 0.0,
            "extrapolated": prediction["extrapolated"],
            "profile_source": prediction["profile_source"],
            **attribution,
            "estimated_bottleneck": dominant_component(nonnegative),
        }
        points.append(point)
        audit = metadata_audit(payload)
        audit["source"] = payload["_source"]
        audits.append(audit)

    coverage = {}
    for tp in args.expected_tp:
        stages = sorted({row["stage"] for row in points if row["tp_size"] == tp})
        if set(stages) == {"decode", "prefill"}:
            status, reason = "measured", None
        elif tp in infeasible:
            status, reason = "infeasible", infeasible[tp]
        else:
            status, reason = "missing", "both Prefill and Decode measurements are required"
        coverage[str(tp)] = {"status": status, "stages": stages, "reason": reason}

    consistency = {}
    for tp in args.expected_tp:
        tp_payloads = [payload for payload in payloads
                       if int(payload["configuration"]["tp_size"]) == tp]
        if not tp_payloads:
            continue
        fields = {
            "deployment_config_sha256": {
                payload.get("deployment", {}).get("config_sha256")
                for payload in tp_payloads},
            "model_config_sha256": {
                payload.get("metadata", {}).get("model", {}).get("config_sha256")
                for payload in tp_payloads},
            "vllm_version": {
                payload.get("metadata", {}).get("software", {}).get("vllm")
                for payload in tp_payloads},
            "dtype": {payload.get("configuration", {}).get("dtype")
                      for payload in tp_payloads},
        }
        conflicts = {key: sorted(str(value) for value in values)
                     for key, values in fields.items() if len(values) != 1 or None in values}
        consistency[str(tp)] = {"consistent": not conflicts, "conflicts": conflicts}

    summaries = {}
    for tp in args.expected_tp:
        tp_rows = [row for row in points if row["tp_size"] == tp]
        if not tp_rows:
            continue
        summaries[str(tp)] = {
            stage: summarize([row for row in tp_rows if row["stage"] == stage])
            for stage in ("prefill", "decode")
            if any(row["stage"] == stage for row in tp_rows)
        }
    errors_pass = all(
        stats["mape_pct"] <= args.mape_threshold_pct
        for stages in summaries.values() for stats in stages.values())
    coverage_complete = all(item["status"] in ("measured", "infeasible")
                            for item in coverage.values())
    metadata_complete = (all(audit["complete"] for audit in audits) and
                         all(item["consistent"] for item in consistency.values()))
    passed = errors_pass and coverage_complete and metadata_complete
    report = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_tp_matrix_validation",
        "configuration": {
            "expected_tp": args.expected_tp,
            "data_dir": str(args.data_dir.resolve()),
            "operator_profile_dir": (str(args.operator_profile_dir.resolve())
                                     if args.operator_profile_dir else None),
            "collective_profile_dir": (str(args.collective_profile_dir.resolve())
                                       if args.collective_profile_dir else None),
            "calibration_enabled": args.enable_calibration,
            "calibration_file": (str(args.calibration_file.resolve())
                                 if args.calibration_file else None),
            "calibration_role": calibration_role,
        },
        "acceptance": {
            "mape_threshold_pct": args.mape_threshold_pct,
            "errors_pass": errors_pass,
            "coverage_complete": coverage_complete,
            "metadata_complete": metadata_complete,
            "passed": passed,
        },
        "tp_coverage": coverage,
        "metadata_audit": audits,
        "metadata_consistency_by_tp": consistency,
        "summary": summaries,
        "points": points,
        "attribution_policy": {
            "attention_ffn": "TP-sharded operator profile when available, otherwise legacy table",
            "collective": "measured AllReduce lookup used by LLMCostModel",
            "unmodeled_runtime_residual": "observed fixed-batch iteration minus modeled total",
            "negative_residual": "model overprediction; excluded from bottleneck selection",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_profile(report, args.output_dir / "fixed_batch_validation_report.json")
    with (args.output_dir / "fixed_batch_validation_points.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    for tp, stages in summaries.items():
        for stage, stats in stages.items():
            print(f"TP{tp} {stage}: MAPE={stats['mape_pct']:.2f}%")
    print(f"passed={passed}; report={args.output_dir / 'fixed_batch_validation_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
