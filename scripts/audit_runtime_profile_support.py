#!/usr/bin/env python3
"""Report the actual operator/collective support loaded by a runtime backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark_protocol import load_benchmark_spec  # noqa: E402
from core.operator_interpolation import operator_work  # noqa: E402
from core.qwen3_cost_model import LLMCostModel  # noqa: E402
from core.runtime_backend import load_runtime_backend  # noqa: E402


def summarize_stage(rows: List[Dict], stage: str) -> Dict:
    length_key = "prompt_length" if stage == "prefill" else "kv_length"
    operators = sorted({
        key.split("::", 1)[1]
        for row in rows for key in row if key.startswith("operator::")
    })
    operator_support = {}
    for operator in operators:
        values = sorted({
            operator_work(
                stage, operator, int(row["batch_size"]), int(row[length_key])
            )
            for row in rows if f"operator::{operator}" in row
        })
        operator_support[operator] = {
            "point_count": len(values),
            "work_min": values[0],
            "work_max": values[-1],
        }
    return {
        "measurement_count": len(rows),
        "batch_sizes": sorted({int(row["batch_size"]) for row in rows}),
        "lengths": sorted({int(row[length_key]) for row in rows}),
        "operator_support": operator_support,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-backend-config", type=Path, required=True)
    parser.add_argument("--benchmark-spec", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    backend = load_runtime_backend(args.runtime_backend_config, PROJECT_ROOT)
    manifest = backend["manifest"]
    model = LLMCostModel(
        str(PROJECT_ROOT / "data"),
        operator_profile_dir=str(backend["operator_profile_dir"]),
        collective_profile_dir=str(backend["collective_profile_dir"]),
        profile_statistic=backend["profile_statistic"],
        operator_profile_statistic=backend["operator_profile_statistic"],
        collective_profile_statistic=backend["collective_profile_statistic"],
        calibration_file=(
            str(backend["calibration_file"])
            if backend["calibration_file"] else None
        ),
        enable_e2e_compensation=backend["calibration_enabled"],
    )
    tp_size = int(manifest["tp_size"])
    stages = model.operator_profile_tables.get(tp_size, {})
    collective = model.comm_tables.get(tp_size, [])
    report = {
        "schema_version": 1,
        "experiment_type": "runtime_profile_support_audit",
        "backend_id": manifest["backend_id"],
        "hardware_id": manifest["hardware_id"],
        "tp_size": tp_size,
        "profile_paths": {
            "operator": str(backend["operator_profile_dir"]),
            "collective": str(backend["collective_profile_dir"]),
        },
        "operator": {
            stage: summarize_stage(stages.get(stage, []), stage)
            for stage in ("prefill", "decode")
        },
        "collective": {
            "all_reduce_message_sizes_mb": sorted({
                float(row["msg_size_mb"]) for row in collective
            }),
        },
    }

    if args.benchmark_spec:
        spec = load_benchmark_spec(args.benchmark_spec)
        cases = []
        for stage in ("prefill", "decode"):
            length_key = "prompt_length" if stage == "prefill" else "kv_length"
            for case in spec["workloads"][stage]:
                batch = int(case["batch_size"])
                length = int(case[length_key])
                estimate = (
                    model.predict_prefill(batch, length, tp_size)
                    if stage == "prefill"
                    else model.predict_decode(batch, length, tp_size)
                )
                cases.append({
                    "case": case["case"],
                    "stage": stage,
                    "batch_size": batch,
                    "length": length,
                    "extrapolated": bool(estimate.get("extrapolated")),
                    "warnings": estimate.get("warnings", []),
                })
        report["benchmark_spec"] = str(args.benchmark_spec)
        report["case_coverage"] = cases

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
