#!/usr/bin/env python3
"""Build an auditable inner-domain subset of schema-v2 operator profiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement_identity(row: Dict[str, Any]) -> Tuple[str, int, int]:
    stage = str(row["stage"])
    shape = row["shape"]
    length_key = "prompt_length" if stage == "prefill" else "kv_length"
    return stage, int(shape["batch_size"]), int(shape[length_key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--holdout-output-dir", type=Path,
        help=(
            "optionally write the successful TP-matched measurements excluded "
            "from the subset as a separate operator profile"
        ),
    )
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--max-prefill-length", type=int, required=True)
    parser.add_argument("--max-decode-kv-length", type=int, required=True)
    args = parser.parse_args()

    source_files = sorted(
        args.source_dir.glob("**/operator_profile_tp*.json")
    )
    if not source_files:
        raise SystemExit(f"no operator profiles found in {args.source_dir}")

    selected: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    held_out: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    template = None
    used_sources: List[Path] = []
    source_measurement_count = 0
    for path in source_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("schema_version") != 2 or
                payload.get("profile_type") != "operator"):
            raise ValueError(f"unsupported operator profile schema: {path}")
        benchmark = payload.get("metadata", {}).get("benchmark", {})
        if int(benchmark.get("tp_size", 0)) != args.tp_size:
            continue
        if template is None:
            template = copy.deepcopy(payload)
        used_sources.append(path.resolve())
        for row in payload.get("measurements", []):
            if row.get("status") != "success":
                continue
            source_measurement_count += 1
            stage, batch, length = measurement_identity(row)
            if stage not in {"prefill", "decode"}:
                continue
            limit = (
                args.max_prefill_length if stage == "prefill"
                else args.max_decode_kv_length
            )
            identity = (stage, batch, length)
            destination = (
                selected
                if batch <= args.max_batch_size and length <= limit
                else held_out
            )
            if identity in destination:
                raise ValueError(
                    "duplicate successful source measurement for "
                    f"{identity}; isolate one training profile before subsetting"
                )
            destination[identity] = copy.deepcopy(row)

    if template is None:
        raise ValueError(f"no TP{args.tp_size} operator profile was found")
    for stage in ("prefill", "decode"):
        if not any(identity[0] == stage for identity in selected):
            raise ValueError(f"subset contains no {stage} measurements")

    ordered = [selected[key] for key in sorted(selected)]
    template["measurements"] = ordered
    benchmark = template.setdefault("metadata", {}).setdefault("benchmark", {})
    benchmark["profile_subset"] = {
        "purpose": "controlled_runtime_extrapolation_evaluation",
        "tp_size": args.tp_size,
        "max_batch_size": args.max_batch_size,
        "max_prefill_length": args.max_prefill_length,
        "max_decode_kv_length": args.max_decode_kv_length,
        "source_measurement_count": source_measurement_count,
        "selected_measurement_count": len(ordered),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"operator_profile_tp{args.tp_size}.json"
    output.write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "experiment_type": "operator_profile_subset_manifest",
        "purpose": "controlled_runtime_extrapolation_evaluation",
        "filters": {
            "tp_size": args.tp_size,
            "max_batch_size": args.max_batch_size,
            "max_prefill_length": args.max_prefill_length,
            "max_decode_kv_length": args.max_decode_kv_length,
        },
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in used_sources
        ],
        "source_measurement_count": source_measurement_count,
        "selected_measurement_count": len(ordered),
        "selected_shapes": [
            {"stage": stage, "batch_size": batch, "length": length}
            for stage, batch, length in sorted(selected)
        ],
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    manifest_path = args.output_dir / "profile_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    holdout_path = None
    if args.holdout_output_dir:
        args.holdout_output_dir.mkdir(parents=True, exist_ok=True)
        holdout_payload = copy.deepcopy(template)
        holdout_payload["measurements"] = [
            held_out[key] for key in sorted(held_out)
        ]
        holdout_benchmark = (
            holdout_payload.setdefault("metadata", {})
            .setdefault("benchmark", {})
        )
        holdout_benchmark["profile_subset"] = {
            "purpose": "controlled_runtime_extrapolation_holdout_complement",
            "tp_size": args.tp_size,
            "excluded_from_training_subset": True,
            "measurement_count": len(held_out),
        }
        holdout_path = (
            args.holdout_output_dir /
            f"operator_profile_tp{args.tp_size}.json"
        )
        holdout_path.write_text(
            json.dumps(holdout_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"selected {len(ordered)}/{source_measurement_count} TP{args.tp_size} "
        f"measurements; profile={output}"
    )
    print(f"manifest: {manifest_path}")
    if holdout_path:
        print(
            f"holdout complement: {len(held_out)} measurements; "
            f"profile={holdout_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
