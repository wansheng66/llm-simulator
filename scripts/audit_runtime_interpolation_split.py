#!/usr/bin/env python3
"""Prove that runtime interpolation cases are unseen interior profile shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


Shape = Tuple[str, int, int]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_shapes(payload: Dict) -> Set[Shape]:
    result: Set[Shape] = set()
    for stage in ("prefill", "decode"):
        length_key = "prompt_length" if stage == "prefill" else "kv_length"
        for row in payload.get("workloads", {}).get(stage, []):
            shape = (stage, int(row["batch_size"]), int(row[length_key]))
            if shape in result:
                raise ValueError(f"duplicate interpolation shape: {shape}")
            result.add(shape)
    return result


def operator_shapes(path: Path) -> Set[Shape]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: Set[Shape] = set()
    for row in payload.get("measurements", []):
        if row.get("status") not in (None, "success"):
            continue
        stage = row["stage"]
        shape = row["shape"]
        length_key = "prompt_length" if stage == "prefill" else "kv_length"
        result.add((stage, int(shape["batch_size"]), int(shape[length_key])))
    if not result:
        raise ValueError(f"operator profile contains no successful shapes: {path}")
    return result


def fixed_batch_shape(payload: Dict) -> Shape | None:
    config = payload.get("configuration")
    if not isinstance(config, dict) or config.get("stage") not in {
        "prefill", "decode"
    }:
        return None
    stage = config["stage"]
    length_key = "prompt_length" if stage == "prefill" else "initial_kv_length"
    if config.get("batch_size") is None or config.get(length_key) is None:
        return None
    return stage, int(config["batch_size"]), int(config[length_key])


def resolve_manifest_file(manifest_path: Path, stored: str) -> Path:
    path = Path(stored)
    if path.exists():
        return path
    local = manifest_path.parent / path.name
    if local.exists():
        return local
    raise FileNotFoundError(f"manifest artifact not found: {stored}")


def json_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.rglob("*.json"))
    else:
        yield path


def forbidden_shapes(paths: Iterable[Path]) -> tuple[Set[Shape], Dict[str, list]]:
    result: Set[Shape] = set()
    sources: Dict[str, list] = {}
    visited: Set[Path] = set()
    queue = [item for path in paths for item in json_files(path)]
    while queue:
        path = queue.pop(0).resolve()
        if path in visited:
            continue
        visited.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment_type") == "fixed_batch_suite_manifest":
            queue.extend(
                resolve_manifest_file(path, stored)
                for stored in payload.get("files", [])
            )
            continue
        shape = fixed_batch_shape(payload)
        if shape is not None:
            result.add(shape)
            sources.setdefault(str(shape), []).append(str(path))
    return result, sources


def bounds(shapes: Set[Shape]) -> Dict[str, Dict[str, int]]:
    result = {}
    for stage in ("prefill", "decode"):
        rows = [(batch, length) for item_stage, batch, length in shapes
                if item_stage == stage]
        if not rows:
            raise ValueError(f"operator profile is missing {stage} shapes")
        result[stage] = {
            "batch_min": min(row[0] for row in rows),
            "batch_max": max(row[0] for row in rows),
            "length_min": min(row[1] for row in rows),
            "length_max": max(row[1] for row in rows),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--operator-profile", type=Path, required=True)
    parser.add_argument(
        "--forbidden-data", type=Path, nargs="*", default=[],
        help="calibration and formal-validation files, manifests, or directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.benchmark_spec.read_text(encoding="utf-8"))
    if spec.get("dataset_role") != "runtime_interpolation_validation":
        raise SystemExit(
            "benchmark spec must declare dataset_role="
            "runtime_interpolation_validation"
        )
    targets = spec_shapes(spec)
    profiled = operator_shapes(args.operator_profile)
    forbidden, forbidden_sources = forbidden_shapes(args.forbidden_data)
    profile_overlap = sorted(targets & profiled)
    forbidden_overlap = sorted(targets & forbidden)
    profile_bounds = bounds(profiled)
    out_of_bounds = []
    for stage, batch, length in sorted(targets):
        limit = profile_bounds[stage]
        if not (limit["batch_min"] <= batch <= limit["batch_max"]
                and limit["length_min"] <= length <= limit["length_max"]):
            out_of_bounds.append((stage, batch, length))

    passed = not profile_overlap and not forbidden_overlap and not out_of_bounds
    report = {
        "schema_version": 1,
        "experiment_type": "runtime_interpolation_split_audit",
        "dataset_role": "runtime_interpolation_validation",
        "benchmark_spec": str(args.benchmark_spec.resolve()),
        "benchmark_spec_sha256": sha256(args.benchmark_spec),
        "operator_profile": str(args.operator_profile.resolve()),
        "forbidden_data": [str(path.resolve()) for path in args.forbidden_data],
        "target_shapes": [list(shape) for shape in sorted(targets)],
        "operator_profile_bounds": profile_bounds,
        "checks": {
            "no_operator_profile_point_overlap": not profile_overlap,
            "no_calibration_or_formal_validation_point_overlap": (
                not forbidden_overlap
            ),
            "all_targets_are_interpolation_not_extrapolation": (
                not out_of_bounds
            ),
        },
        "conflicts": {
            "operator_profile_overlap": [list(shape) for shape in profile_overlap],
            "forbidden_data_overlap": [
                {
                    "shape": list(shape),
                    "sources": forbidden_sources.get(str(shape), []),
                }
                for shape in forbidden_overlap
            ],
            "out_of_profile_bounds": [list(shape) for shape in out_of_bounds],
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"target points: {len(targets)}")
    print(f"operator overlap: {len(profile_overlap)}")
    print(f"forbidden-data overlap: {len(forbidden_overlap)}")
    print(f"out-of-bounds points: {len(out_of_bounds)}")
    print(f"passed={passed}")
    print(f"report: {args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
