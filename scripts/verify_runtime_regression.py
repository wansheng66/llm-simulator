#!/usr/bin/env python3
"""Reject changes to a previously frozen hardware prediction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_points(payload: dict) -> dict:
    points = {}
    for stage in ("prefill", "decode"):
        for row in payload.get(stage, []):
            points[row["case"]] = float(row["total_ms"])
    return points


def compare(golden: dict, candidate: dict, tolerance_pct: float) -> list[dict]:
    golden_id = golden.get("meta", {}).get("gpu_type")
    candidate_id = candidate.get("meta", {}).get("gpu_type")
    if golden_id != candidate_id:
        raise ValueError(
            f"hardware differs: golden={golden_id}, candidate={candidate_id}")
    old = load_points(golden)
    new = load_points(candidate)
    if set(old) != set(new):
        raise ValueError("golden and candidate workload cases differ")
    changes = []
    for case in sorted(old):
        delta_pct = abs(new[case] / old[case] - 1.0) * 100.0
        if not math.isclose(old[case], new[case], rel_tol=0.0, abs_tol=0.0):
            changes.append({
                "case": case,
                "golden_ms": old[case],
                "candidate_ms": new[case],
                "absolute_change_pct": delta_pct,
                "within_tolerance": delta_pct <= tolerance_pct,
            })
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance-pct", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tolerance_pct < 0:
        raise SystemExit("--tolerance-pct cannot be negative")
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    changes = compare(golden, candidate, args.tolerance_pct)
    golden_backend = golden.get("meta", {}).get("runtime_backend") or {}
    candidate_backend = candidate.get("meta", {}).get("runtime_backend") or {}
    fingerprint_match = (
        golden_backend.get("fingerprints") ==
        candidate_backend.get("fingerprints"))
    passed = (fingerprint_match and
              all(row["within_tolerance"] for row in changes))
    result = {
        "schema_version": 1,
        "experiment_type": "runtime_backend_regression_check",
        "hardware_id": golden.get("meta", {}).get("gpu_type"),
        "tolerance_pct": args.tolerance_pct,
        "changed_case_count": len(changes),
        "backend_fingerprint_match": fingerprint_match,
        "passed": passed,
        "changes": changes,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    print(f"changed cases: {len(changes)}")
    print(f"passed={passed}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
