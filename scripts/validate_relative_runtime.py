#!/usr/bin/env python3
"""Validate predicted relative hardware performance against real A/B truth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark_protocol import compare_reports  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def truth_ratios(payload: dict, reference_id: str,
                 candidate_id: str) -> dict:
    stored_reference = payload.get("reference", {}).get("hardware_id")
    stored_candidate = payload.get("candidate", {}).get("hardware_id")
    same = stored_reference == reference_id and stored_candidate == candidate_id
    reverse = stored_reference == candidate_id and stored_candidate == reference_id
    if not (same or reverse):
        raise ValueError("prediction hardware IDs do not match the ground-truth pair")
    result = {}
    for point in payload.get("points", []):
        stage = point["stage"]
        batch = int(point["batch_size"])
        length = int(point["representative_length"])
        case = (f"P_B{batch}_L{length}" if stage == "prefill"
                else f"D_B{batch}_KV{length}")
        ratio = float(point["candidate_speedup_vs_reference"])
        result[case] = ratio if same else 1.0 / ratio
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--reference-prediction", type=Path, required=True)
    parser.add_argument("--candidate-prediction", type=Path, required=True)
    parser.add_argument("--max-relative-mape-pct", type=float, default=20.0)
    parser.add_argument("--max-score-error-pct", type=float, default=15.0)
    parser.add_argument("--min-rank-agreement", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ground_truth = load(args.ground_truth)
    reference = load(args.reference_prediction)
    candidate = load(args.candidate_prediction)
    reference_id = reference.get("meta", {}).get("gpu_type")
    candidate_id = candidate.get("meta", {}).get("gpu_type")
    ratios = truth_ratios(ground_truth, reference_id, candidate_id)
    result = compare_reports(candidate, reference, ratios)
    validation = result["validation"]
    required_metrics = (
        validation.get("relative_mape_pct"),
        validation.get("prefill_score_error_pct"),
        validation.get("decode_score_error_pct"),
        validation.get("rank_agreement_ratio"),
    )
    passed = all(value is not None for value in required_metrics) and all((
        result["status"] == "verified",
        validation["relative_mape_pct"] <= args.max_relative_mape_pct,
        validation["prefill_score_error_pct"] <= args.max_score_error_pct,
        validation["decode_score_error_pct"] <= args.max_score_error_pct,
        validation["rank_agreement_ratio"] >= args.min_rank_agreement,
    ))
    result.update({
        "experiment_type": "fixed_batch_relative_runtime_validation",
        "acceptance": {
            "passed": passed,
            "max_relative_mape_pct": args.max_relative_mape_pct,
            "max_score_error_pct": args.max_score_error_pct,
            "min_rank_agreement": args.min_rank_agreement,
        },
        "sources": {
            "ground_truth": str(args.ground_truth.resolve()),
            "reference_prediction": str(args.reference_prediction.resolve()),
            "candidate_prediction": str(args.candidate_prediction.resolve()),
        },
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "relative_runtime_validation.json"
    details = args.output_dir / "relative_runtime_cases.csv"
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    with details.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["points"][0]))
        writer.writeheader()
        writer.writerows(result["points"])
    print(f"relative MAPE: {validation['relative_mape_pct']:.2f}%")
    print(f"P-Score error: {validation['prefill_score_error_pct']:.2f}%")
    print(f"D-Score error: {validation['decode_score_error_pct']:.2f}%")
    print(f"rank agreement: {validation['rank_agreement_ratio']:.2%}")
    print(f"passed={passed}")
    print(f"report: {report}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
