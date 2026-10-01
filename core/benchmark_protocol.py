"""Versioned workload and relative-score rules for hardware benchmarks.

The central rule is that an iso-workload score only compares identical work.
Memory-adjusted cases belong to a separate best-feasible track and must never
silently enter the relative hardware ranking.
"""

from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROTOCOL_VERSION = "0.1"


def load_benchmark_spec(path: str | Path) -> Dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported benchmark spec schema_version")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported benchmark protocol_version")
    if not payload.get("model", {}).get("name"):
        raise ValueError("benchmark spec requires model.name")
    for stage in ("prefill", "decode"):
        cases = payload.get("workloads", {}).get(stage, [])
        if not cases:
            raise ValueError(f"benchmark spec requires {stage} workloads")
        for case in cases:
            required = {"case", "batch_size"}
            required.add("prompt_length" if stage == "prefill" else "kv_length")
            missing = required - set(case)
            if missing:
                raise ValueError(f"{stage} case is missing {sorted(missing)}")
    return payload


def attention_type(num_attention_heads: int, num_kv_heads: int) -> str:
    if num_attention_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("attention head counts must be positive")
    if num_attention_heads == num_kv_heads:
        return "MHA"
    if num_kv_heads == 1:
        return "MQA"
    if num_attention_heads % num_kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    return "GQA"


def adapt_legacy_report(report: Dict, spec: Dict) -> Dict:
    """Return a non-mutating protocol-0.1 view of a legacy report.

    This repairs the old batch-token throughput formulas, but intentionally
    marks the provenance as adapted and therefore never upgrades trust beyond
    ``provisional``.
    """
    payload = copy.deepcopy(report)
    meta = payload.setdefault("meta", {})
    if meta.get("protocol_version") == PROTOCOL_VERSION:
        return payload
    meta.update({
        "protocol_version": PROTOCOL_VERSION,
        "model_id": spec["model"]["model_id"],
        "model_name": spec["model"]["name"],
        "dtype": spec["model"]["dtype"],
        "comparison_status": "provisional",
        "report_provenance": "legacy_v0_adapted_in_memory",
    })
    for stage in ("prefill", "decode"):
        for item in payload.get(stage, []):
            total_ms = float(item.get("total_ms", 0))
            batch = int(item.get("batch", 0))
            if total_ms <= 0 or batch <= 0:
                continue
            token_count = batch * int(item.get("seq", 0)) if stage == "prefill" else batch
            item["throughput_tok_s"] = token_count * 1000.0 / total_ms
    return payload


def _case_signature(stage: str, item: Dict) -> Tuple:
    length_key = "seq" if stage == "prefill" else "kv"
    return (
        stage,
        item.get("case"),
        int(item.get("batch", 0)),
        int(item.get(length_key, 0)),
    )


def _valid_iso_case(stage: str, item: Dict) -> bool:
    if item.get("memory_oom"):
        return False
    original = item.get("original_batch", item.get("batch"))
    return int(item.get("batch", 0)) == int(original or 0) and (
        int(item.get("seq" if stage == "prefill" else "kv", 0)) > 0
    )


def geometric_mean(values: Iterable[float]) -> Optional[float]:
    values = [float(value) for value in values if value is not None and value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare_reports(candidate: Dict, reference: Dict,
                    ground_truth_ratios: Optional[Dict[str, float]] = None) -> Dict:
    """Compare reports using paired, identical iso-workload cases only.

    Ratios are candidate throughput / reference throughput.  A ratio greater
    than one means the candidate is faster.  Real A/B ratios are optional; in
    their absence the output is explicitly provisional.
    """
    ground_truth_ratios = ground_truth_ratios or {}
    candidate_meta = candidate.get("meta", {})
    reference_meta = reference.get("meta", {})
    same_protocol = (
        candidate_meta.get("protocol_version") == reference_meta.get("protocol_version")
        == PROTOCOL_VERSION
    )
    same_model = candidate_meta.get("model_id") == reference_meta.get("model_id")
    same_tp = candidate_meta.get("tp") == reference_meta.get("tp")
    candidate_track = candidate_meta.get(
        "comparison_track", "iso_logical_tp")
    reference_track = reference_meta.get(
        "comparison_track", "iso_logical_tp")
    same_track = candidate_track == reference_track
    comparison_track = (
        candidate_track if same_track else "incompatible")
    candidate_physical = candidate_meta.get("physical_accelerators")
    reference_physical = reference_meta.get("physical_accelerators")
    same_physical = (
        candidate_physical is not None
        and reference_physical is not None
        and int(candidate_physical) == int(reference_physical)
    )
    compatibility_warnings: List[str] = []
    if not same_protocol:
        compatibility_warnings.append("reports do not both declare protocol 0.1")
    if not same_model:
        compatibility_warnings.append("model_id differs or is missing")
    if not same_track:
        compatibility_warnings.append("comparison tracks differ")
    if comparison_track == "iso_physical_accelerators":
        if not same_physical:
            compatibility_warnings.append(
                "physical accelerator counts differ or are missing")
    elif not same_tp:
        compatibility_warnings.append(
            "TP differs; iso-logical-TP comparison is invalid")

    points = []
    stage_ratios: Dict[str, List[float]] = {"prefill": [], "decode": []}
    for stage in ("prefill", "decode"):
        reference_by_signature = {
            _case_signature(stage, item): item
            for item in reference.get(stage, []) if _valid_iso_case(stage, item)
        }
        for item in candidate.get(stage, []):
            signature = _case_signature(stage, item)
            other = reference_by_signature.get(signature)
            if other is None or not _valid_iso_case(stage, item):
                continue
            candidate_rate = float(item.get("throughput_tok_s", 0))
            reference_rate = float(other.get("throughput_tok_s", 0))
            if candidate_rate <= 0 or reference_rate <= 0:
                continue
            ratio = candidate_rate / reference_rate
            truth = ground_truth_ratios.get(item["case"])
            relative_error = None if truth is None else abs(ratio / truth - 1.0) * 100.0
            points.append({
                "stage": stage,
                "case": item["case"],
                "batch_size": item["batch"],
                "representative_length": item["seq" if stage == "prefill" else "kv"],
                "candidate_throughput_tok_s": candidate_rate,
                "reference_throughput_tok_s": reference_rate,
                "speedup": ratio,
                "ground_truth_speedup": truth,
                "relative_error_pct": relative_error,
            })
            stage_ratios[stage].append(ratio)

    resource_compatible = (
        same_physical if comparison_track == "iso_physical_accelerators"
        else same_tp
    )
    compatible = (
        same_protocol and same_model and same_track and resource_compatible)
    has_truth = bool(points) and all(point["ground_truth_speedup"] is not None for point in points)
    status = "invalid" if not compatible or not points else ("verified" if has_truth else "provisional")
    measured_errors = [point["relative_error_pct"] for point in points
                       if point["relative_error_pct"] is not None]
    rank_matches = [
        (point["speedup"] >= 1.0) == (point["ground_truth_speedup"] >= 1.0)
        for point in points if point["ground_truth_speedup"] is not None
    ]
    stage_truth = {
        stage: geometric_mean(
            point["ground_truth_speedup"] for point in points
            if point["stage"] == stage and point["ground_truth_speedup"] is not None)
        for stage in ("prefill", "decode")
    }
    stage_prediction = {
        stage: geometric_mean(
            point["speedup"] for point in points if point["stage"] == stage)
        for stage in ("prefill", "decode")
    }
    stage_error = {
        stage: (abs(stage_prediction[stage] / stage_truth[stage] - 1.0) * 100.0
                if stage_prediction[stage] and stage_truth[stage] else None)
        for stage in ("prefill", "decode")
    }
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "comparison_track": comparison_track,
        "status": status,
        "status_explanation": {
            "verified": "all paired cases include independent real A/B ground truth",
            "provisional": "paired cases are comparable, but independent real A/B ratios are absent",
            "invalid": "reports are incompatible or contain no identical feasible cases",
        }[status],
        "candidate": {
            "gpu_type": candidate_meta.get("gpu_type", "Unknown"),
            "tp": candidate_meta.get("tp"),
            "physical_accelerators": candidate_physical,
        },
        "reference": {
            "gpu_type": reference_meta.get("gpu_type", "Unknown"),
            "tp": reference_meta.get("tp"),
            "physical_accelerators": reference_physical,
        },
        "compatibility": {
            "same_protocol": same_protocol,
            "same_model": same_model,
            "same_comparison_track": same_track,
            "same_tp": same_tp,
            "same_physical_accelerator_count": same_physical,
            "warnings": compatibility_warnings,
        },
        "scores": {
            "prefill_speedup": geometric_mean(stage_ratios["prefill"]),
            "decode_speedup": geometric_mean(stage_ratios["decode"]),
            "paired_prefill_cases": len(stage_ratios["prefill"]),
            "paired_decode_cases": len(stage_ratios["decode"]),
        },
        "validation": {
            "ground_truth_complete": has_truth,
            "relative_mape_pct": (sum(measured_errors) / len(measured_errors)
                                   if measured_errors else None),
            "rank_agreement_ratio": (sum(rank_matches) / len(rank_matches)
                                     if rank_matches else None),
            "ground_truth_prefill_speedup": stage_truth["prefill"],
            "ground_truth_decode_speedup": stage_truth["decode"],
            "prefill_score_error_pct": stage_error["prefill"],
            "decode_score_error_pct": stage_error["decode"],
        },
        "points": points,
    }
