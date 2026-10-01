#!/usr/bin/env python3
"""Freeze an accepted relative benchmark into a reproducible release record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_source(stored: str, owner: Path) -> Path:
    path = Path(stored)
    if path.exists():
        return path.resolve()
    local = owner.parent / path.name
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(f"source artifact not found: {stored}")


def artifact(label: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "label": label,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def require_accepted(validation: dict[str, Any], truth: dict[str, Any]) -> None:
    if validation.get("experiment_type") != "fixed_batch_relative_runtime_validation":
        raise ValueError("validation report has the wrong experiment_type")
    if validation.get("status") != "verified":
        raise ValueError("runtime validation must have status=verified")
    if validation.get("acceptance", {}).get("passed") is not True:
        raise ValueError("runtime validation has not passed acceptance")
    if truth.get("experiment_type") != "fixed_batch_relative_ground_truth":
        raise ValueError("ground truth has the wrong experiment_type")
    if truth.get("acceptance", {}).get("passed") is not True:
        raise ValueError("ground-truth comparison has not passed acceptance")


def build_release(validation_path: Path, name: str) -> dict[str, Any]:
    validation_path = validation_path.resolve()
    validation = load_json(validation_path)
    sources = validation.get("sources", {})
    truth_path = resolve_source(sources["ground_truth"], validation_path)
    reference_prediction = resolve_source(
        sources["reference_prediction"], validation_path)
    candidate_prediction = resolve_source(
        sources["candidate_prediction"], validation_path)
    truth = load_json(truth_path)
    require_accepted(validation, truth)

    reference_manifest = resolve_source(
        truth["reference"]["manifest"], truth_path)
    candidate_manifest = resolve_source(
        truth["candidate"]["manifest"], truth_path)
    reference_manifest_payload = load_json(reference_manifest)
    candidate_manifest_payload = load_json(candidate_manifest)

    measured_files: list[dict[str, Any]] = []
    for side, manifest_path, manifest in (
        ("reference", reference_manifest, reference_manifest_payload),
        ("candidate", candidate_manifest, candidate_manifest_payload),
    ):
        for index, stored in enumerate(manifest.get("files", []), start=1):
            measured_files.append(artifact(
                f"{side}_ground_truth_point_{index:02d}",
                resolve_source(stored, manifest_path),
            ))

    scores = validation["scores"]
    metrics = validation["validation"]
    points = validation.get("points", [])
    runtime_artifacts = [
        artifact("runtime_validation", validation_path),
        artifact("relative_ground_truth", truth_path),
        artifact("reference_prediction", reference_prediction),
        artifact("candidate_prediction", candidate_prediction),
        artifact("reference_manifest", reference_manifest),
        artifact("candidate_manifest", candidate_manifest),
    ]

    return {
        "schema_version": 1,
        "experiment_type": "relative_benchmark_release",
        "name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "comparison": {
            "reference": validation["reference"],
            "candidate": validation["candidate"],
            "protocol_version": validation.get("protocol_version"),
            "comparison_track": validation.get("comparison_track"),
        },
        "fairness": {
            "identity": truth.get("identity"),
            "runtime": truth.get("runtime"),
            "measurement_policy": truth.get("measurement_policy"),
            "paired_point_count": truth.get("acceptance", {}).get(
                "paired_point_count"),
            "ground_truth_accepted": True,
            "runtime_validation_accepted": True,
            "all_points_have_ground_truth": metrics.get(
                "ground_truth_complete"),
        },
        "results": {
            "ground_truth_p_score": metrics.get(
                "ground_truth_prefill_speedup"),
            "predicted_p_score": scores.get("prefill_speedup"),
            "p_score_error_pct": metrics.get("prefill_score_error_pct"),
            "ground_truth_d_score": metrics.get(
                "ground_truth_decode_speedup"),
            "predicted_d_score": scores.get("decode_speedup"),
            "d_score_error_pct": metrics.get("decode_score_error_pct"),
            "relative_mape_pct": metrics.get("relative_mape_pct"),
            "rank_agreement_ratio": metrics.get("rank_agreement_ratio"),
            "paired_prefill_cases": scores.get("paired_prefill_cases"),
            "paired_decode_cases": scores.get("paired_decode_cases"),
        },
        "artifacts": runtime_artifacts,
        "ground_truth_files": measured_files,
        "points": points,
    }


def markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    fairness = report["fairness"]
    result = report["results"]
    reference = comparison["reference"]["gpu_type"]
    candidate = comparison["candidate"]["gpu_type"]
    rank = result["rank_agreement_ratio"]
    return f"""# {report['name']}

状态：`{report['status']}`  
对比方向：`{candidate} / {reference}`  
协议版本：`{comparison['protocol_version']}`

## 公平性检查

- Ground Truth：严格离线固定 Batch，已通过验收
- 提交方式：`{fairness['measurement_policy'].get('submission_mode')}`
- TP：{fairness['identity'].get('tp_size')}
- 精度：`{fairness['identity'].get('dtype')}`
- 配对 Case：{fairness['paired_point_count']}
- Ground Truth 完整：{fairness['all_points_have_ground_truth']}

## 最终结果

| 指标 | 真实值 | 预测值 | 误差 |
|---|---:|---:|---:|
| P-Score | {result['ground_truth_p_score']:.10f}x | {result['predicted_p_score']:.10f}x | {result['p_score_error_pct']:.2f}% |
| D-Score | {result['ground_truth_d_score']:.10f}x | {result['predicted_d_score']:.10f}x | {result['d_score_error_pct']:.2f}% |

- 18 点相对 MAPE：{result['relative_mape_pct']:.2f}%
- 排序一致率：{rank:.2%}
- Prefill Case：{result['paired_prefill_cases']}
- Decode Case：{result['paired_decode_cases']}

## 数据溯源

`benchmark_release.json` 保存了验证报告、预测文件、两份 manifest 和全部 Ground Truth 点的路径、大小与 SHA256。任何文件变化都会导致指纹变化。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--name", default="Qwen3-32B TP4 relative benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_release(args.validation_report, args.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_release.json"
    md_path = args.output_dir / "benchmark_release.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"P-Score: {report['results']['ground_truth_p_score']:.4f}x real, "
          f"{report['results']['predicted_p_score']:.4f}x predicted")
    print(f"D-Score: {report['results']['ground_truth_d_score']:.4f}x real, "
          f"{report['results']['predicted_d_score']:.4f}x predicted")
    print(f"release: {json_path}")
    print(f"summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
