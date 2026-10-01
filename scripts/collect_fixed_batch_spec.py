#!/usr/bin/env python3
"""Collect an exact BenchmarkSpec workload with the generic vLLM backend.

This is the NVIDIA/general-purpose companion to the Ascend-specific exact
collector.  It preserves the existing fixed-batch collector and merely passes
an explicit, non-Cartesian case list to it.  Prefill and Decode are collected
in separate shared-engine runs because their scheduler token budgets can
differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_COLLECTOR = PROJECT_ROOT / "scripts" / "collect_fixed_batch_suite.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option_value(argv: List[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cases(spec_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported benchmark spec schema_version")
    if spec.get("dataset_role") != "runtime_extrapolation_validation":
        raise ValueError(
            "exact-spec collector requires dataset_role="
            "runtime_extrapolation_validation"
        )
    cases: List[Dict[str, Any]] = []
    for stage in ("prefill", "decode"):
        length_key = "prompt_length" if stage == "prefill" else "kv_length"
        for item in spec.get("workloads", {}).get(stage, []):
            cases.append({
                "stage": stage,
                "case": str(item["case"]),
                "batch_size": int(item["batch_size"]),
                "length": int(item[length_key]),
                "length_key": length_key,
                "extrapolation_axis": item.get("extrapolation_axis"),
            })
    if not cases:
        raise ValueError("benchmark spec contains no workloads")
    identities = {
        (case["stage"], case["batch_size"], case["length"])
        for case in cases
    }
    if len(identities) != len(cases):
        raise ValueError("benchmark spec contains duplicate workload shapes")
    return spec, cases


def validate_stage_config(
    spec: Dict[str, Any], cases: List[Dict[str, Any]], stage: str,
    config_path: Path, expected_tp: int,
) -> Dict[str, Any]:
    deployment = json.loads(config_path.read_text(encoding="utf-8"))
    configured_tp = deployment.get(
        "tensor_parallel_size", deployment.get("tp_size")
    )
    if configured_tp is not None and int(configured_tp) != expected_tp:
        raise ValueError(
            f"{stage} deployment TP={configured_tp}, expected {expected_tp}"
        )
    if deployment.get("enable_prefix_caching") is True:
        raise ValueError(f"{stage} deployment must disable prefix caching")
    if deployment.get("enable_chunked_prefill") is True:
        raise ValueError(f"{stage} deployment must disable chunked prefill")

    required = spec.get("deployment", {})
    required_sequences = int(required["max_num_seqs"])
    configured_sequences = int(deployment.get("max_num_seqs", 0))
    if configured_sequences != required_sequences:
        raise ValueError(
            f"{stage} max_num_seqs={configured_sequences}, "
            f"BenchmarkSpec requires {required_sequences}"
        )

    budgets = spec.get("measurement", {}).get(
        "collection_max_num_batched_tokens", {}
    )
    required_budget = int(
        budgets.get(stage, required["max_num_batched_tokens"])
    )
    configured_budget = int(deployment.get("max_num_batched_tokens", 0))
    if configured_budget != required_budget:
        raise ValueError(
            f"{stage} max_num_batched_tokens={configured_budget}, "
            f"BenchmarkSpec collection requires {required_budget}"
        )

    max_model_len = int(deployment.get("max_model_len", 0))
    failures = []
    for case in cases:
        initialization_tokens = case["batch_size"] * case["length"]
        if case["batch_size"] > configured_sequences:
            failures.append(f"{case['case']} exceeds max_num_seqs")
        if initialization_tokens > configured_budget:
            failures.append(
                f"{case['case']} needs {initialization_tokens} batched tokens"
            )
        if case["length"] > max_model_len:
            failures.append(f"{case['case']} exceeds max_model_len")
    if failures:
        raise ValueError("; ".join(failures))
    return deployment


def stage_command(
    forwarded: List[str], cases: List[Dict[str, Any]], stage: str,
    config_path: Path, output_dir: Path,
) -> List[str]:
    exact_args: List[str] = []
    for case in cases:
        exact_args.extend([
            "--exact-case",
            f"{stage}:{case['batch_size']}:{case['length']}",
        ])
    return [
        sys.executable,
        str(SUITE_COLLECTOR),
        *forwarded,
        "--collector-mode", "offline",
        "--offline-engine-lifecycle", "suite",
        "--dataset-role", "runtime_extrapolation_validation",
        "--deployment-config", str(config_path),
        *exact_args,
        "--output-dir", str(output_dir),
    ]


def _resolve_artifact(source: str, manifest_path: Path) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    beside_manifest = manifest_path.parent / path
    return beside_manifest if beside_manifest.exists() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Remaining arguments (model, tokenizer, TP, repeats, dtype and "
            "metadata) are forwarded to collect_fixed_batch_suite.py."
        ),
    )
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--prefill-deployment-config", type=Path, required=True)
    parser.add_argument("--decode-deployment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, forwarded = parser.parse_known_args()

    forbidden = {
        "--collector-mode", "--offline-engine-lifecycle",
        "--offline-cooldown-seconds", "--dataset-role",
        "--deployment-config", "--output-dir", "--exact-case",
        "--stages", "--batch-sizes", "--prefill-lengths",
        "--decode-kv-lengths", "--base-url", "--api-key",
        "--submission-mode",
    }
    conflicts = sorted(
        name for name in forbidden
        if any(value == name or value.startswith(name + "=")
               for value in forwarded)
    )
    if conflicts:
        raise SystemExit(
            "collection mode, shapes and output come from this wrapper/spec: "
            + ", ".join(conflicts)
        )

    tp_text = option_value(forwarded, "--tp-size")
    if tp_text is None:
        raise SystemExit("forwarded collector arguments require --tp-size")
    expected_tp = int(tp_text)
    spec_path = resolve_project_path(args.benchmark_spec).resolve()
    spec, cases = load_cases(spec_path)
    output_root = resolve_project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config_paths = {
        "prefill": resolve_project_path(args.prefill_deployment_config).resolve(),
        "decode": resolve_project_path(args.decode_deployment_config).resolve(),
    }
    stage_cases = {
        stage: [case for case in cases if case["stage"] == stage]
        for stage in ("prefill", "decode")
    }
    deployments = {
        stage: validate_stage_config(
            spec, stage_cases[stage], stage, config_paths[stage], expected_tp
        )
        for stage in ("prefill", "decode")
    }

    stage_manifests: Dict[str, Path] = {}
    point_by_shape = {
        (case["stage"], case["batch_size"], case["length"]): case
        for case in cases
    }
    files: List[str] = []
    points: List[Dict[str, Any]] = []
    for stage in ("prefill", "decode"):
        stage_dir = output_root / ".stage_suites" / stage
        print(
            f"collect {len(stage_cases[stage])} exact {stage} cases "
            "with one shared engine",
            flush=True,
        )
        subprocess.run(
            stage_command(
                forwarded, stage_cases[stage], stage,
                config_paths[stage], stage_dir,
            ),
            cwd=PROJECT_ROOT,
            check=True,
        )
        manifest_path = stage_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_manifests[stage] = manifest_path.resolve()
        for source in manifest.get("files", []):
            artifact = _resolve_artifact(str(source), manifest_path).resolve()
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if (payload.get("valid") is not True or
                    payload.get("fixed_batch_valid") is not True):
                raise ValueError(f"invalid exact case artifact: {artifact}")
            configuration = payload.get("configuration", {})
            length_key = (
                "prompt_length" if stage == "prefill"
                else "initial_kv_length"
            )
            identity = (
                stage,
                int(configuration.get("batch_size", 0)),
                int(configuration.get(length_key, 0)),
            )
            case = point_by_shape.get(identity)
            if case is None:
                raise ValueError(f"unexpected collected shape: {identity}")
            files.append(str(artifact))
            points.append({
                **case,
                "file": str(artifact),
                "file_sha256": sha256(artifact),
                "stage_manifest": str(manifest_path.resolve()),
                "deployment_config_path": str(config_paths[stage]),
                "deployment_config_sha256": sha256(config_paths[stage]),
                "max_num_batched_tokens": deployments[stage][
                    "max_num_batched_tokens"
                ],
            })

    if len(points) != len(cases):
        raise ValueError(f"expected {len(cases)} points, collected {len(points)}")
    manifest = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_suite_manifest",
        "dataset_role": spec["dataset_role"],
        "collector_mode": "offline",
        "offline_engine_lifecycle": "shared_engine_per_stage",
        "submission_mode": "offline_enqueue_barrier",
        "tp_size": expected_tp,
        "benchmark_spec": str(spec_path),
        "benchmark_spec_sha256": sha256(spec_path),
        "stage_manifests": {
            stage: str(path) for stage, path in stage_manifests.items()
        },
        "deployment_configs": {
            stage: {
                "path": str(config_paths[stage]),
                "sha256": sha256(config_paths[stage]),
                "max_num_batched_tokens": deployments[stage][
                    "max_num_batched_tokens"
                ],
            }
            for stage in ("prefill", "decode")
        },
        "files": files,
        "points": points,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: collected {len(files)} exact cases with two engine runs; "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
