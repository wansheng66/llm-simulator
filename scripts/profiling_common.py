"""Shared schemas, statistics and environment capture for hardware profiling.

The helpers in this module intentionally use only the Python standard library.
GPU-specific benchmark scripts import torch themselves, which keeps report
validation and unit tests runnable on machines without CUDA.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 2
SAFE_ENV_PREFIXES = (
    "CUDA_", "NCCL_", "VLLM_", "TORCH_",
    "ASCEND_", "HCCL_", "NPU_", "PYTORCH_NPU_",
)
SENSITIVE_ENV_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe_ms(values: Sequence[float]) -> Dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("at least one timing sample is required")
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 0.50),
        "p90_ms": percentile(samples, 0.90),
        "p99_ms": percentile(samples, 0.99),
        "max_ms": max(samples),
    }


def dominant_component(components: Mapping[str, float], balanced_ratio: float = 1.15) -> str:
    positive = {name: max(float(value), 0.0) for name, value in components.items()}
    if not positive or max(positive.values(), default=0.0) <= 0:
        return "unknown"
    ordered = sorted(positive.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[0][1] < ordered[1][1] * balanced_ratio:
        return "balanced"
    return ordered[0][0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(command: Sequence[str], timeout: float = 10.0) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command), check=False, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace")
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_runtime() -> Dict:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    result = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_build_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["visible_device_count"] = torch.cuda.device_count()
        result["visible_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        try:
            result["nccl_version"] = torch.cuda.nccl.version()
        except Exception:
            result["nccl_version"] = None
        try:
            result["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            result["cudnn_version"] = None
    return result


def _ascend_runtime() -> Dict:
    """Collect Ascend information without requiring the collector to own an NPU.

    HTTP-only collectors intentionally run with Torch backend auto-loading
    disabled.  In that mode importing torch_npu may fail or be undesirable, so
    package versions and npu-smi output are collected independently.
    """
    package_versions = _package_versions(("torch-npu", "vllm-ascend"))
    npu_list = _command(["npu-smi", "info", "-l"])
    npu_info = _command(["npu-smi", "info"], timeout=20.0)
    cann_version = None
    checked_paths = (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
        Path("/usr/local/Ascend/cann/version.info"),
        Path("/usr/local/Ascend/cann/version.cfg"),
        Path("/usr/local/Ascend/driver/version.info"),
    )
    for path in checked_paths:
        try:
            if path.is_file():
                cann_version = path.read_text(
                    encoding="utf-8", errors="replace").strip()
                if cann_version:
                    break
        except OSError:
            continue
    return {
        "available": bool(npu_list or npu_info or package_versions["torch-npu"]),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch_npu_version": package_versions["torch-npu"],
        "vllm_ascend_version": package_versions["vllm-ascend"],
        "npu_smi_list": npu_list,
        "npu_smi_info": npu_info,
        "cann_or_driver_version_file": cann_version,
        "note": (
            "NPU visibility may be absent in an HTTP-only collector; the "
            "authoritative serving allocation is stored in deployment.config"
        ),
    }


def _model_metadata(model_path: Optional[str]) -> Dict:
    if not model_path:
        return {}
    path = Path(model_path).expanduser().resolve()
    config_path = path / "config.json" if path.is_dir() else path
    result = {"path": str(path), "config_path": str(config_path)}
    if not config_path.exists():
        result["config_present"] = False
        return result
    result["config_present"] = True
    result["config_sha256"] = sha256_file(config_path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["config_error"] = f"{type(exc).__name__}: {exc}"
        return result
    keys = (
        "model_type", "architectures", "num_hidden_layers", "hidden_size",
        "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "torch_dtype", "vocab_size", "max_position_embeddings",
    )
    result["config"] = {key: config.get(key) for key in keys if key in config}
    return result


def collect_runtime_metadata(
    *, model_path: Optional[str] = None, benchmark: Optional[Mapping] = None,
    project_root: Optional[Path] = None,
) -> Dict:
    """Capture reproducibility metadata without leaking arbitrary environment values."""
    gpu_query = _command([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    topology = _command(["nvidia-smi", "topo", "-m"])
    root = Path(project_root).resolve() if project_root else None
    git_commit = _command(["git", "-C", str(root), "rev-parse", "HEAD"]) if root else None
    git_dirty = _command(["git", "-C", str(root), "status", "--porcelain"]) if root else None
    safe_environment = {
        key: value for key, value in sorted(os.environ.items())
        if key == "CUDA_VISIBLE_DEVICES" or key.startswith(SAFE_ENV_PREFIXES)
        if not any(fragment in key.upper() for fragment in SENSITIVE_ENV_FRAGMENTS)
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "software": _package_versions(
            ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers",
             "triton", "numpy", "openai", "nvidia-nccl-cu12")),
        "torch_runtime": _torch_runtime(),
        "cuda": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi_query": gpu_query,
            "topology": topology,
        },
        "ascend": _ascend_runtime(),
        "model": _model_metadata(model_path),
        "benchmark": dict(benchmark or {}),
        "environment": safe_environment,
        "source": {
            "project_root": str(root) if root else None,
            "git_commit": git_commit,
            "git_dirty": bool(git_dirty),
        },
    }


def write_profile(payload: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def validate_profile_document(payload: Mapping, expected_type: Optional[str] = None) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected profiling schema {SCHEMA_VERSION}")
    if expected_type and payload.get("profile_type") != expected_type:
        raise ValueError(f"expected profile_type={expected_type!r}")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("profile metadata is required")
    if not isinstance(payload.get("measurements"), list) or not payload["measurements"]:
        raise ValueError("profile measurements must be a non-empty list")
