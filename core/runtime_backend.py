"""Versioned hardware-runtime backend manifests.

The benchmark protocol is shared across hardware.  Profiling inputs,
statistics and calibration remain backend-owned so adding one accelerator
cannot silently change predictions for another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*.json") if item.is_file())
    if not files:
        raise ValueError(f"runtime profile directory contains no JSON: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve(project_root: Path, stored: str) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else project_root / path


def load_runtime_backend(path: str | Path, project_root: str | Path,
                         require_artifacts: bool = True) -> Dict:
    """Load and validate a backend manifest, resolving artifact paths."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported runtime backend schema_version")
    required = {
        "backend_id", "hardware_id", "platform_family", "protocol_version",
        "tp_size", "model_config_sha256", "profiles", "calibration",
        "validation",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"runtime backend is missing {sorted(missing)}")

    root = Path(project_root).resolve()
    profiles = payload["profiles"]
    statistic = profiles.get("statistic", "mean")
    operator_statistic = profiles.get("operator_statistic", statistic)
    collective_statistic = profiles.get("collective_statistic", statistic)
    if operator_statistic not in {"mean", "p50"}:
        raise ValueError(
            "runtime backend operator statistic must be mean or p50")
    if collective_statistic not in {"mean", "p50"}:
        raise ValueError(
            "runtime backend collective statistic must be mean or p50")
    operator_dir = _resolve(root, profiles["operator_profile_dir"])
    collective_dir = _resolve(root, profiles["collective_profile_dir"])
    calibration = payload["calibration"]
    calibration_file = (
        _resolve(root, calibration["file"])
        if calibration.get("enabled") else None
    )
    if calibration.get("enabled") and not calibration.get("file"):
        raise ValueError("enabled runtime calibration requires a file")
    calibrated_stages = set(calibration.get("stages", []))
    if not calibrated_stages <= {"prefill", "decode"}:
        raise ValueError("runtime calibration stages must be prefill/decode")

    artifacts = [operator_dir, collective_dir]
    if calibration_file is not None:
        artifacts.append(calibration_file)
    if require_artifacts:
        missing_artifacts = [str(item) for item in artifacts if not item.exists()]
        if missing_artifacts:
            raise FileNotFoundError(
                "runtime backend artifacts are missing: " +
                ", ".join(missing_artifacts))

    fingerprints = {"manifest_sha256": _sha256_file(manifest_path)}
    if require_artifacts:
        fingerprints.update({
            "operator_profiles_sha256": _sha256_directory(operator_dir),
            "collective_profiles_sha256": _sha256_directory(collective_dir),
            "calibration_sha256": (
                _sha256_file(calibration_file)
                if calibration_file is not None else None),
        })
    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "operator_profile_dir": operator_dir,
        "collective_profile_dir": collective_dir,
        "profile_statistic": statistic,
        "operator_profile_statistic": operator_statistic,
        "collective_profile_statistic": collective_statistic,
        "calibration_file": calibration_file,
        "calibration_enabled": bool(calibration.get("enabled")),
        "fingerprints": fingerprints,
    }


def validate_backend_identity(backend: Dict, spec: Dict,
                              hardware_id: str, tp_size: int) -> None:
    manifest = backend["manifest"]
    checks = {
        "hardware_id": (manifest["hardware_id"], hardware_id),
        "tp_size": (int(manifest["tp_size"]), int(tp_size)),
        "protocol_version": (
            manifest["protocol_version"], spec["protocol_version"]),
        "model_config_sha256": (
            manifest["model_config_sha256"],
            spec["model"].get("config_sha256")),
    }
    mismatches = [
        f"{name}: backend={left!r}, requested={right!r}"
        for name, (left, right) in checks.items() if left != right
    ]
    if mismatches:
        raise ValueError("runtime backend identity mismatch: " + "; ".join(mismatches))
