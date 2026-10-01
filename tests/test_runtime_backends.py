import json
import tempfile
import unittest
from pathlib import Path

from core.runtime_backend import load_runtime_backend, validate_backend_identity
from scripts.verify_runtime_regression import compare


class RuntimeBackendTests(unittest.TestCase):
    def test_verified_l20_extrapolation_backend_records_independent_evidence(self):
        root = Path(__file__).resolve().parents[1]
        backend = load_runtime_backend(
            root / "configs/runtime_backends/"
            "djs_l20_8gpu_node_tp4_extrapolation_verified.json",
            root,
            require_artifacts=False,
        )
        manifest = backend["manifest"]
        validation = manifest["validation"]
        metrics = validation["metrics"]

        self.assertEqual(validation["status"], "verified")
        self.assertEqual(manifest["tp_size"], 4)
        self.assertEqual(metrics["decode_mape_pct"], 2.66)
        self.assertTrue(metrics["all_predictions_extrapolated"])
        self.assertIn("v4", validation["independent_spec"])
        self.assertEqual(
            manifest["calibration"]["stages"], ["decode"])

    def test_backend_owns_profiles_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("operators", "collectives"):
                target = root / name
                target.mkdir()
                (target / "profile.json").write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "backend_id": "test_backend",
                "hardware_id": "test_hardware",
                "platform_family": "test",
                "protocol_version": "0.1",
                "tp_size": 4,
                "model_config_sha256": "model-hash",
                "profiles": {
                    "operator_profile_dir": "operators",
                    "collective_profile_dir": "collectives",
                    "statistic": "mean",
                },
                "calibration": {
                    "enabled": False, "file": None, "stages": []},
                "validation": {"status": "pending"},
            }
            path = root / "backend.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            backend = load_runtime_backend(path, root)
            self.assertEqual(backend["operator_profile_statistic"], "mean")
            self.assertEqual(backend["collective_profile_statistic"], "mean")
            spec = {
                "protocol_version": "0.1",
                "model": {"config_sha256": "model-hash"},
            }
            validate_backend_identity(
                backend, spec, "test_hardware", 4)
            with self.assertRaisesRegex(ValueError, "hardware_id"):
                validate_backend_identity(backend, spec, "other", 4)

    def test_backend_can_own_distinct_operator_and_collective_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("operators", "collectives"):
                target = root / name
                target.mkdir()
                (target / "profile.json").write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "backend_id": "mixed_stats",
                "hardware_id": "ascend",
                "platform_family": "ascend",
                "protocol_version": "0.1",
                "tp_size": 8,
                "model_config_sha256": "model-hash",
                "profiles": {
                    "operator_profile_dir": "operators",
                    "collective_profile_dir": "collectives",
                    "statistic": "mean",
                    "operator_statistic": "mean",
                    "collective_statistic": "p50",
                },
                "calibration": {
                    "enabled": False, "file": None, "stages": []},
                "validation": {"status": "verified"},
            }
            path = root / "backend.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            backend = load_runtime_backend(path, root)
            self.assertEqual(backend["operator_profile_statistic"], "mean")
            self.assertEqual(backend["collective_profile_statistic"], "p50")

    def test_regression_check_detects_only_changed_hardware_points(self):
        golden = {
            "meta": {"gpu_type": "l20"},
            "prefill": [{"case": "P", "total_ms": 10.0}],
            "decode": [{"case": "D", "total_ms": 5.0}],
        }
        same = json.loads(json.dumps(golden))
        self.assertEqual(compare(golden, same, 0.0), [])
        same["decode"][0]["total_ms"] = 5.5
        changes = compare(golden, same, 5.0)
        self.assertEqual(changes[0]["case"], "D")
        self.assertFalse(changes[0]["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
