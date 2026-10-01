import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_runtime_interpolation_split import (
    bounds,
    fixed_batch_shape,
    operator_shapes,
    spec_shapes,
)
from scripts.validate_runtime_interpolation import summarize


class RuntimeInterpolationValidationTests(unittest.TestCase):
    def test_runtime_cases_are_distinct_and_inside_profile_bounds(self):
        spec = {
            "workloads": {
                "prefill": [
                    {"case": "P", "batch_size": 5, "prompt_length": 384}
                ],
                "decode": [
                    {"case": "D", "batch_size": 7, "kv_length": 640}
                ],
            }
        }
        profile = {
            "measurements": [
                {"stage": stage, "status": "success", "shape": shape}
                for stage, shape in (
                    ("prefill", {"batch_size": 1, "prompt_length": 128}),
                    ("prefill", {"batch_size": 8, "prompt_length": 1024}),
                    ("decode", {"batch_size": 1, "kv_length": 128}),
                    ("decode", {"batch_size": 8, "kv_length": 1024}),
                )
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            targets = spec_shapes(spec)
            profiled = operator_shapes(path)
        self.assertFalse(targets & profiled)
        limits = bounds(profiled)
        for stage, batch, length in targets:
            self.assertLessEqual(limits[stage]["batch_min"], batch)
            self.assertGreaterEqual(limits[stage]["batch_max"], batch)
            self.assertLessEqual(limits[stage]["length_min"], length)
            self.assertGreaterEqual(limits[stage]["length_max"], length)

    def test_fixed_batch_shape_uses_initial_decode_kv(self):
        payload = {
            "configuration": {
                "stage": "decode",
                "batch_size": 5,
                "initial_kv_length": 384,
            }
        }
        self.assertEqual(fixed_batch_shape(payload), ("decode", 5, 384))

    def test_summary_reports_mape_and_bias(self):
        result = summarize([{"error_pct": -10}, {"error_pct": 20}])
        self.assertEqual(result["mape_pct"], 15)
        self.assertEqual(result["bias_pct"], 5)
        self.assertEqual(result["max_absolute_error_pct"], 20)


if __name__ == "__main__":
    unittest.main()
