import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_fixed_batch_suite import merge_isolated_sample_runs


class FixedBatchIsolatedEngineSuiteTests(unittest.TestCase):
    def make_payload(self, observed_ms: float):
        sample = {
            "repeat_id": "repeat_1",
            "successful_requests": 1,
            "scheduled_spread_ms": 0.0,
            "first_token_spread_ms": 0.0,
            "observed_iteration_ms": observed_ms,
            "requests": [{
                "request_id": "offline_prefill_repeat_1_1",
                "engine_request_id": "2",
                "actual_prompt_tokens": 128,
                "actual_output_tokens": 1,
                "status": "success",
            }],
        }
        return {
            "schema_version": 2,
            "collector": "vllm_fixed_batch_offline",
            "configuration": {
                "stage": "prefill",
                "tp_size": 8,
                "batch_size": 1,
                "prompt_length": 128,
                "warmup_batches": 1,
                "measured_batches": 1,
                "submission_mode": "offline_enqueue_barrier",
            },
            "metadata": {"benchmark": {}},
            "warmup": [],
            "samples": [sample],
            "summary": {"count": 1, "mean_ms": observed_ms},
            "valid": True,
            "fixed_batch_valid": True,
            "validity": {"output_shape_valid": True},
            "limitations": [],
        }

    def test_merge_records_independent_engine_lifecycle(self):
        args = argparse.Namespace(
            repeats=3,
            warmup=1,
            max_scheduled_spread_ms=5.0,
            max_first_token_spread_ms=10.0,
            offline_cooldown_seconds=60.0,
        )
        payloads = [
            self.make_payload(88.624),
            self.make_payload(82.849),
            self.make_payload(87.599),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_paths = [root / f"engine_{index}.json" for index in range(1, 4)]
            output = root / "prefill_b1_l128.json"
            merged = merge_isolated_sample_runs(
                payloads, raw_paths, args, output
            )

            self.assertTrue(merged["fixed_batch_valid"])
            self.assertEqual(merged["summary"]["count"], 3)
            self.assertAlmostEqual(merged["summary"]["mean_ms"], 86.3573333333)
            self.assertEqual(
                merged["configuration"]["engine_lifecycle"],
                "isolated_engine_per_sample",
            )
            self.assertEqual(merged["configuration"]["engine_instances"], 3)
            self.assertEqual(
                [sample["repeat_id"] for sample in merged["samples"]],
                ["repeat_1", "repeat_2", "repeat_3"],
            )
            self.assertTrue(output.exists())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["engine_lifecycle"]["engine_instances"], 3)

    def test_merge_rejects_invalid_isolated_run(self):
        args = argparse.Namespace(
            repeats=1,
            warmup=1,
            max_scheduled_spread_ms=5.0,
            max_first_token_spread_ms=10.0,
            offline_cooldown_seconds=60.0,
        )
        payload = self.make_payload(80.0)
        payload["fixed_batch_valid"] = False
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "one valid sample"):
                merge_isolated_sample_runs(
                    [payload], [root / "engine_1.json"], args,
                    root / "point.json",
                )


if __name__ == "__main__":
    unittest.main()
