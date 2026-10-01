import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_relative_benchmark_release.py"
SPEC = importlib.util.spec_from_file_location("benchmark_release", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkReleaseTests(unittest.TestCase):
    def test_rejects_unaccepted_validation(self):
        validation = {
            "experiment_type": "fixed_batch_relative_runtime_validation",
            "status": "verified",
            "acceptance": {"passed": False},
        }
        truth = {
            "experiment_type": "fixed_batch_relative_ground_truth",
            "acceptance": {"passed": True},
        }
        with self.assertRaisesRegex(ValueError, "has not passed"):
            MODULE.require_accepted(validation, truth)

    def test_markdown_contains_final_scores(self):
        report = {
            "name": "test benchmark",
            "status": "frozen",
            "comparison": {
                "reference": {"gpu_type": "L20"},
                "candidate": {"gpu_type": "A3"},
                "protocol_version": "0.1",
            },
            "fairness": {
                "measurement_policy": {
                    "submission_mode": "offline_enqueue_barrier"
                },
                "identity": {"tp_size": 4, "dtype": "bfloat16"},
                "paired_point_count": 18,
                "all_points_have_ground_truth": True,
            },
            "results": {
                "ground_truth_p_score": 2.3674049714,
                "predicted_p_score": 2.16476992998,
                "p_score_error_pct": 8.56,
                "ground_truth_d_score": 0.4060306929,
                "predicted_d_score": 0.41448567289,
                "d_score_error_pct": 2.08,
                "relative_mape_pct": 13.03,
                "rank_agreement_ratio": 1.0,
                "paired_prefill_cases": 9,
                "paired_decode_cases": 9,
            },
        }
        text = MODULE.markdown(report)
        self.assertIn("2.3674049714x", text)
        self.assertIn("0.4144856729x", text)
        self.assertIn("13.03%", text)
        self.assertIn("100.00%", text)

    def test_artifact_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            item = MODULE.artifact("sample", path)
        self.assertEqual(item["label"], "sample")
        self.assertEqual(len(item["sha256"]), 64)
        self.assertGreater(item["size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
