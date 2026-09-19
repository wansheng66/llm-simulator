import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_fixed_batch_suites import compare_suites, load_suite


class FixedBatchSuiteComparisonTests(unittest.TestCase):
    def make_suite(self, root: Path, hardware_id: str,
                   prefill_ms: float, decode_ms: float,
                   cv_pct: float = 0.5) -> Path:
        root.mkdir(parents=True)
        files = []
        for stage in ("prefill", "decode"):
            for batch in (1, 4, 8):
                for length in (128, 512, 1024):
                    mean = prefill_ms if stage == "prefill" else decode_ms
                    config = {
                        "stage": stage,
                        "tp_size": 4,
                        "batch_size": batch,
                        "decode_tokens": 1 if stage == "prefill" else 32,
                        "dtype": "bfloat16",
                        "warmup_batches": 2,
                        "measured_batches": 7,
                        "submission_mode": "offline_enqueue_barrier",
                        "release_policy": "enqueue all then wait",
                        "v1_engine_core_multiprocessing": False,
                        "async_scheduling": False,
                        "enable_chunked_prefill": False,
                        "prompt_policy": "request-unique first 16-token block",
                    }
                    config["prompt_length" if stage == "prefill"
                           else "initial_kv_length"] = length
                    payload = {
                        "schema_version": 2,
                        "collector": "vllm_fixed_batch_offline",
                        "valid": True,
                        "fixed_batch_valid": True,
                        "configuration": config,
                        "deployment": {"config": {
                            "tensor_parallel_size": 4,
                            "dtype": "bfloat16",
                            "max_model_len": 8192,
                            "max_num_batched_tokens": 8192,
                            "max_num_seqs": 32,
                            "enforce_eager": True,
                            "enable_prefix_caching": False,
                            "block_size_tokens": 16,
                        }},
                        "metadata": {"model": {
                            "config_sha256": "same-model",
                        }},
                        "serving_host_hardware": {"snapshot": {
                            "hardware_identity": {"hardware_id": hardware_id},
                        }},
                        "summary": {
                            "count": 7,
                            "mean_ms": mean,
                            "std_ms": mean * cv_pct / 100.0,
                        },
                    }
                    path = root / f"{stage}_b{batch}_l{length}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    files.append(str(path))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "experiment_type": "fixed_batch_suite_manifest",
            "collector_mode": "offline",
            "tp_size": 4,
            "submission_mode": "offline_enqueue_barrier",
            "files": files,
        }), encoding="utf-8")
        return manifest

    def test_p_and_d_scores_use_paired_throughput_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = load_suite(self.make_suite(
                root / "reference", "reference", 100.0, 10.0))
            candidate = load_suite(self.make_suite(
                root / "candidate", "candidate", 50.0, 20.0))
            result = compare_suites(reference, candidate, 3.0)
            self.assertAlmostEqual(result["scores"]["p_score"], 2.0)
            self.assertAlmostEqual(result["scores"]["d_score"], 0.5)
            self.assertTrue(result["acceptance"]["passed"])
            self.assertEqual(result["winner_counts"]["prefill"]["candidate"], 9)
            self.assertEqual(result["winner_counts"]["decode"]["reference"], 9)
            self.assertIsNone(result["scores"]["combined_score"])

    def test_cv_threshold_rejects_unstable_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = load_suite(self.make_suite(
                root / "reference", "reference", 100.0, 10.0))
            candidate = load_suite(self.make_suite(
                root / "candidate", "candidate", 50.0, 20.0, cv_pct=4.0))
            result = compare_suites(reference, candidate, 3.0)
            self.assertFalse(result["acceptance"]["passed"])
            self.assertFalse(result["acceptance"]["all_points_repeatable"])


if __name__ == "__main__":
    unittest.main()
