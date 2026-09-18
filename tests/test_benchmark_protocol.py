import unittest
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from core.benchmark_protocol import adapt_legacy_report, attention_type, compare_reports
from scripts.recommend_benchmark_workload import recommend
from scripts.collect_relative_ground_truth import reusable
from scripts.compare_relative_ground_truth import throughput


ROOT = Path(__file__).resolve().parents[1]


def report(gpu, tp=4, model="Qwen/Qwen3-32B", adjusted=False):
    return {
        "meta": {
            "gpu_type": gpu,
            "tp": tp,
            "protocol_version": "0.1",
            "model_id": model,
        },
        "prefill": [{
            "case": "P_S", "batch": 1, "original_batch": 1,
            "seq": 512, "throughput_tok_s": 200 if gpu == "B" else 100,
            "memory_oom": False,
        }],
        "decode": [{
            "case": "D_M", "batch": 4 if adjusted else 8,
            "original_batch": 8, "kv": 1024,
            "throughput_tok_s": 120 if gpu == "B" else 100,
            "memory_oom": adjusted,
        }],
    }


class BenchmarkProtocolTests(unittest.TestCase):
    def test_attention_family(self):
        self.assertEqual(attention_type(64, 64), "MHA")
        self.assertEqual(attention_type(64, 1), "MQA")
        self.assertEqual(attention_type(64, 8), "GQA")

    def test_relative_score_pairs_identical_cases(self):
        result = compare_reports(report("B"), report("A"))
        self.assertEqual(result["status"], "provisional")
        self.assertAlmostEqual(result["scores"]["prefill_speedup"], 2.0)
        self.assertAlmostEqual(result["scores"]["decode_speedup"], 1.2)

    def test_adjusted_batch_is_excluded_from_iso_workload(self):
        result = compare_reports(report("B", adjusted=True), report("A"))
        self.assertEqual(result["scores"]["paired_decode_cases"], 0)

    def test_ground_truth_makes_complete_comparison_verified(self):
        result = compare_reports(report("B"), report("A"), {"P_S": 2.0, "D_M": 1.25})
        self.assertEqual(result["status"], "verified")
        decode = next(point for point in result["points"] if point["stage"] == "decode")
        self.assertAlmostEqual(decode["relative_error_pct"], 4.0)
        self.assertAlmostEqual(result["validation"]["relative_mape_pct"], 2.0)
        self.assertEqual(result["validation"]["rank_agreement_ratio"], 1.0)

    def test_different_tp_is_invalid(self):
        result = compare_reports(report("B", tp=8), report("A", tp=4))
        self.assertEqual(result["status"], "invalid")

    def test_workload_recommender_derives_gqa_and_context(self):
        spec = recommend({
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_position_embeddings": 2048,
        }, "local/Qwen3-32B")
        self.assertEqual(spec["model"]["attention_type"], "GQA")
        self.assertEqual(spec["workloads"]["prefill"][-1]["prompt_length"], 2048)
        self.assertTrue(spec["recommendation"]["review_required"])

    def test_legacy_throughput_is_batch_aware_and_non_mutating(self):
        legacy = {
            "meta": {"gpu_type": "A", "tp": 4},
            "prefill": [{"case": "P", "batch": 4, "seq": 100, "total_ms": 200}],
            "decode": [{"case": "D", "batch": 8, "kv": 100, "total_ms": 40}],
        }
        spec = {"model": {"model_id": "m", "name": "m", "dtype": "bf16"}}
        adapted = adapt_legacy_report(legacy, spec)
        self.assertEqual(adapted["prefill"][0]["throughput_tok_s"], 2000)
        self.assertEqual(adapted["decode"][0]["throughput_tok_s"], 200)
        self.assertNotIn("protocol_version", legacy["meta"])

    def test_fixed_batch_truth_uses_batch_tokens(self):
        prefill, cv = throughput(
            "prefill",
            {"case": "P", "batch_size": 4, "prompt_length": 100},
            {"summary": {"mean_ms": 200, "std_ms": 10}},
        )
        decode, _ = throughput(
            "decode",
            {"case": "D", "batch_size": 8, "kv_length": 100},
            {"summary": {"mean_ms": 40, "std_ms": 0}},
        )
        self.assertEqual(prefill, 2000)
        self.assertEqual(decode, 200)
        self.assertEqual(cv, 0.05)

    def test_ground_truth_resume_requires_exact_valid_shape(self):
        payload = {
            "collector": "vllm_fixed_batch_streaming",
            "model": "model",
            "valid": True,
            "fixed_batch_valid": True,
            "configuration": {
                "stage": "decode", "tp_size": 4, "batch_size": 8,
                "initial_kv_length": 1024, "measured_batches": 3,
                "submission_mode": "batched_prompt",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(reusable(path, "model", 4, "decode", 8, 1024, 3))
            self.assertFalse(reusable(path, "model", 4, "decode", 4, 1024, 3))

    def test_ground_truth_comparison_cli(self):
        spec = {
            "schema_version": 1, "protocol_version": "0.1",
            "benchmark_name": "test",
            "model": {"name": "m", "model_id": "m", "dtype": "bfloat16"},
            "workloads": {
                "prefill": [{"case": "P", "batch_size": 1, "prompt_length": 16}],
                "decode": [{"case": "D", "batch_size": 2, "kv_length": 16}],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
            manifests = []
            for hardware, mean in (("A", 20.0), ("B", 10.0)):
                run_dir = root / hardware
                run_dir.mkdir()
                entries = []
                for stage, case, batch, key in (
                    ("prefill", "P", 1, "prompt_length"),
                    ("decode", "D", 2, "kv_length"),
                ):
                    artifact = run_dir / f"{case}.json"
                    artifact.write_text(json.dumps({
                        "valid": True,
                        "summary": {"mean_ms": mean, "std_ms": 0.1},
                        "deployment": {"config": {
                            "tensor_parallel_size": 4, "dtype": "bfloat16"}},
                    }), encoding="utf-8")
                    entry = {"stage": stage, "case": case,
                             "batch_size": batch, key: 16,
                             "file": str(artifact)}
                    entries.append(entry)
                manifest = run_dir / "manifest.json"
                manifest.write_text(json.dumps({
                    "schema_version": 1,
                    "experiment_type": "relative_ground_truth_run",
                    "protocol_version": "0.1", "hardware_id": hardware,
                    "model_id": "m", "tp_size": 4, "dtype": "bfloat16",
                    "benchmark_spec_sha256": digest, "entries": entries,
                }), encoding="utf-8")
                manifests.append(manifest)
            output = root / "comparison"
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "compare_relative_ground_truth.py"),
                "--reference-manifest", str(manifests[0]),
                "--candidate-manifest", str(manifests[1]),
                "--benchmark-spec", str(spec_path),
                "--output-dir", str(output),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "relative_ground_truth_report.json").read_text())
            self.assertAlmostEqual(report["ground_truth_scores"]["prefill_speedup"], 2.0)
            self.assertAlmostEqual(report["ground_truth_scores"]["decode_speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
