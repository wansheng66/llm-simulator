import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from core.operator_interpolation import interpolate_operator_table
from core.qwen3_cost_model import LLMCostModel
from scripts.profiling_common import (
    SCHEMA_VERSION,
    describe_ms,
    dominant_component,
    validate_profile_document,
)
from scripts.collect_fixed_batch_suite import reusable_output
from scripts.collect_fixed_batch_vllm import (
    _choice_index,
    fixed_batch_eligibility,
    sample_usage_matches,
    unique_prompt_ids,
)
from scripts.collect_fixed_batch_vllm_offline import (
    request_timing,
    run_batch,
    spread_ms,
)
from scripts.fit_fixed_batch_decode_calibration import fit_log_shape
from scripts.validate_operator_holdout import csv_fieldnames, interpolate


ROOT = Path(__file__).resolve().parents[1]


def operator_row(stage, batch, length, attention, ffn, tp=4):
    key = "prompt_length" if stage == "prefill" else "kv_length"
    return {
        "status": "success",
        "stage": stage,
        "shape": {"batch_size": batch, key: length, "tp_size": tp},
        "single_layer": {
            "attention_ms": attention,
            "ffn_ms": ffn,
            "total_compute_ms": attention + ffn,
        },
    }


class Stage2ProfilingTests(unittest.TestCase):
    def test_describe_and_bottleneck(self):
        stats = describe_ms([1, 2, 3])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["p50_ms"], 2)
        self.assertEqual(dominant_component({"attention": 10, "ffn": 1}),
                         "attention")
        self.assertEqual(dominant_component({"attention": 10, "ffn": 9}),
                         "balanced")

    def test_profile_schema_requires_metadata_and_measurements(self):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profile_type": "operator",
            "metadata": {},
            "measurements": [operator_row("prefill", 1, 128, 1, 2)],
        }
        validate_profile_document(payload, "operator")
        with self.assertRaises(ValueError):
            validate_profile_document({**payload, "measurements": []}, "operator")

    def test_holdout_interpolation_uses_training_grid(self):
        rows = [
            operator_row("prefill", batch, length,
                         attention=batch * length / 128,
                         ffn=2 * batch * length / 128)
            for batch in (1, 4) for length in (128, 512)
        ]
        predicted, extrapolated = interpolate(
            rows, "prefill", 2, 256, "total_compute_ms")
        self.assertFalse(extrapolated)
        self.assertGreater(predicted, 3)
        self.assertLess(predicted, 48)

    def test_mixed_stage_csv_uses_union_of_fields(self):
        fields = csv_fieldnames([
            {"stage": "prefill", "prompt_length": 128},
            {"stage": "decode", "kv_length": 128},
        ])
        self.assertEqual(fields, ["stage", "prompt_length", "kv_length"])

    def test_shape_aware_interpolation_does_not_double_count_tokens(self):
        # All three shapes contain 512 flattened tokens. Token-wise kernels
        # should therefore remain approximately constant at the holdout shape,
        # instead of being scaled once by batch and again by sequence length.
        table = [
            {
                "batch_size": 1,
                "prompt_length": 512,
                "operator::ffn_gate_projection": 1.6,
            },
            {
                "batch_size": 4,
                "prompt_length": 128,
                "operator::ffn_gate_projection": 1.6,
            },
        ]
        predicted, extrapolated = interpolate_operator_table(
            table, "prefill", 2, 256)
        self.assertFalse(extrapolated)
        self.assertAlmostEqual(predicted["ffn_gate_projection"], 1.6)

    def test_cost_model_prefers_tp_sharded_operator_profile(self):
        rows = []
        for stage in ("prefill", "decode"):
            for batch in (1, 2):
                for length in (128, 256):
                    rows.append(operator_row(stage, batch, length, 1.0, 2.0))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profile_type": "operator",
            "metadata": {"benchmark": {"tp_size": 4}},
            "measurements": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator_profile_tp4.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            model = LLMCostModel(
                str(ROOT / "data"), operator_profile_dir=directory)
            result = model.predict_prefill(1, 128, 4)
        self.assertEqual(result["profile_source"], "tp_sharded_operator_profile")
        self.assertAlmostEqual(result["operator_breakdown"]["attention_ms"], 64.0)
        self.assertAlmostEqual(result["operator_breakdown"]["ffn_ms"], 128.0)
        self.assertIn(result["estimated_bottleneck"],
                      {"attention", "ffn", "collective", "balanced"})

    def test_collective_profile_can_use_robust_p50(self):
        payload = {
            "schema_version": 2,
            "profile_type": "collective",
            "metadata": {},
            "measurements": [{
                "status": "success",
                "operation": "all_reduce",
                "world_size": 4,
                "message_size_mb_per_rank": 16,
                "timing": {"mean_ms": 4.0, "p50_ms": 0.4},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collective_profile_tp4.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            mean = LLMCostModel._load_collective_profiles(
                Path(directory), "mean_ms")
            median = LLMCostModel._load_collective_profiles(
                Path(directory), "p50_ms")
        self.assertEqual(mean[4][0]["allreduce_ms"], 4.0)
        self.assertEqual(median[4][0]["allreduce_ms"], 0.4)

    def test_fixed_batch_resume_only_reuses_matching_valid_point(self):
        payload = {
            "schema_version": 2,
            "collector": "vllm_fixed_batch_streaming",
            "model": "model",
            "valid": True,
            "fixed_batch_valid": True,
            "configuration": {
                "stage": "prefill",
                "tp_size": 2,
                "batch_size": 8,
                "prompt_length": 512,
                "submission_mode": "batched_prompt",
                "prompt_policy": (
                    "exact-length prompts with a request-unique first 16-token block"),
            },
        }
        args = Namespace(model="model", tp_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(reusable_output(path, args, "prefill", 8, 512))
            payload["valid"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(reusable_output(path, args, "prefill", 8, 512))

    def test_fixed_batch_resume_accepts_strict_offline_point(self):
        payload = {
            "schema_version": 2,
            "collector": "vllm_fixed_batch_offline",
            "model": "/models/Qwen3-32B",
            "valid": True,
            "fixed_batch_valid": True,
            "configuration": {
                "stage": "prefill",
                "tp_size": 4,
                "batch_size": 2,
                "prompt_length": 128,
                "submission_mode": "offline_enqueue_barrier",
                "prompt_policy": (
                    "exact-length token-id prompts with a request-unique "
                    "first 16-token block"),
            },
        }
        args = Namespace(
            model="/models/Qwen3-32B",
            tp_size=4,
            collector_mode="offline",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(reusable_output(path, args, "prefill", 2, 128))

    def test_fixed_batch_prompts_are_exact_length_and_request_unique(self):
        base = [11, 12, 13, 14] * 16
        first = unique_prompt_ids(base, "repeat_1_request_1")
        second = unique_prompt_ids(base, "repeat_1_request_2")
        self.assertEqual(len(first), len(base))
        self.assertEqual(len(second), len(base))
        self.assertNotEqual(first[:16], second[:16])
        self.assertEqual(first[16:], base[16:])

    def test_strict_fixed_batch_requires_joint_submission_and_budget(self):
        args = Namespace(
            submission_mode="batched_prompt", stage="prefill",
            batch_size=2, length=128, decode_tokens=32)
        deployment = {"config": {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8192,
        }}
        eligibility = fixed_batch_eligibility(args, deployment)
        self.assertTrue(eligibility["eligible"])
        sample = {"server_usage": {
            "prompt_tokens": 256,
            "completion_tokens": 2,
        }}
        self.assertTrue(sample_usage_matches(sample, args))

        args.length = 4097
        self.assertFalse(fixed_batch_eligibility(args, deployment)["eligible"])

    def test_batched_choice_indices_are_validated(self):
        self.assertEqual(_choice_index({"index": 1}, 2), 1)
        self.assertIsNone(_choice_index({"index": 2}, 2))
        self.assertIsNone(_choice_index({}, 2))

    def test_offline_request_timing_uses_engine_timestamps(self):
        class Metrics:
            scheduled_ts = 10.0
            first_token_ts = 10.2
            last_token_ts = 10.5
            first_token_latency = 0.25
            num_generation_tokens = 4

        class Completion:
            token_ids = [1, 2, 3, 4]

        class Output:
            metrics = Metrics()
            outputs = [Completion()]
            prompt_token_ids = [7] * 128

        timing = request_timing(Output())
        self.assertAlmostEqual(timing["prefill_ms"], 200.0)
        self.assertAlmostEqual(timing["tpot_ms"], 100.0)
        self.assertEqual(timing["actual_prompt_tokens"], 128)
        self.assertEqual(timing["actual_output_tokens"], 4)
        self.assertAlmostEqual(spread_ms([
            {"scheduled_ts": 1.000}, {"scheduled_ts": 1.004},
        ], "scheduled_ts"), 4.0)

    def test_offline_fixed_batch_enqueues_every_request_before_waiting(self):
        class Metrics:
            first_token_latency = 0.01
            last_token_ts = 1.02
            num_generation_tokens = 1

            def __init__(self, scheduled_ts):
                self.scheduled_ts = scheduled_ts
                self.first_token_ts = scheduled_ts + 0.01

        class Completion:
            token_ids = [1]

        class Output:
            outputs = [Completion()]
            prompt_token_ids = [7] * 32

            def __init__(self, request_id, scheduled_ts):
                self.request_id = request_id
                self.metrics = Metrics(scheduled_ts)

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def enqueue(self, prompts, sampling_params, use_tqdm):
                self.calls.append(("enqueue", len(prompts), use_tqdm))
                return ["10-first", "11-second"]

            def wait_for_completion(self, use_tqdm):
                self.calls.append(("wait", use_tqdm))
                # Deliberately reverse completion order. Logical request IDs
                # must still be associated through IDs returned by enqueue().
                return [Output("11", 1.001), Output("10", 1.000)]

        llm = FakeLLM()
        args = Namespace(
            batch_size=2,
            stage="prefill",
            max_scheduled_spread_ms=5.0,
            max_first_token_spread_ms=10.0,
        )
        result = run_batch(llm, object(), [7, 8] * 16, args, "repeat_1")
        self.assertEqual(llm.calls, [("enqueue", 2, False), ("wait", False)])
        self.assertEqual(result["enqueued_request_ids"],
                         ["10-first", "11-second"])
        self.assertEqual(
            [row["request_id"] for row in result["requests"]],
            ["offline_prefill_repeat_1_2", "offline_prefill_repeat_1_1"],
        )
        self.assertAlmostEqual(result["scheduled_spread_ms"], 1.0)

    def test_decode_shape_calibration_recovers_synthetic_power_law(self):
        rows = []
        for batch in (2, 3, 6):
            for kv_length in (80, 272, 784):
                baseline = 50.0
                factor = 0.8 * (batch / 4.0) ** 0.2 * (
                    kv_length / 512.0) ** 0.1
                rows.append({
                    "batch_size": batch,
                    "representative_kv_length": kv_length,
                    "observed_ms": baseline * factor,
                    "baseline_predicted_ms": baseline,
                })
        fitted = fit_log_shape(rows, 4.0, 512.0)
        self.assertAlmostEqual(fitted["base_factor"], 0.8, places=8)
        self.assertAlmostEqual(fitted["batch_exponent"], 0.2, places=8)
        self.assertAlmostEqual(fitted["kv_exponent"], 0.1, places=8)


if __name__ == "__main__":
    unittest.main()
