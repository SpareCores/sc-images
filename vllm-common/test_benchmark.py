"""Autoconfig scaling: 1 vCPU through the largest fleet size (896)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import benchmark as b

SMOL = b.DEFAULT_MODELS[0]
QWEN = b.DEFAULT_MODELS[1]
GEMMA = b.DEFAULT_MODELS[2]
LLAMA8 = b.DEFAULT_MODELS[3]

ENV_KEYS = (
    "BENCHMARK_VLLM_VCPUS",
    "BENCHMARK_VLLM_NUMA_NODES",
    "BENCHMARK_VLLM_CPU_TOPOLOGY",
    "BENCHMARK_VLLM_CPU_TP",
    "BENCHMARK_VLLM_CPU_DP",
    "BENCHMARK_VLLM_GPU_TP",
    "BENCHMARK_VLLM_GPU_DP",
    "BENCHMARK_VLLM_MAX_NUM_SEQS",
    "VLLM_CPU_OMP_THREADS_BIND",
    "VLLM_CPU_GPU_MEMORY_UTILIZATION",
)


class FakeMemory:
    def __init__(self, total_gb: float, avail_gb: float) -> None:
        self.total = int(total_gb * 1e9)
        self.available = int(avail_gb * 1e9)


def topology(n_numa: int, cpus_per_node: int) -> str:
    parts = []
    cpu = 0
    for _ in range(n_numa):
        parts.append(f"{cpu}-{cpu + cpus_per_node - 1}")
        cpu += cpus_per_node
    return "|".join(parts)


def bind_cpu_ids(bind: str) -> list[int]:
    ids: list[int] = []
    for part in bind.split("|"):
        ids.extend(b.parse_cpu_id_list(part))
    return ids


def chat_budget() -> b.BudgetPlan:
    return b.BudgetPlan(
        per_run_sec=240,
        total_runs=8,
        overall_timeout_sec=7200,
        reserve_sec=600,
    )


class CpuScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in ENV_KEYS}
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        b.reset_autoconfig_state()

    def tearDown(self) -> None:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        b.reset_autoconfig_state()

    def _host(self, vcpus: int, n_numa: int, ram_gb: float = 2000.0) -> None:
        assert vcpus % n_numa == 0, "test topology must divide evenly"
        os.environ["BENCHMARK_VLLM_VCPUS"] = str(vcpus)
        os.environ["BENCHMARK_VLLM_NUMA_NODES"] = str(n_numa)
        os.environ["BENCHMARK_VLLM_CPU_TOPOLOGY"] = topology(n_numa, vcpus // n_numa)
        b.reset_autoconfig_state()
        b._HOST = b.HostProfile(vcpus=vcpus, ram_total_gb=ram_gb, ram_avail_gb=ram_gb * 0.9)

    def test_compact_parse_roundtrip(self) -> None:
        ids = [0, 1, 2, 5, 7, 8, 9]
        self.assertEqual(b.parse_cpu_id_list(b.compact_cpu_ids(ids)), ids)

    def test_one_vcpu_runs_a_single_rank(self) -> None:
        self._host(1, 1, ram_gb=8.0)
        self.assertEqual(b.cpu_tensor_parallel_size(SMOL), 1)
        self.assertEqual(b.cpu_data_parallel_size(SMOL), 1)
        bind = b.cpu_omp_threads_bind(SMOL)
        self.assertEqual(bind, "0")
        tuning = b.compute_tuning("cpu", SMOL, chat_budget(), max_model_len=2048)
        self.assertGreaterEqual(tuning.max_concurrency, 1)
        self.assertEqual(tuning.max_workers, 1)
        self.assertGreaterEqual(tuning.max_seconds_per_strategy, 1)
        self.assertLess(tuning.rampup_duration, tuning.max_seconds_per_strategy)
        self.assertGreaterEqual(
            tuning.max_seconds_per_strategy - tuning.rampup_duration, 1
        )

    def test_explicit_bind_keeps_smt_and_all_logical_cpus(self) -> None:
        # 96 advertised vCPUs, 2-way SMT, 2 sockets: auto would keep ~48 cores
        # on NUMA 0 only. Explicit bind must list every logical CPU (minus 1
        # reserved for the frontend).
        self._host(96, 2)
        bind = b.cpu_omp_threads_bind(SMOL)
        self.assertNotEqual(bind.lower(), "auto")
        ids = bind_cpu_ids(bind)
        self.assertEqual(len(ids), 95)
        self.assertEqual(len(set(ids)), 95)
        self.assertEqual(min(ids), 0)
        self.assertEqual(set(range(96)) - set(ids), {95})

    def test_smol_adds_dp_replicas_on_large_boxes(self) -> None:
        self._host(896, 8)
        self.assertEqual(b.cpu_tensor_parallel_size(SMOL), 1)
        dp = b.cpu_data_parallel_size(SMOL)
        self.assertGreaterEqual(dp, 8)
        self.assertLessEqual(dp, b.MAX_CPU_DP)
        bind = b.cpu_omp_threads_bind(SMOL)
        parts = bind.split("|")
        self.assertEqual(len(parts), dp)
        self.assertEqual(len(bind_cpu_ids(bind)), 895)

    def test_llama8_uses_tp_across_numa(self) -> None:
        self._host(896, 8)
        self.assertEqual(b.cpu_tensor_parallel_size(LLAMA8), 8)
        dp = b.cpu_data_parallel_size(LLAMA8)
        self.assertGreaterEqual(dp, 1)
        bind = b.cpu_omp_threads_bind(LLAMA8)
        self.assertEqual(len(bind.split("|")), 8 * dp)

    def test_rampup_does_not_eat_throughput_stage(self) -> None:
        """The old v/4 rampup hit 24s at 96 cores and 30s at 128+, with ~40s stages."""
        measure = {}
        for vcpus, numa in ((1, 1), (48, 2), (96, 2), (192, 4), (416, 8), (896, 8)):
            self._host(vcpus, numa)
            tuning = b.compute_tuning("cpu", SMOL, chat_budget(), max_model_len=2048)
            leftover = tuning.max_seconds_per_strategy - tuning.rampup_duration
            measure[vcpus] = leftover
            self.assertLessEqual(tuning.rampup_duration, b.RAMPUP_MAX_SEC)
            self.assertGreaterEqual(leftover, b.MIN_THROUGHPUT_MEASURE_SEC - 1)
        self.assertGreaterEqual(measure[896], measure[96] - 5)

    def test_user_bind_override_is_kept(self) -> None:
        self._host(16, 1)
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-7"
        self.assertEqual(b.cpu_omp_threads_bind(SMOL), "0-7")

    def test_auto_env_still_gets_explicit_lists(self) -> None:
        self._host(8, 1)
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "auto"
        bind = b.cpu_omp_threads_bind(SMOL)
        self.assertNotEqual(bind.lower(), "auto")
        self.assertTrue(bind.replace("-", "").replace(",", "").replace("|", "").isdigit())

    def test_max_num_seqs_env_override(self) -> None:
        self._host(32, 1)
        os.environ["BENCHMARK_VLLM_MAX_NUM_SEQS"] = "7"
        b.reset_autoconfig_state()
        tuning = b.compute_tuning("cpu", SMOL, chat_budget(), max_model_len=2048)
        self.assertEqual(tuning.max_num_seqs, 7)

    def test_ranks_per_memory_node_counts_colocated_ranks(self) -> None:
        self._host(32, 1)
        self.assertEqual(b.cpu_ranks_per_memory_node(SMOL, 1, 1), 1)
        self.assertEqual(b.cpu_ranks_per_memory_node(SMOL, 1, 4), 4)
        self._host(96, 4)
        self.assertEqual(b.cpu_ranks_per_memory_node(SMOL, 1, 4), 1)
        self.assertEqual(b.cpu_ranks_per_memory_node(SMOL, 1, 12), 3)
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
        self.assertEqual(b.cpu_ranks_per_memory_node(SMOL, 1, 4), 4)

    def test_dp_ranks_split_the_numa_memory_budget(self) -> None:
        """Each CPU worker reserves util × its NUMA node's RAM, so DP must divide it.

        Regression: r8i.8xlarge (32 vCPU, 1 node, 266 GB) served util=0.50 to every
        rank, so DP=2 asked for 266 GB of a 263 GB node and EngineCore_DP0 died.
        """
        self._host(512, 1, ram_gb=266.0)
        with mock.patch.object(b, "virtual_memory", lambda: FakeMemory(266.0, 262.8)):
            for dp in (1, 2, 4, 8, 16, b.MAX_CPU_DP):
                os.environ["BENCHMARK_VLLM_CPU_DP"] = str(dp)
                b.reset_autoconfig_state()
                b._HOST = b.HostProfile(vcpus=512, ram_total_gb=266.0, ram_avail_gb=262.8)
                ranks = b.cpu_ranks_per_memory_node(SMOL)
                self.assertEqual(ranks, dp)
                tuning = b.compute_tuning("cpu", SMOL, chat_budget(), max_model_len=2048)
                reserved_gb = tuning.kv_memory_util * ranks * 266.0
                self.assertLessEqual(reserved_gb, 262.8, f"dp={dp} over-reserves RAM")
                self.assertTrue(
                    b.workload_kv_fits(SMOL, "cpu", 2048, tuning.kv_memory_util),
                    f"dp={dp} left no usable KV cache",
                )

    def test_single_rank_keeps_the_full_memory_fraction(self) -> None:
        self._host(96, 4, ram_gb=1536.0)
        with mock.patch.object(b, "virtual_memory", lambda: FakeMemory(1536.0, 1500.0)):
            os.environ["BENCHMARK_VLLM_CPU_DP"] = "4"
            b.reset_autoconfig_state()
            b._HOST = b.HostProfile(vcpus=96, ram_total_gb=1536.0, ram_avail_gb=1500.0)
            self.assertEqual(b.cpu_ranks_per_memory_node(SMOL), 1)
            tuning = b.compute_tuning("cpu", SMOL, chat_budget(), max_model_len=2048)
            self.assertAlmostEqual(tuning.kv_memory_util, 0.50, places=2)

    def test_gpu_dp_override(self) -> None:
        os.environ["BENCHMARK_VLLM_GPU_DP"] = "4"
        self.assertEqual(b.gpu_data_parallel_size(SMOL), 4)
        os.environ.pop("BENCHMARK_VLLM_GPU_DP")
        self.assertEqual(b.gpu_data_parallel_size(SMOL), 1)

    def test_kv_bytes_come_from_model_config(self) -> None:
        metadata = b.ModelMetadata(
            weight_bytes=16 * 1024**3,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            torch_dtype="bfloat16",
            source="test",
        )
        with mock.patch.object(b, "model_metadata", lambda _spec: metadata):
            # K+V × 32 layers × 8 KV heads × 128 head_dim × bf16.
            self.assertEqual(b.kv_bytes_per_token(LLAMA8, tp=1), 131_072)
            self.assertEqual(b.kv_bytes_per_token(LLAMA8, tp=4), 32_768)

    def test_nobind_large_model_is_rejected_per_memory_node(self) -> None:
        """Aggregate RAM is not enough: all nobind workers allocate on node 0."""
        self._host(384, 6, ram_gb=3000.0)
        metadata = b.ModelMetadata(
            weight_bytes=16 * 1024**3,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            torch_dtype="bfloat16",
            source="test",
        )
        with (
            mock.patch.object(b, "model_metadata", lambda _spec: metadata),
            mock.patch.object(b, "current_ram_gb", lambda: (3000.0, 2940.0)),
        ):
            b.host_memory_by_numa_gb.cache_clear()
            explicit = b.cpu_layout_memory_plan(
                LLAMA8, tp=1, dp=24, max_model_len=2048
            )
            self.assertTrue(explicit["fits"])
            self.assertEqual(max(n["ranks"] for n in explicit["nodes"]), 4)
            os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
            nobind = b.cpu_layout_memory_plan(
                LLAMA8, tp=1, dp=24, max_model_len=2048
            )
            self.assertFalse(nobind["fits"])
            self.assertEqual(nobind["nodes"][0]["ranks"], 24)
            self.assertGreater(
                nobind["nodes"][0]["required_gib"],
                nobind["nodes"][0]["budget_gib"],
            )

    def test_gpu_fit_is_per_device_not_aggregate_vram(self) -> None:
        metadata = b.ModelMetadata(
            weight_bytes=40 * 1024**3,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=8,
            hidden_size=5120,
            torch_dtype="bfloat16",
            source="test",
        )
        gpu = {
            "gpu_count": 4,
            "gpu_model": "test",
            "vram_gb": 24.0,
            "total_vram_gb": 96.0,
            "vram_per_device_gb": [24.0] * 4,
        }
        with (
            mock.patch.object(b, "model_metadata", lambda _spec: metadata),
            mock.patch.object(b, "gpu_info", lambda: gpu),
        ):
            self.assertFalse(
                b.gpu_layout_memory_plan(
                    b.DEFAULT_MODELS[4], tp=1, dp=4, max_model_len=2048
                )["fits"]
            )
            self.assertTrue(
                b.gpu_layout_memory_plan(
                    b.DEFAULT_MODELS[4], tp=4, dp=1, max_model_len=2048
                )["fits"]
            )

    def test_default_model_ladder_contains_larger_models(self) -> None:
        names = [spec.short_name for spec in b.DEFAULT_MODELS]
        self.assertEqual(
            names,
            [
                "smol-135m",
                "qwen-0.5b",
                "gemma-2b",
                "llama-8b",
                "phi-4",
                "llama-70b",
            ],
        )

    def test_max_num_seqs_is_capped_by_config_derived_kv_budget(self) -> None:
        self._host(32, 1, ram_gb=64.0)
        metadata = b.ModelMetadata(
            weight_bytes=16 * 1024**3,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            torch_dtype="bfloat16",
            source="test",
        )
        with (
            mock.patch.object(b, "model_metadata", lambda _spec: metadata),
            mock.patch.object(b, "current_ram_gb", lambda: (64.0, 60.0)),
        ):
            b.host_memory_by_numa_gb.cache_clear()
            cap = b.max_sequences_for_layout(
                LLAMA8,
                mode="cpu",
                tp=1,
                dp=1,
                max_model_len=2048,
                cpu_memory_util=0.5,
                gpu_memory_util_value=0.9,
            )
            # 16 GiB weights + runtime leave a finite KV budget; the cap is
            # computed from the exact 128 KiB/token config footprint.
            self.assertGreater(cap, 1)
            self.assertLess(cap, 128)

    def test_quantized_70b_preflight_matches_pipeline_parallel_server(self) -> None:
        metadata = b.ModelMetadata(
            weight_bytes=48 * 1024**3,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            hidden_size=8192,
            torch_dtype="bfloat16",
            source="test",
            head_dim=128,
        )
        gpu = {
            "gpu_count": 4,
            "gpu_model": "test",
            "vram_gb": 24.0,
            "total_vram_gb": 96.0,
            "vram_per_device_gb": [24.0] * 4,
            "available_vram_per_device_gb": [24.0] * 4,
        }
        with (
            mock.patch.object(b, "model_metadata", lambda _spec: metadata),
            mock.patch.object(b, "gpu_info", lambda: gpu),
        ):
            plan = b.model_host_plan(b.DEFAULT_MODELS[-1], "gpu")
            self.assertTrue(plan["runnable"])
            self.assertEqual(plan["feasible_layouts"], [{"tp": 1, "dp": 1, "pp": 4}])


class CpuDtypeAndGpuCompatTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dtype = os.environ.get("VLLM_CPU_DTYPE")
        os.environ.pop("VLLM_CPU_DTYPE", None)
        b.reset_autoconfig_state()

    def tearDown(self) -> None:
        os.environ.pop("VLLM_CPU_DTYPE", None)
        if self._dtype is None:
            os.environ.pop("VLLM_CPU_DTYPE", None)
        else:
            os.environ["VLLM_CPU_DTYPE"] = self._dtype
        b.reset_autoconfig_state()

    def test_cpu_flags_parse_arm_features(self) -> None:
        cpuinfo = (
            "processor\t: 0\n"
            "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp\n"
            "CPU part\t: 0xd0c\n"
        )
        with mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)):
            b.get_cpu_flags.cache_clear()
            flags = b.get_cpu_flags()
        self.assertIn("asimd", flags)
        self.assertNotIn("bf16", flags)
        self.assertFalse(b.cpu_has_bf16())

    def test_cpu_flags_detect_arm_bf16(self) -> None:
        cpuinfo = "Features\t: fp asimd bf16 sve\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)):
            b.get_cpu_flags.cache_clear()
            self.assertTrue(b.cpu_has_bf16())

    def test_cpu_serve_dtype_float16_without_bf16(self) -> None:
        with mock.patch.object(b, "cpu_has_bf16", return_value=False):
            self.assertEqual(b.cpu_serve_dtype(SMOL), "float16")

    def test_cpu_serve_dtype_bfloat16_with_bf16(self) -> None:
        with mock.patch.object(b, "cpu_has_bf16", return_value=True):
            self.assertEqual(b.cpu_serve_dtype(SMOL), "bfloat16")

    def test_cpu_serve_dtype_override(self) -> None:
        os.environ["VLLM_CPU_DTYPE"] = "float32"
        with mock.patch.object(b, "cpu_has_bf16", return_value=True):
            self.assertEqual(b.cpu_serve_dtype(SMOL), "float32")

    def test_gpu_supports_ampere(self) -> None:
        info = {"compute_capabilities": [(8, 9), (8, 9)]}
        self.assertTrue(b.gpu_supports_current_image(info))

    def test_gpu_rejects_turing(self) -> None:
        info = {"compute_capabilities": [(7, 5), (7, 5)]}
        self.assertFalse(b.gpu_supports_current_image(info))

    def test_gpu_unknown_caps_allowed(self) -> None:
        self.assertTrue(b.gpu_supports_current_image({"compute_capabilities": []}))


if __name__ == "__main__":
    unittest.main()