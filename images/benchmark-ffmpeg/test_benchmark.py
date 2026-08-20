#!/usr/bin/env python3
"""Unit tests for the FFmpeg capacity benchmark."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("benchmark.py")
SPEC = importlib.util.spec_from_file_location("benchmark_ffmpeg", MODULE_PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def host(*, gpus: tuple[object, ...] = ()) -> object:
    return bench.HostProfile(
        cpu_count=8,
        affinity_cpu_count=8,
        affinity_cpus=tuple(range(8)),
        cpu_quota_cores=None,
        cpu_capacity=8.0,
        architecture="x86_64",
        ram_total_gb=64.0,
        ram_avail_gb=60.0,
        memory_limit_gb=None,
        pids_limit=None,
        pids_current=None,
        cgroup_cpu_max="max 100000",
        cgroup_cpuset_effective="0-7",
        cgroup_cpu_stat_before={},
        numa_nodes=(bench.NumaNode(0, tuple(range(8))),),
        numa_binding=False,
        gpus=gpus,
        physical_cpu_count=8,
        logical_cpu_count=8,
    )


class ScenarioTests(unittest.TestCase):
    def _audio(self, name: str):
        return next(item for item in bench.AUDIO_SCENARIOS if item.name == name)

    def test_audio_profiles_are_exact(self) -> None:
        profiles = {
            item.name: (
                item.codec,
                item.bitrate_kbps,
                item.compression_level,
                item.sample_rate_hz,
            )
            for item in bench.AUDIO_SCENARIOS
        }
        self.assertEqual(
            profiles,
            {
                "ogg_vorbis_160k": ("libvorbis", 160, None, 44_100),
                "opus_128k": ("libopus", 128, None, 48_000),
                "aac_128k": ("aac", 128, None, 44_100),
                "mp3_192k": ("libmp3lame", 192, None, 44_100),
                "flac_lossless": ("flac", None, 5, 44_100),
            },
        )

    def test_parse_frame_rate(self) -> None:
        self.assertEqual(bench.parse_frame_rate("24/1"), 24.0)
        self.assertEqual(bench.parse_frame_rate("30000/1001"), 30000 / 1001)
        self.assertEqual(bench.parse_frame_rate("0/0"), 0.0)

    def test_all_existing_video_scenarios_remain(self) -> None:
        self.assertEqual(
            [item.name for item in bench.VIDEO_SCENARIOS],
            [
                "cpu_h264_encode",
                "cpu_h265_encode",
                "cpu_h264_decode",
                "cpu_h265_decode",
                "gpu_h264_encode",
                "gpu_h264_decode",
                "gpu_h265_encode",
                "gpu_h265_decode",
            ],
        )

    def _video(self, name: str):
        return next(item for item in bench.VIDEO_SCENARIOS if item.name == name)

    def test_audio_command_is_fully_specified_and_cpu_only(self) -> None:
        command = bench.build_worker_command(
            self._audio("ogg_vorbis_160k"),
            Path("/source.flac"),
            123.5,
            gpu_index=None,
        )
        joined = " ".join(command)
        self.assertIn("-stream_loop -1", joined)
        self.assertIn("-map 0:a:0", joined)
        self.assertIn("-ar 44100 -ac 2", joined)
        self.assertIn("-threads:a 1", joined)
        self.assertIn("-c:a libvorbis -b:a 160k", joined)
        self.assertIn("-f null -", joined)
        self.assertNotIn("cuda", joined)

    def test_new_lossy_audio_encoders_in_command(self) -> None:
        for name, codec, bitrate in (
            ("opus_128k", "libopus", "128k"),
            ("aac_128k", "aac", "128k"),
            ("mp3_192k", "libmp3lame", "192k"),
        ):
            command = bench.build_worker_command(
                self._audio(name), Path("/source.flac"), 10.0, gpu_index=None
            )
            joined = " ".join(command)
            self.assertIn(f"-c:a {codec} -b:a {bitrate}", joined)
            self.assertNotIn("cuda", joined)
        opus = " ".join(
            bench.build_worker_command(
                self._audio("opus_128k"), Path("/source.flac"), 10.0, gpu_index=None
            )
        )
        self.assertIn("-ar 48000 -ac 2", opus)

    def test_video_decode_codec_is_an_input_option(self) -> None:
        command = bench.build_worker_command(
            self._video("cpu_h264_decode"),
            Path("/source.mp4"),
            12.0,
            gpu_index=None,
        )
        self.assertLess(command.index("-c:v"), command.index("-i"))
        self.assertEqual(command[command.index("-c:v") + 1], "h264")
        joined = " ".join(command)
        self.assertIn("-threads 1", joined)
        self.assertIn("-threads:v 1", joined)
        self.assertEqual(joined.count("-threads:v 1"), 1)

    def test_hevc_decode_uses_hevc_fixture_decoder(self) -> None:
        cpu = bench.build_worker_command(
            self._video("cpu_h265_decode"),
            Path("/source-hevc.mp4"),
            12.0,
            gpu_index=None,
        )
        self.assertEqual(cpu[cpu.index("-c:v") + 1], "hevc")
        gpu = bench.build_worker_command(
            self._video("gpu_h265_decode"),
            Path("/source-hevc.mp4"),
            12.0,
            gpu_index=0,
        )
        self.assertEqual(gpu[gpu.index("-c:v") + 1], "hevc_cuvid")

    def test_cpu_video_encode_uses_one_thread(self) -> None:
        for scenario in (self._video("cpu_h264_encode"), self._video("cpu_h265_encode")):
            command = bench.build_worker_command(
                scenario, Path("/source.mp4"), 12.0, gpu_index=None
            )
            joined = " ".join(command)
            self.assertIn("-threads 1", joined)
            self.assertLess(joined.index("-threads:v 1"), joined.index("-i"))
            self.assertGreater(joined.rindex("-threads:v 1"), joined.index("-i"))
            if scenario.codec == "libx265":
                self.assertIn("frame-threads=1:pools=none", joined)

    def test_gpu_video_uses_cuvid_for_encode_and_decode(self) -> None:
        for name, decoder, encoder in (
            ("gpu_h264_encode", "h264_cuvid", "h264_nvenc"),
            ("gpu_h264_decode", "h264_cuvid", None),
            ("gpu_h265_encode", "h264_cuvid", "hevc_nvenc"),
            ("gpu_h265_decode", "hevc_cuvid", None),
        ):
            command = bench.build_worker_command(
                self._video(name),
                Path("/source.mp4"),
                12.0,
                gpu_index=1,
            )
            self.assertLess(command.index("-c:v"), command.index("-i"))
            self.assertEqual(command[command.index("-c:v") + 1], decoder)
            joined = " ".join(command)
            self.assertIn("-hwaccel_output_format cuda", joined)
            if encoder:
                self.assertIn(f"-c:v {encoder}", joined)

    def test_gpu_search_anchor_is_one_session_per_gpu(self) -> None:
        host = bench.HostProfile(
            cpu_count=96,
            affinity_cpu_count=96,
            affinity_cpus=tuple(range(96)),
            cpu_quota_cores=None,
            cpu_capacity=96.0,
            architecture="x86_64",
            ram_total_gb=64.0,
            ram_avail_gb=64.0,
            memory_limit_gb=None,
            pids_limit=100000,
            pids_current=1,
            cgroup_cpu_max="max 100000",
            cgroup_cpuset_effective="0-95",
            cgroup_cpu_stat_before={},
            numa_nodes=(bench.NumaNode(0, tuple(range(96))),),
            numa_binding=False,
            gpus=tuple(bench.GpuInfo(i, "GPU") for i in range(4)),
            physical_cpu_count=48,
            logical_cpu_count=96,
        )
        encode_target, encode_max = bench.max_workers_for_host(
            host, self._video("gpu_h264_encode"), 4
        )
        decode_target, decode_max = bench.max_workers_for_host(
            host, self._video("gpu_h264_decode"), 4
        )
        self.assertEqual(encode_target, 4)
        self.assertEqual(decode_target, 4)
        self.assertGreaterEqual(encode_max, 16)
        self.assertGreaterEqual(decode_max, 16)
        self.assertEqual(bench.worker_ladder(encode_target, encode_max, backend="gpu"), [1, 4])

    def test_multi_gpu_assignment_is_round_robin(self) -> None:
        gpus = [bench.GpuInfo(2, "a"), bench.GpuInfo(7, "b")]
        self.assertEqual([bench.assign_gpu(i, gpus) for i in range(6)], [2, 7, 2, 7, 2, 7])

    def test_numa_assignment_is_proportional(self) -> None:
        nodes = (
            bench.NumaNode(0, (0, 1, 2, 3, 4, 5)),
            bench.NumaNode(1, (6, 7)),
        )
        plan = bench.assign_workers_to_nodes(8, nodes)
        self.assertEqual(sum(node.index == 0 for node in plan), 6)
        self.assertEqual(sum(node.index == 1 for node in plan), 2)

    def test_audio_codecs_have_no_gpu_requirement(self) -> None:
        encoders = {"libvorbis", "libopus", "aac", "libmp3lame", "flac"}
        for scenario in bench.AUDIO_SCENARIOS:
            self.assertEqual(scenario.backend, "cpu")
            self.assertTrue(bench.scenario_supported(scenario, encoders, set(), [])[0])


class CapacityTests(unittest.TestCase):
    def test_worker_ladder_uses_postgres_style_anchors(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFMPEG_BENCH_WORKERS", None)
            with patch.object(bench, "OVERSUBSCRIPTION", 1.0):
                self.assertEqual(bench.worker_ladder(96, 192), [1, 48, 96])
                self.assertEqual(bench.worker_ladder(8, 32), [1, 4, 8])
                self.assertEqual(bench.worker_ladder(1, 8), [1])
                self.assertEqual(bench.worker_ladder(2, 8), [1, 2])
                self.assertEqual(bench.worker_ladder(4096, 8192), [1, 2048, 4096])

    def test_oversubscription_adds_a_single_extra_point(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FFMPEG_BENCH_WORKERS", None)
            with patch.object(bench, "OVERSUBSCRIPTION", 2.0):
                self.assertEqual(bench.worker_ladder(8, 32), [1, 4, 8, 16])
                self.assertEqual(bench.worker_ladder(96, 96), [1, 48, 96])

    def test_gpu_ladder_starts_at_one_session_per_gpu(self) -> None:
        with patch.object(bench, "OVERSUBSCRIPTION", 1.0):
            self.assertEqual(bench.worker_ladder(4, 32, backend="gpu"), [1, 4])
            self.assertEqual(bench.worker_ladder(1, 8, backend="gpu"), [1])

    def test_gpu_doubles_until_throughput_drops(self) -> None:
        spec = next(s for s in bench.VIDEO_SCENARIOS if s.name == "gpu_h264_encode")
        improving = [
            bench.ScalingStep(1, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 187, 187.0, None, None, 0.0, 1, 0, 0),
            ]),
            bench.ScalingStep(4, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 370, 370.0, None, None, 0.0, 4, 0, 0),
            ]),
            bench.ScalingStep(8, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 647, 647.0, None, None, 0.0, 8, 0, 0),
            ]),
        ]
        self.assertEqual(bench.maybe_next_workers(spec, improving[:1], maximum=32, anchor=4), 2)
        self.assertEqual(bench.maybe_next_workers(spec, improving[:2], maximum=32, anchor=4), 8)
        self.assertEqual(bench.maybe_next_workers(spec, improving, maximum=32, anchor=4), 16)
        dropped = improving + [
            bench.ScalingStep(16, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 619, 619.0, None, None, 0.0, 16, 0, 0),
            ]),
        ]
        self.assertIsNone(bench.maybe_next_workers(spec, dropped, maximum=32, anchor=4))

    def test_cpu_ht_probe_only_when_physical_cores_still_scale(self) -> None:
        spec = next(s for s in bench.VIDEO_SCENARIOS if s.name == "cpu_h264_encode")
        scaled = [
            bench.ScalingStep(24, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 480, 480.0, None, None, 0.0, 24, 0, 0),
            ]),
            bench.ScalingStep(48, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 641, 641.0, None, None, 0.0, 48, 0, 0),
            ]),
        ]
        self.assertEqual(bench.maybe_next_workers(spec, scaled, maximum=96, anchor=48), 96)
        saturated = [
            bench.ScalingStep(24, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 11200, 11200.0, None, None, 0.0, 24, 0, 0),
            ]),
            bench.ScalingStep(48, 1.0, 0, [
                bench.RepetitionResult(1, 1.0, 10386, 10386.0, None, None, 0.0, 48, 0, 0),
            ]),
        ]
        self.assertIsNone(bench.maybe_next_workers(spec, saturated, maximum=96, anchor=48))

    def test_explicit_worker_ladder_is_capped_and_deduplicated(self) -> None:
        with patch.dict(os.environ, {"FFMPEG_BENCH_WORKERS": "1,4,4,999"}):
            self.assertEqual(bench.worker_ladder(8, 16), [1, 4, 16])

    def test_cpu_quota_parsing(self) -> None:
        with patch.object(bench, "read_text", return_value="150000 100000"):
            value, quota = bench.cgroup_cpu_profile()
        self.assertEqual(value, "150000 100000")
        self.assertEqual(quota, 1.5)

    def test_cpu_list_parser(self) -> None:
        self.assertEqual(bench.parse_cpu_list("0-3,8,10-11"), (0, 1, 2, 3, 8, 10, 11))

    def test_resource_caps_limit_process_count(self) -> None:
        constrained = bench.HostProfile(
            **{
                **host().__dict__,
                "ram_avail_gb": 1.0,
                "pids_limit": 40,
                "pids_current": 16,
            }
        )
        target, maximum = bench.max_workers_for_host(constrained, bench.AUDIO_SCENARIOS[0], 0)
        self.assertEqual(target, 8)
        self.assertEqual(maximum, 2)

    def test_numa_wrapper_is_optional(self) -> None:
        command = ["ffmpeg", "-i", "x"]
        node = bench.NumaNode(3, (12, 13))
        self.assertIs(bench.wrap_for_numa(command, node, False), command)
        self.assertEqual(
            bench.wrap_for_numa(command, node, True)[:3],
            ["numactl", "--cpunodebind=3", "--membind=3"],
        )


class MeasurementTests(unittest.TestCase):
    def test_aggregate_uses_group_makespan_not_sum_of_worker_rates(self) -> None:
        result = bench.summarize_repetition(
            bench.AUDIO_SCENARIOS[0],
            workers=2,
            media_duration_sec=100.0,
            repetition=1,
            finish_times=[10.0, 12.0],
            return_codes=[0, 0],
        )
        self.assertEqual(result.processed_audio_seconds, 200.0)
        self.assertAlmostEqual(result.audio_seconds_per_sec, 200.0 / 12.0)
        self.assertNotAlmostEqual(
            result.audio_seconds_per_sec, 100.0 / 10.0 + 100.0 / 12.0
        )
        self.assertAlmostEqual(result.finish_spread_sec, 2.0)

    def test_video_reports_frame_count_and_fps(self) -> None:
        result = bench.summarize_repetition(
            next(s for s in bench.VIDEO_SCENARIOS if s.name == "cpu_h264_encode"),
            workers=2,
            media_duration_sec=10.0,
            repetition=1,
            finish_times=[10.0, 12.0],
            return_codes=[0, 0],
            video_fps=30.0,
        )
        self.assertEqual(result.processed_frames, 600)
        self.assertEqual(result.aggregate_fps, 50.0)
        self.assertIsNone(result.processed_audio_seconds)

    def test_failed_worker_contributes_no_completed_media(self) -> None:
        result = bench.summarize_repetition(
            bench.AUDIO_SCENARIOS[0],
            workers=2,
            media_duration_sec=100.0,
            repetition=1,
            finish_times=[10.0, 12.0],
            return_codes=[0, 1],
            error_text="encoder failed",
        )
        self.assertEqual(result.processed_audio_seconds, 100.0)
        self.assertAlmostEqual(result.audio_seconds_per_sec, 100.0 / 12.0)
        self.assertEqual(result.successful_workers, 1)
        self.assertEqual(result.failed_workers, 1)

    def test_step_preserves_raw_repetitions(self) -> None:
        values = [10.0, 14.0, 12.0]

        def fake_run(*args: object, **kwargs: object) -> object:
            index = int(args[-2]) - 1
            return bench.RepetitionResult(
                repetition=index + 1,
                wall_time_sec=1.0,
                processed_frames=None,
                aggregate_fps=None,
                processed_audio_seconds=values[index],
                audio_seconds_per_sec=values[index],
                finish_spread_sec=0.1,
                successful_workers=2,
                failed_workers=0,
                timed_out_workers=0,
            )

        with (
            patch.object(bench, "run_group_once", side_effect=fake_run),
            patch.object(bench, "MAX_REPETITIONS", 3),
        ):
            step = bench.run_scaling_step(
                bench.AUDIO_SCENARIOS[0], [Path("x")], 2, 30.0, [], host(), 3, 999999.0,
            )
        self.assertEqual(len(step.repetitions), 3)
        self.assertEqual(
            [run.audio_seconds_per_sec for run in step.repetitions], values
        )
        self.assertFalse(hasattr(step, "throughput_mean"))

    def test_variable_step_adds_repetitions_up_to_configured_cap(self) -> None:
        values = [10.0, 14.0, 12.0, 12.0]

        def fake_run(*args: object, **kwargs: object) -> object:
            index = int(args[-2]) - 1
            value = values[index]
            return bench.RepetitionResult(
                index + 1, 1.0, None, None, value, value,
                0.1, 2, 0, 0,
            )

        with (
            patch.object(bench, "run_group_once", side_effect=fake_run),
            patch.object(bench, "MAX_REPETITIONS", 4),
            patch.object(bench, "CV_THRESHOLD", 0.01),
        ):
            step = bench.run_scaling_step(
                bench.AUDIO_SCENARIOS[0], [Path("x")], 2, 30.0, [], host(), 3, 999999.0,
            )
        self.assertEqual(len(step.repetitions), 4)

    def test_failed_run_stops_repetitions_immediately(self) -> None:
        failed = bench.RepetitionResult(
            1, 1.0, None, None, 1.0, 1.0, 0.1, 1, 1, 0, "fail"
        )
        later = bench.RepetitionResult(
            2, 1.0, None, None, 80.0, 80.0, 0.1, 2, 0, 0
        )
        with patch.object(bench, "run_group_once", side_effect=[failed, later]) as run:
            step = bench.run_scaling_step(
                bench.AUDIO_SCENARIOS[0], [Path("x")], 2, 30.0, [], host(), 3, 999999.0,
            )
        self.assertEqual(step.failed_workers, 1)
        self.assertEqual(len(step.repetitions), 1)
        self.assertEqual(run.call_count, 1)

    def test_runtime_probe_failure_skips_scenario(self) -> None:
        calibration = bench.RepetitionResult(
            0, 1.0, None, None, 0.0, 0.0, 0.0, 0, 1, 0, "probe failed",
        )
        with patch.object(bench, "run_group_once", return_value=calibration):
            result = bench.run_scenario(
                bench.AUDIO_SCENARIOS[0], [Path("x")], host(), [], 999999.0,
            )
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "probe failed")

    def test_calibration_targets_measured_wall_duration(self) -> None:
        calibration = bench.RepetitionResult(
            0, 1.0, None, None, 20.0, 20.0, 0.0, 1, 0, 0,
        )
        self.assertEqual(
            bench.calibrated_duration(bench.AUDIO_SCENARIOS[0], calibration, 15.0),
            min(bench.MAX_MEDIA_DURATION_SEC, 20.0 * bench.TARGET_REPETITION_SEC),
        )

    def test_group_pilot_resizes_to_target_wall_time(self) -> None:
        pilot = bench.RepetitionResult(
            -1, 20.0, None, None, 40.0, 2.0, 0.0, 4, 0, 0
        )
        with patch.object(bench, "run_group_once", return_value=pilot):
            duration, observed = bench.calibrate_group_duration(
                bench.AUDIO_SCENARIOS[0],
                [Path("x")],
                4,
                10.0,
                [],
                host(),
                999999.0,
            )
        self.assertIs(observed, pilot)
        self.assertEqual(
            duration,
            bench.clamp_media_duration(
                10.0 * bench.TARGET_REPETITION_SEC / pilot.wall_time_sec
            ),
        )


class RawSchemaTests(unittest.TestCase):
    def test_scenario_schema_has_no_derived_rollups(self) -> None:
        payload = bench.scenario_to_dict(
            bench.skipped_result(bench.AUDIO_SCENARIOS[0], "test")
        )
        for key in (
            "recommended_workers",
            "peak_audio_seconds_per_sec",
            "peak_tracks_per_hour",
            "peak_aggregate_fps",
            "peak_aggregate_realtime_factor",
            "max_realtime_workers",
            "single_stream_fps",
        ):
            self.assertNotIn(key, payload)


if __name__ == "__main__":
    unittest.main()
