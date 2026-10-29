# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from snapshot_benchmarks import measurement, storage, timing, report, run, cli
from snapshot_benchmarks.engines import ModelSpec
from test_benchmarks import config, cluster
from test_s3 import SETTINGS


def local_exec(namespace, pod, command, **kwargs):
    return subprocess.check_output(["bash", "-c", command], text=True)


def test_model_directory_measures_regular_bytes_with_safe_paths(monkeypatch, tmp_path):
    root = tmp_path / "model space ' $(touch SHOULD_NOT_EXIST)"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"123456")
    (root / "config\nfile").write_bytes(b"123")
    (root / "link").symlink_to(root / "weights.bin")
    (root / "empty").mkdir()
    monkeypatch.setattr(storage.k8s, "exec_command", local_exec)
    value = measurement.model_directory_measurement(
        config(), SimpleNamespace(metadata=SimpleNamespace(name="pod")), str(root), tmp_path,
    )
    assert value["total_bytes"] == 9
    assert len(value["files"]) == 2
    assert value["symlinks"] == "excluded"
    assert json.loads((tmp_path / "model-directory.json").read_text()) == value
    (root / "weights.bin").unlink()
    (root / "config\nfile").unlink()
    assert measurement.model_directory_measurement(
        config(), SimpleNamespace(metadata=SimpleNamespace(name="pod")), str(root), tmp_path,
    )["total_bytes"] == 0


@pytest.mark.parametrize("output", ["", "partial output", "failed\n", storage.COMPLETE + "error"])
def test_remote_measurement_rejects_incomplete_output(monkeypatch, output):
    monkeypatch.setattr(storage.k8s, "exec_command", Mock(return_value=storage.k8s.PAYLOAD_MARKER + "\n" + output))
    with pytest.raises(RuntimeError, match="failed or timed out"):
        storage.checked_exec("ns", "pod", "false", container="pagebroker")


def test_remote_measurement_ignores_shell_startup_output(monkeypatch):
    def execute(namespace, pod, command, **kwargs):
        return "Shell startup notice\n" + local_exec(namespace, pod, command, **kwargs)
    monkeypatch.setattr(storage.k8s, "exec_command", execute)
    output = storage.checked_exec(
        "ns", "pod", "printf '%s' '{\"bytes\":1024,\"count\":1}'", container="pagebroker",
    )
    assert json.loads(output) == {"bytes": 1024, "count": 1}


def test_remote_measurement_missing_payload_marker_hides_diagnostics(monkeypatch):
    monkeypatch.setattr(storage.k8s, "exec_command", Mock(return_value="private diagnostic"))
    with pytest.raises(RuntimeError, match="^Storage measurement failed or timed out$") as error:
        storage.checked_exec("ns", "pod", "false", container="pagebroker")
    assert error.value.__suppress_context__


def test_remote_measurement_does_not_mask_failed_commands(monkeypatch):
    def execute(namespace, pod, command, **kwargs):
        return subprocess.run(["bash", "-c", command], capture_output=True, text=True).stdout
    monkeypatch.setattr(storage.k8s, "exec_command", execute)
    with pytest.raises(RuntimeError):
        storage.checked_exec("ns", "pod", "false\nprintf unreachable", container="pagebroker")


@pytest.mark.parametrize("virtual", ["0", "1"])
def test_rclone_uses_mounted_profile_and_ca_without_ambient_keys(monkeypatch, tmp_path, virtual):
    binary = tmp_path / "rclone"
    binary.write_text("#!/usr/bin/python3\nimport os,sys,json\nprint(json.dumps(dict(args=sys.argv[1:],env=dict(os.environ))))\n")
    binary.chmod(0o755)
    credentials = tmp_path / "credentials"
    credentials.write_text("[default]\naws_access_key_id=FAKE\naws_secret_access_key=FAKE\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "SHOULD_NOT_LEAK")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SHOULD_NOT_LEAK")
    monkeypatch.setenv("AWS_CA_BUNDLE", "/mounted/ca file.pem")
    monkeypatch.setenv("RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING", virtual)

    def execute(namespace, pod, command, **kwargs):
        assert kwargs["container"] == "pagebroker"
        return local_exec(namespace, pod, command.replace("/usr/local/bin/rclone", str(binary)))
    monkeypatch.setattr(storage.k8s, "exec_command", execute)
    value = json.loads(storage.rclone(config(), "pod", SETTINGS, "size", "prefix/quoted ' key"))
    assert "AWS_ACCESS_KEY_ID" not in value["env"]
    assert value["env"]["AWS_SHARED_CREDENTIALS_FILE"] == str(credentials)
    assert value["env"]["AWS_EC2_METADATA_DISABLED"] == "true"
    assert "SHOULD_NOT_LEAK" not in json.dumps(value)
    assert "--s3-force-path-style=" + ("false" if virtual == "1" else "true") in value["args"]
    assert value["args"][-2:] == ["--ca-cert", "/mounted/ca file.pem"]
    assert value["args"][1] == ":s3:test-bucket/prefix/quoted ' key"


@pytest.fixture
def s3_objects(monkeypatch):
    content = dict(metadata=dict(uid="uid"), status=dict(storage=dict(artifacts=[
        dict(containerName=name, artifactHandle=f"artifacts/uid/containers/{name}")
        for name in ("main", "worker")
    ])))
    index = dict(files=[dict(path="manifest.yaml", size=20), dict(path="pages.img", size=80)])
    remote = dict(bytes=100, count=2)
    calls = []

    def rclone(cfg, pod, settings, operation, key, **kwargs):
        calls.append((operation, key))
        if operation == "size":
            return json.dumps(remote)
        if key.endswith("index.json"):
            return json.dumps(index)
        return "cudaRestore:\n  pids: [123]\n  customStorage: true\n"
    monkeypatch.setattr(storage, "rclone", rclone)
    monkeypatch.setattr(storage.lifecycle, "checkpoint_agent_pod", Mock(return_value="agent"))
    return content, index, remote, calls


def test_s3_size_aggregates_only_published_container_payloads(s3_objects):
    content, index, _, calls = s3_objects
    value = storage.s3_measurement(config(), "node", content, SETTINGS)
    assert value["payload_bytes"] == value["local_payload_bytes"] == 200
    assert value["object_count"] == 4
    assert value["metadata_bytes"] == 2 * len(json.dumps(index).encode())
    assert value["total_bytes"] == 200 + value["metadata_bytes"]
    assert value["gpu_checkpoint_mode"] == "custom-storage"
    assert [key for op, key in calls if op == "size"] == [
        f"benchmarks/artifacts/uid/containers/{name}/data" for name in ("main", "worker")
    ]


@pytest.mark.parametrize("failure", ["bytes", "count", "negative", "handle", "duplicate"])
def test_s3_size_mismatch_and_invalid_inventory_fail(s3_objects, failure):
    content, index, remote, _ = s3_objects
    if failure in ("bytes", "count"):
        remote[failure] += 1
    elif failure == "negative":
        index["files"][0]["size"] = -1
    elif failure == "handle":
        content["status"]["storage"]["artifacts"][0]["artifactHandle"] = "artifacts"
    else:
        content["status"]["storage"]["artifacts"].append(copy.deepcopy(content["status"]["storage"]["artifacts"][0]))
    with pytest.raises(ValueError):
        storage.s3_measurement(config(), "node", content, SETTINGS)


@pytest.mark.parametrize("manifest, expected", [
    ("cudaRestore: {pids: [1], customStorage: true}", "custom-storage"),
    ("cudaRestore: {pids: [1]}", "file-based"),
    ("{}", "unknown"),
])
def test_gpu_mode_comes_from_checkpoint_manifest(manifest, expected):
    assert storage.gpu_mode(manifest) == expected


@pytest.mark.parametrize("duration, expected", [(2, 2), (0, None), (None, None), (-1, None)])
def test_effective_throughput_uses_decimal_gb(duration, expected):
    assert timing.throughput(4_000_000_000, duration) == expected


def test_report_keeps_attempt_rates_and_unknown_historical_sizes(tmp_path):
    with cluster() as mocks:
        mocks["metadata.collect_environment"].return_value.storage_type = "s3"
        mocks["metadata.collect_environment"].return_value.s3 = SETTINGS
        content = dict(metadata=dict(name="content", uid="uid"),
                       spec=dict(storage=dict(storeID="store-test"), source=dict(podRef=dict(containers=["main"]))),
                       status=dict(storage=dict(artifacts=[dict(containerName="main", artifactHandle="artifact")])) )
        mocks["lifecycle.wait_for_snapshot_ready"].return_value = ({}, content)
        mocks["measurement.source_measurement"].return_value[0]["model_directory_bytes"] = 5_000_000_000
        mocks["measurement.checkpoint_measurement"].return_value = (4_000_000_000, 20, dict(
            payload_bytes=4_000_000_000, local_payload_bytes=4_000_000_000,
            publication_s=2, publication_gbps=2, gpu_checkpoint_mode="file-based",
        ))
        mocks["measurement.restored_measurement"].side_effect = [
            dict(total_s=10, agent_s=8, setup_s=5, staging_s=t, other_setup_s=5-t,
                 criu_s=2, cuda_s=1, wake_handoff_s=2) for t in (1, 2, 4)
        ]
        result = run.run_benchmark(replace(config(), storage_type="s3"), cli.ENGINES["vllm"],
                                   ModelSpec("Test", "/hf-cache/test"), image="test", run_id="rates", evidence_dir=tmp_path)
    assert [a.measurement["staging_gbps"] for a in result.restore_attempts] == [4, 2, 1]
    assert json.loads((tmp_path / "rates/restore-01/metrics.json").read_text())["staging_gbps"] == 4
    text = report.render([result.to_json_dict()])
    assert "2.33 ± 1.53" in text  # Mean of per-attempt rates, not bytes / mean duration.
    assert "5.00 GB" in text and "file-based" in text
    assert "Publication (GB/s)" in text
    assert "Weights (reported)" in report.render([dict(schema_version=1, model={}, environment={})])
