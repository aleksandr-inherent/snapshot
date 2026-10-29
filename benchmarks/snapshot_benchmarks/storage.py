# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint object measurements using the PageBroker container's rclone."""

import json
import shlex

import yaml
from snapshot_e2e import k8s, lifecycle

COMPLETE = "\nBENCHMARK_STORAGE_COMPLETE\n"
MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


def checked_exec(namespace, pod, script, *, container):
    # The existing exec helper combines stdout/stderr and omits the exit status.
    # A marker is emitted only after a successful, bounded remote command.
    command = "timeout 300s bash -c " + shlex.quote(
        "set -euo pipefail\n" + script + "\nprintf '%s' " + shlex.quote(COMPLETE)
    )
    try:
        output = k8s.exec_payload(namespace, pod, command, container=container)
    except AssertionError:
        # The helper's missing-marker error includes unfiltered remote output.
        raise RuntimeError("Storage measurement failed or timed out") from None
    if not output.endswith(COMPLETE):
        # Do not include remote diagnostics: they may contain private configuration.
        raise RuntimeError("Storage measurement failed or timed out")
    return output[:-len(COMPLETE)]


def rclone(cfg, pod, settings, operation, key, *, limit=None):
    args = [
        "/usr/local/bin/rclone", operation,
        ":s3:" + settings["bucket"] + "/" + key,
        "--config", "/dev/null", "--s3-provider", "AWS", "--s3-env-auth",
        "--s3-profile", "default", "--s3-encoding", "None", "--stats", "0", "--s3-region", settings["region"],
        "--s3-endpoint", settings["endpoint"], "--retries", "1",
        "--low-level-retries", "1", "--contimeout", "20s", "--timeout", "60s",
    ]
    if operation == "size":
        args.append("--json")
    else:
        args += ["--head", str(limit + 1)]
    # Use only the mounted default profile, not ambient keys or instance roles.
    script = '''test -r "${AWS_SHARED_CREDENTIALS_FILE:?missing credentials file}"
ca=()
if [ -n "${AWS_CA_BUNDLE:-}" ]; then ca=(--ca-cert "$AWS_CA_BUNDLE"); fi
path_style=true
if [ "${RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING:-0}" = 1 ]; then path_style=false; fi
env -i PATH=/usr/local/bin:/usr/bin:/bin AWS_EC2_METADATA_DISABLED=true AWS_CONFIG_FILE=/dev/null \\
AWS_SHARED_CREDENTIALS_FILE="$AWS_SHARED_CREDENTIALS_FILE" ''' + shlex.join(args) + ' --s3-force-path-style="$path_style" "${ca[@]}"'
    text = checked_exec(cfg.snapshot_namespace, pod, script, container="pagebroker")
    if limit is not None and len(text.encode()) > limit:
        raise ValueError("Checkpoint metadata exceeds measurement limit")
    return text


def gpu_mode(manifest):
    cuda = (yaml.safe_load(manifest) or {}).get("cudaRestore") or {}
    if cuda.get("customStorage") is True:
        return "custom-storage"
    if cuda.get("pids"):
        return "file-based"
    return "unknown"


def s3_measurement(cfg, node, content, settings):
    pod = lifecycle.checkpoint_agent_pod(cfg.snapshot_e2e_config(), node)
    artifacts = []
    seen = set()
    for artifact in content["status"]["storage"]["artifacts"]:
        handle = artifact["artifactHandle"]
        # Only measure the confirmed container artifact, never a shared bucket root.
        expected = f"artifacts/{content['metadata']['uid']}/containers/{artifact['containerName']}"
        if handle != expected or handle in seen:
            raise ValueError("Unexpected or duplicate checkpoint artifact handle")
        seen.add(handle)
        prefix = "/".join(p for p in (settings["prefix"], handle) if p)
        index_text = rclone(cfg, pod, settings, "cat", prefix + "/index.json", limit=MAX_INDEX_BYTES)
        index = json.loads(index_text)
        files = index["files"]
        if not files or any(type(f.get("size")) is not int or f["size"] < 0 for f in files):
            raise ValueError("Invalid pre-upload file inventory")
        local_bytes = sum(f["size"] for f in files)
        remote = json.loads(rclone(cfg, pod, settings, "size", prefix + "/data"))
        if (type(remote.get("bytes")) is not int or type(remote.get("count")) is not int
                or remote["bytes"] != local_bytes or remote["count"] != len(files)):
            raise ValueError("S3 payload size/count does not match pre-upload inventory")
        manifest = rclone(cfg, pod, settings, "cat", prefix + "/data/manifest.yaml", limit=MAX_MANIFEST_BYTES)
        artifacts.append(dict(
            container=artifact["containerName"], handle=handle,
            local_payload_bytes=local_bytes, payload_bytes=remote["bytes"],
            object_count=remote["count"], metadata_bytes=len(index_text.encode()),
            gpu_checkpoint_mode=gpu_mode(manifest),
            files=files,
        ))
    if not artifacts:
        raise ValueError("No published S3 checkpoint artifacts")
    return dict(
        artifacts=artifacts,
        local_payload_bytes=sum(a["local_payload_bytes"] for a in artifacts),
        payload_bytes=sum(a["payload_bytes"] for a in artifacts),
        object_count=sum(a["object_count"] for a in artifacts),
        metadata_bytes=sum(a["metadata_bytes"] for a in artifacts),
        total_bytes=sum(a["payload_bytes"] + a["metadata_bytes"] for a in artifacts),
        gpu_checkpoint_mode=", ".join(sorted({a["gpu_checkpoint_mode"] for a in artifacts})),
    )
