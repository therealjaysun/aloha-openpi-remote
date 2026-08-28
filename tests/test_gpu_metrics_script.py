import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest

SCRIPT = Path("scripts/collect_gpu_metrics.sh")
SHA = "a" * 40
PROFILE = "pi0_aloha_sim"
RUN_ID = "b" * 32


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _stage_sampler(
    tmp_path: Path, *, flock_busy: bool = False, fail_identity_after: int = 0
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    runtime = repo / ".runtime"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    runtime.mkdir(mode=0o700)
    fake_bin.mkdir()
    staged = scripts / SCRIPT.name
    shutil.copy2(SCRIPT, staged)
    staged.chmod(0o755)

    _write_executable(
        repo / ".venv/bin/python",
        """#!/usr/bin/env bash
set -eu
if (( $# == 2 )); then
    printf '%s\\t%s\\n' '2000000000' '2026-08-28T08:00:01.000Z'
    exit 0
fi
count=0
if [[ -f \"$FAKE_IDENTITY_COUNT\" ]]; then count=\"$(cat \"$FAKE_IDENTITY_COUNT\")\"; fi
count=$((count + 1))
printf '%s\\n' \"$count\" >\"$FAKE_IDENTITY_COUNT\"
if (( FAKE_FAIL_IDENTITY_AFTER > 0 && count > FAKE_FAIL_IDENTITY_AFTER )); then exit 1; fi
printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \\
    \"$FAKE_SERVER_PID\" \"$FAKE_PROFILE\" \"$FAKE_SOURCE_SHA\" \\
    \"$((1000000000 + count * 100000000))\" '2026-08-28T08:00:00.000Z' '987654'
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
set -eu
printf 'called\\n' >>\"$FAKE_SMI_COUNT\"
printf '1234, 56\\n'
""",
    )
    _write_executable(
        fake_bin / "timeout",
        """#!/usr/bin/env bash
set -eu
shift 3
exec \"$@\"
""",
    )
    _write_executable(
        fake_bin / "flock",
        """#!/usr/bin/env bash
set -eu
[[ \"${FAKE_FLOCK_BUSY:-0}\" != 1 ]]
""",
    )
    _write_executable(
        fake_bin / "stat",
        """#!/usr/bin/env bash
set -eu
printf '%s:700\\n' \"$(id -u)\"
""",
    )
    _write_executable(
        fake_bin / "realpath",
        """#!/usr/bin/env bash
set -eu
[[ "${1-}" == -e ]] && shift
[[ "${1-}" == -- ]] && shift
exec /bin/realpath "$1"
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_FLOCK_BUSY": "1" if flock_busy else "0",
        "FAKE_IDENTITY_COUNT": str(tmp_path / "identity-count"),
        "FAKE_FAIL_IDENTITY_AFTER": str(fail_identity_after),
        "FAKE_SERVER_PID": "4242",
        "FAKE_PROFILE": PROFILE,
        "FAKE_SOURCE_SHA": SHA,
        "FAKE_SMI_COUNT": str(tmp_path / "smi-count"),
    }
    return staged, environment


def _args(script: Path, output: Path) -> list[str]:
    return [str(script), str(output), RUN_ID, PROFILE, "4242", SHA, "0.1"]


@pytest.mark.parametrize(("stop_signal", "expected_status"), [(signal.SIGHUP, 129), (signal.SIGTERM, 143)])
def test_sampler_writes_private_sanitized_jsonl_and_stops_on_signal(
    tmp_path: Path, stop_signal: signal.Signals, expected_status: int
) -> None:
    script, environment = _stage_sampler(tmp_path)
    output = script.parent.parent / ".runtime/metrics.jsonl"
    process = subprocess.Popen(_args(script, output), env=environment, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if output.exists() and output.read_text(encoding="utf-8").count("\n") >= 2:
            break
        assert process.poll() is None, process.stderr.read() if process.stderr is not None else ""
        time.sleep(0.02)
    else:
        process.kill()
        pytest.fail("sampler did not write a GPU sample")

    process.send_signal(stop_signal)
    _, stderr = process.communicate(timeout=5)
    assert process.returncode == expected_status, stderr
    assert output.stat().st_mode & 0o777 == 0o600
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    samples = [row for row in rows if row["event"] == "gpu_sample"]
    assert rows[0]["event"] == "sampler_started"
    assert rows[-1]["event"] == "sampler_stopped"
    assert rows[-1]["status"] == "interrupted"
    assert samples
    assert all(sample["memory_used_mib"] == 1234 for sample in samples)
    assert all(sample["utilization_percent"] == 56 for sample in samples)
    assert all(sample["server_rss_kib"] == 987654 for sample in samples)
    assert len(samples) == Path(environment["FAKE_SMI_COUNT"]).read_text(encoding="utf-8").count("called")
    forbidden = {"hostname", "username", "gpu_name", "gpu_uuid", "bus_id", "ip", "output_path"}
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_sampler_fails_closed_when_server_identity_changes(tmp_path: Path) -> None:
    script, environment = _stage_sampler(tmp_path, fail_identity_after=3)
    output = script.parent.parent / ".runtime/metrics.jsonl"
    result = subprocess.run(
        _args(script, output), env=environment, capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["sampler_started", "gpu_sample", "sampler_stopped"]
    assert rows[-1]["status"] == "failed"
    assert "identity changed" in result.stderr


def test_sampler_refuses_duplicate_owner_without_creating_output(tmp_path: Path) -> None:
    script, environment = _stage_sampler(tmp_path, flock_busy=True)
    output = script.parent.parent / ".runtime/metrics.jsonl"
    result = subprocess.run(
        _args(script, output), env=environment, capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode == 1
    assert "already owns" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("position", "value", "message"),
    [
        (1, "bad/id", "Run ID must be"),
        (2, "unknown", "Invalid profile"),
        (3, "1", "Invalid server PID"),
        (4, "not-a-sha", "Invalid source SHA"),
        (5, "0", "Interval must be"),
    ],
)
def test_sampler_rejects_invalid_identity_arguments(tmp_path: Path, position: int, value: str, message: str) -> None:
    output = tmp_path / "metrics.jsonl"
    arguments = [str(output), RUN_ID, PROFILE, "4242", SHA, "1"]
    arguments[position] = value
    result = subprocess.run([str(SCRIPT), *arguments], capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 2
    assert message in result.stderr


def test_sampler_source_has_bounded_private_single_query_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "umask 077" in source
    assert "flock -n 9" in source
    assert "process_record import verify_record" in source
    assert "timeout --signal=TERM --kill-after=2s 5s" in source
    assert source.count('5s "$smi"') == 1
    assert "--query-gpu=memory.used,utilization.gpu" in source
    assert 'Path("/proc") / str(record.pid) / "status"' in source
    assert '"server_rss_kib"' in source
    assert "trap 'exit 129' HUP" in source
    assert "--loop" not in source
    assert "hostname" not in source
    assert "gpu_uuid" not in source
    assert "bus_id" not in source
