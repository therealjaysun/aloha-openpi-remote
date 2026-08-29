import os
from pathlib import Path
import subprocess

SCRIPT = Path("scripts/pr_status.sh").resolve()


def _fake_gh(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/bin/sh
set -eu
if [ "$1 $2" = "auth status" ]; then
  exit 0
fi
[ "$1 $2" = "pr view" ] || exit 2
number=$3
case "$number" in
  1) base=main; head=codex/00-bootstrap ;;
  2) base=codex/00-bootstrap; head=codex/01-mac-simulation ;;
  3) base=codex/01-mac-simulation; head=codex/02-remote-gpu-server ;;
  4) base=codex/02-remote-gpu-server; head=codex/03-secure-connectivity ;;
  5) base=codex/03-secure-connectivity; head=codex/04-end-to-end-control ;;
  6) base=codex/04-end-to-end-control; head=codex/05-observability ;;
  7) base=codex/05-observability; head=codex/06-hardening-docs ;;
  *) exit 2 ;;
esac
state=OPEN
[ "${BAD_PR:-}" != "$number" ] || state=CLOSED
checks=SUCCESS,SKIPPED
[ "${NO_CHECKS:-}" != "$number" ] || checks=
printf '%s|%s|false|%s|%s|false|%s\n' "$number" "$state" "$head" "$base" "$checks"
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}


def test_pr_status_accepts_exact_green_manual_stack(tmp_path: Path) -> None:
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=_fake_gh(tmp_path))
    assert result.returncode == 0
    assert result.stdout.count(" passed: ") == 8
    assert "all seven pull requests" in result.stdout


def test_pr_status_fails_closed_for_non_open_pr(tmp_path: Path) -> None:
    env = _fake_gh(tmp_path)
    env["BAD_PR"] = "4"
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "PR 4 is CLOSED" in result.stderr
    assert "gh pr reopen 4" in result.stderr


def test_pr_status_requires_reported_checks(tmp_path: Path) -> None:
    env = _fake_gh(tmp_path)
    env["NO_CHECKS"] = "7"
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "PR 7 has no reported checks" in result.stderr
    assert "gh pr checks 7" in result.stderr
