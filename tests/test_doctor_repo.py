import os
from pathlib import Path
import subprocess

SCRIPT = Path("scripts/doctor_repo.sh").resolve()
REPO = SCRIPT.parent.parent


def _fake_commands(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text(
        f"""#!/bin/sh
set -eu
case "$*" in
  "rev-parse --show-toplevel") printf '%s\\n' '{REPO}' ;;
  "status --porcelain=v1 --untracked-files=all") [ -z "${{FAKE_STATUS_ERROR:-}}" ] || exit 2; printf '%s' "${{FAKE_DIRTY:-}}" ;;
  "remote get-url origin"|"remote get-url --push origin") printf '%s\\n' 'https://github.com/therealjaysun/pi-robotics.git' ;;
  "remote get-url upstream") printf '%s\\n' 'https://github.com/Physical-Intelligence/openpi.git' ;;
  "remote get-url --push upstream") printf '%s\\n' "${{FAKE_UPSTREAM_PUSH:-DISABLED}}" ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/bin/sh
set -eu
case "$1 $2" in
  "auth status") exit 0 ;;
  "repo view") printf '%s\n' 'therealjaysun/pi-robotics|PUBLIC' ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    git.chmod(0o755)
    gh.chmod(0o755)
    return {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}


def test_repository_doctor_passes_verified_clean_public_checkout(tmp_path: Path) -> None:
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=_fake_commands(tmp_path))
    assert result.returncode == 0
    assert "Repository doctor passed" in result.stdout


def test_repository_doctor_fails_closed_with_recovery_for_dirty_tree(tmp_path: Path) -> None:
    env = _fake_commands(tmp_path)
    env["FAKE_DIRTY"] = "?? untracked"
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "worktree is not clean" in result.stderr
    assert "Recovery: Run: git status --short" in result.stderr


def test_repository_doctor_requires_disabled_upstream_push(tmp_path: Path) -> None:
    env = _fake_commands(tmp_path)
    env["FAKE_UPSTREAM_PUSH"] = "https://github.com/Physical-Intelligence/openpi.git"
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "upstream push is not disabled" in result.stderr
    assert "git remote set-url --push upstream DISABLED" in result.stderr


def test_repository_doctor_fails_when_git_status_cannot_run(tmp_path: Path) -> None:
    env = _fake_commands(tmp_path)
    env["FAKE_STATUS_ERROR"] = "1"
    result = subprocess.run([SCRIPT], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "git could not inspect the worktree" in result.stderr
