from pathlib import Path
import subprocess

SCRIPT = Path("scripts/public_repo_audit.sh")
BASE = "215abfb217dbac7d5f1273282331b9b1866c0479"


def _run(command: list[str], cwd: Path, *, check: bool = True, env: dict[str, str] | None = None):
    return subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True, env=env)


def test_public_audit_is_read_only_fail_closed_and_scoped_to_project_history() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "gitleaks is required" not in source  # generic prerequisite loop avoids path disclosure
    assert 'command -v "$command"' in source
    assert f"upstream_base={BASE}" in source
    assert '"$upstream_base..HEAD"' in source
    assert "log -m --format= -z --name-only" in source
    assert "log -m --format= -z --numstat" in source
    assert "git ls-files --others --exclude-standard -z" in source
    assert "--no-ext-diff --no-textconv" in source
    assert "--redact" in source
    assert "upstream push must remain disabled" in source
    assert "LICENSE_GEMMA.txt" in source
    assert "third_party/aloha" in source
    assert "third_party/libero" in source
    assert "git reset" not in source
    assert "git checkout" not in source
    assert "git clean" not in source
    assert ".runtime/" not in source


def test_public_audit_rejects_forbidden_current_project_content() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden_contracts = [
        "private key material",
        "Windows machine hostname",
        "macOS user home",
        "Linux user home",
        "Windows user home",
        "RFC1918 address",
        "generated/private artifact filename",
        "binary project-added content",
    ]
    for contract in forbidden_contracts:
        assert contract in source


def test_public_audit_limits_historical_fixture_exception_to_original_test_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "historical_test_fixtures = {" in source
    assert '("history:tests/test_telemetry.py", "DESKTOP-" + "EXAMPLE")' in source
    assert '("history:tests/test_telemetry.py", "192" + ".168.1.2")' in source
    assert 'text = text.replace(fixture, "")' in source


def test_public_audit_shell_rejects_arguments_before_repository_checks(tmp_path: Path) -> None:
    result = _run([str(SCRIPT.resolve()), "unexpected"], tmp_path, check=False)
    assert result.returncode == 2
    assert "accepts no arguments" in result.stderr


def test_security_guidance_documents_loopback_ssh_and_raw_evidence_boundary() -> None:
    security = Path("SECURITY.md").read_text(encoding="utf-8")
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "policy port must remain loopback-only" in security
    assert "SSH local forwarding is its security boundary" in security
    assert "Never commit or attach weights" in security
    assert "Publish only newly constructed, allowlisted summaries" in security
    assert "make public-audit" in security
    assert "raw logs, or raw telemetry" in contributing


def test_baseline_security_workflow_prepares_exact_remotes_before_public_audit() -> None:
    workflow = Path(".github/workflows/baseline-security.yml").read_text(encoding="utf-8")
    prepare = workflow.index("Prepare audited remotes")
    audit = workflow.index("scripts/public_repo_audit.sh")
    assert prepare < audit
    assert "fetch-depth: 0" in workflow
    assert "persist-credentials: false" in workflow
    assert "git remote set-url origin https://github.com/therealjaysun/pi-robotics.git" in workflow
    assert "git remote add upstream https://github.com/Physical-Intelligence/openpi.git" in workflow
    assert "git remote set-url --push upstream DISABLED" in workflow
    assert "git fetch --no-tags upstream main:refs/remotes/upstream/main" in workflow
