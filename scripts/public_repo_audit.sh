#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

[[ $# -eq 0 ]] || { echo 'public_repo_audit.sh accepts no arguments.' >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
upstream_base=215abfb217dbac7d5f1273282331b9b1866c0479
expected_origin=https://github.com/therealjaysun/pi-robotics.git
expected_upstream=https://github.com/Physical-Intelligence/openpi.git

fail() {
    echo "Public repository audit failed: $1" >&2
    exit 1
}

for command in git gitleaks mktemp python3; do
    command -v "$command" >/dev/null || fail "$command is required"
done
cd "$repo_root"
[[ "$(git rev-parse --show-toplevel 2>/dev/null)" == "$repo_root" ]] || fail 'repository root is not canonical'
git rev-parse --verify "$upstream_base^{commit}" >/dev/null 2>&1 || fail 'pinned upstream base is unavailable'
git merge-base --is-ancestor "$upstream_base" HEAD || fail 'pinned upstream base is not an ancestor of HEAD'
git rev-parse --verify upstream/main^{commit} >/dev/null 2>&1 || fail 'upstream/main is unavailable'
[[ "$(git merge-base HEAD upstream/main)" == "$upstream_base" ]] || fail 'upstream merge base differs from the audited pin'

[[ "$(git remote get-url origin)" == "$expected_origin" ]] || fail 'origin fetch URL is not the independent project'
[[ "$(git remote get-url --push origin)" == "$expected_origin" ]] || fail 'origin push URL is not the independent project'
[[ "$(git remote get-url upstream)" == "$expected_upstream" ]] || fail 'upstream fetch URL is not official OpenPI'
[[ "$(git remote get-url --push upstream)" == DISABLED ]] || fail 'upstream push must remain disabled'

for license in LICENSE LICENSE_GEMMA.txt; do
    [[ -f "$license" && ! -L "$license" ]] || fail "$license is missing or is not a regular file"
    git diff --quiet "$upstream_base" -- "$license" || fail "$license differs from the pinned upstream version"
done
[[ -f .gitmodules && ! -L .gitmodules ]] || fail '.gitmodules is missing or unsafe'
git diff --quiet "$upstream_base" -- .gitmodules third_party/aloha third_party/libero ||
    fail 'submodule URLs or pinned revisions differ from upstream'
[[ "$(git config -f .gitmodules --get submodule.third_party/aloha.url)" == https://github.com/Physical-Intelligence/aloha.git ]] ||
    fail 'ALOHA submodule URL changed'
[[ "$(git config -f .gitmodules --get submodule.third_party/libero.url)" == https://github.com/Lifelong-Robot-Learning/LIBERO.git ]] ||
    fail 'LIBERO submodule URL changed'
for submodule in third_party/aloha third_party/libero; do
    [[ "$(git ls-files -s -- "$submodule" | awk '{print $1}')" == 160000 ]] || fail "$submodule is not a gitlink"
done
[[ -f README.md && ! -L README.md ]] || fail 'README.md is missing or unsafe'
grep -Fq 'independent integration project' README.md || fail 'independent-project notice is missing'
grep -Fq 'not endorsed by Physical Intelligence' README.md || fail 'non-endorsement notice is missing'
grep -Fq "$upstream_base" README.md || fail 'pinned upstream attribution is missing'
grep -Fq 'https://github.com/Physical-Intelligence/openpi' README.md || fail 'upstream source link is missing'

audit_tmp="$(mktemp -d "${TMPDIR:-/tmp}/pi-robotics-public-audit.XXXXXX")"
cleanup() {
    [[ ! -L "$audit_tmp" && -d "$audit_tmp" ]] && rm -rf -- "$audit_tmp"
}
trap cleanup EXIT HUP INT TERM

git -c core.quotePath=false log -m -p --format= --unified=0 --no-ext-diff --no-textconv \
    "$upstream_base..HEAD" >"$audit_tmp/history.patch"
git -c core.quotePath=false diff -p --unified=0 --no-ext-diff --no-textconv \
    "$upstream_base" -- >"$audit_tmp/current.patch"
git -c core.quotePath=false log -m --format= -z --name-only --diff-filter=ACMR \
    "$upstream_base..HEAD" >"$audit_tmp/history.paths"
git diff --name-only -z --diff-filter=ACMR "$upstream_base" -- >"$audit_tmp/current.paths"
git ls-files --others --exclude-standard -z >"$audit_tmp/untracked.paths"
git -c core.quotePath=false log -m --format= -z --numstat "$upstream_base..HEAD" >"$audit_tmp/history.numstat"
git diff --numstat -z "$upstream_base" -- >"$audit_tmp/current.numstat"
git log -m --format=%B "$upstream_base..HEAD" >"$audit_tmp/history.messages"

python3 - "$repo_root" "$audit_tmp" <<'PY'
from __future__ import annotations

from pathlib import Path
import re
import sys

repo = Path(sys.argv[1])
audit = Path(sys.argv[2])
findings: set[tuple[str, str]] = set()

forbidden_suffixes = {
    ".avi",
    ".bag",
    ".ckpt",
    ".jsonl",
    ".key",
    ".log",
    ".mcap",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
forbidden_components = {".runtime", "checkpoints", "outputs", "policy_records", "wandb"}
forbidden_names = {".env", "authorized_keys", "id_ed25519", "id_rsa", "known_hosts"}
content_patterns = {
    "private key material": re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Windows machine hostname": re.compile(r"\bDESKTOP-[A-Za-z0-9_-]{5,}\b", re.IGNORECASE),
    "macOS user home": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "Linux user home": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "Windows user home": re.compile(r"\b[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]", re.IGNORECASE),
    "RFC1918 address": re.compile(
        r"(?<![0-9])(?:10(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})(?![0-9])"
    ),
}
# These fake sanitizer inputs predate this audit in published project history.
# Current tests construct them dynamically; only their original path is exempt.
historical_test_fixtures = {
    ("history:tests/test_telemetry.py", "DESKTOP-" + "EXAMPLE"),
    ("history:tests/test_telemetry.py", "192" + ".168.1.2"),
}


def report(kind: str, location: str) -> None:
    findings.add((kind, location))


def scan_path(raw_path: str, scope: str) -> None:
    path = raw_path.strip()
    if not path:
        return
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        report("control character in filename", scope)
        return
    normalized = path.replace("\\", "/")
    scan_content(normalized, f"{scope}:filename:{path}")
    parts = Path(normalized).parts
    basename = parts[-1].casefold() if parts else ""
    suffix = Path(basename).suffix
    if any(part.casefold() in forbidden_components for part in parts):
        report("generated/private directory", f"{scope}:{path}")
    if basename in forbidden_names or basename.startswith("id_rsa.") or basename.startswith("id_ed25519."):
        report("SSH/environment identity filename", f"{scope}:{path}")
    if suffix in forbidden_suffixes:
        report("generated/private artifact filename", f"{scope}:{path}")


def scan_content(text: str, location: str) -> None:
    for fixture_location, fixture in historical_test_fixtures:
        if location == fixture_location:
            text = text.replace(fixture, "")
    for kind, pattern in content_patterns.items():
        if pattern.search(text):
            report(kind, location)


def patch_additions(path: Path, scope: str):
    current_path = "unknown"
    in_hunk = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
            marker = " b/"
            current_path = line.rsplit(marker, 1)[-1] if marker in line else "unknown"
        elif line.startswith("+++ ") and not in_hunk:
            value = line[4:]
            current_path = value[2:] if value.startswith("b/") else current_path
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            yield line[1:], f"{scope}:{current_path}"


for paths_file, scope, null_separated in (
    (audit / "history.paths", "history", True),
    (audit / "current.paths", "current", True),
    (audit / "untracked.paths", "untracked", True),
):
    data = paths_file.read_bytes()
    values = data.split(b"\0") if null_separated else data.splitlines()
    for value in values:
        if value:
            scan_path(value.decode("utf-8", errors="replace"), scope)

for patch, scope in ((audit / "history.patch", "history"), (audit / "current.patch", "current")):
    for added, location in patch_additions(patch, scope):
        scan_content(added, location)

scan_content((audit / "history.messages").read_text(encoding="utf-8", errors="replace"), "history:messages")

for numstat, scope in ((audit / "history.numstat", "history"), (audit / "current.numstat", "current")):
    for record in numstat.read_bytes().split(b"\0"):
        fields = record.decode("utf-8", errors="replace").split("\t", 2)
        if len(fields) == 3 and fields[:2] == ["-", "-"]:
            report("binary project-added content", f"{scope}:{fields[2]}")

for raw_path in (audit / "untracked.paths").read_bytes().split(b"\0"):
    if not raw_path:
        continue
    relative = raw_path.decode("utf-8", errors="strict")
    candidate = repo / relative
    if candidate.is_symlink():
        scan_content(candidate.readlink().as_posix(), f"untracked:{relative}")
    elif candidate.is_file():
        data = candidate.read_bytes()
        if len(data) > 1024 * 1024 or b"\0" in data:
            report("large or binary untracked content", f"untracked:{relative}")
        else:
            scan_content(data.decode("utf-8", errors="replace"), f"untracked:{relative}")

if findings:
    for kind, location in sorted(findings):
        print(f"forbidden {kind} in {location!r}", file=sys.stderr)
    raise SystemExit(1)
PY

{
    cat "$audit_tmp/history.messages" "$audit_tmp/history.patch" "$audit_tmp/current.patch"
    while IFS= read -r -d '' path; do
        if [[ -L "$path" ]]; then
            readlink -- "$path"
        elif [[ -f "$path" ]]; then
            cat -- "$path"
        fi
    done <"$audit_tmp/untracked.paths"
} >"$audit_tmp/project-content.bundle"

gitleaks git --no-banner --redact --timeout 180 --log-opts="$upstream_base..HEAD" . ||
    fail 'Gitleaks rejected project-added history'
gitleaks dir --no-banner --redact --timeout 180 "$audit_tmp/project-content.bundle" ||
    fail 'Gitleaks rejected current project content'

echo "Public repository audit passed: project history and current publishable files are clean from $upstream_base."
