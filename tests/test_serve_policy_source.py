import ast
from pathlib import Path
import subprocess


def test_policy_server_host_and_gpu_metadata_patch_is_localized() -> None:
    path = Path("scripts/serve_policy.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    args_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    host = next(
        node
        for node in args_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "host"
    )
    assert isinstance(host.value, ast.Constant)
    assert host.value.value == "0.0.0.0"
    server_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "WebsocketPolicyServer"
    ]
    assert len(server_calls) == 1
    host_keyword = next(keyword for keyword in server_calls[0].keywords if keyword.arg == "host")
    assert ast.unparse(host_keyword.value) == "args.host"
    assert "socket.gethostname" not in source
    assert "socket.gethostbyname" not in source
    assert "require_jax_platform" in source
    assert "require_jax_device" in source
    assert '"action_dimension": 14' in source


def test_wsl_start_wrapper_has_exact_two_profile_routes_and_loopback() -> None:
    source = Path("scripts/start_policy_server.sh").read_text(encoding="utf-8")
    assert "pi0_aloha_sim)" in source
    assert "environment=ALOHA_SIM" in source
    assert "pi05_aloha_base)" in source
    assert "environment=ALOHA" in source
    assert 'prompt=(--default-prompt="Transfer cube")' in source
    assert '[[ "$host" == 127.0.0.1 ]]' in source
    assert "eval" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "--require-jax-platform=gpu" in source
    assert "--require-jax-device=3090" in source
    assert '"XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_mem_fraction"' in source
    assert "0.75|0.80|0.85|0.90|0.95" in source
    assert "process_record launch" in source
    assert 'kill -TERM "$pid"' not in source

    smoke = Path("scripts/smoke_policy.sh").read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=10s" in smoke
    assert "--query-gpu=timestamp,name,memory.used,utilization.gpu" in smoke
    assert "process_record verify" in smoke
    assert "--query-compute-apps=pid,used_gpu_memory" in smoke

    client = Path("tools/remote_aloha/policy_smoke.py").read_text(encoding="utf-8")
    assert '"images": {"cam_high": image}' in client
    assert '"prompt"' not in client


def test_runtime_evidence_paths_are_ignored() -> None:
    for path in (".runtime/server.json", ".runtime/secret-scan.sha", "policy_records/example"):
        subprocess.run(["git", "check-ignore", "--quiet", path], check=True)


def test_secret_scan_receipt_is_symlink_safe_and_atomic() -> None:
    source = Path("scripts/secret_scan.sh").read_text(encoding="utf-8")
    assert "[[ ! -L .runtime" in source
    assert "mktemp .runtime/.secret-scan.sha" in source
    assert 'mv -f -- "$receipt_tmp" .runtime/secret-scan.sha' in source
