import subprocess
import sys


def test_phase2_pure_modules_do_not_import_model_or_gpu_stacks() -> None:
    script = """
import sys
import tools.remote_aloha.config
import tools.remote_aloha.policy_contract
import tools.remote_aloha.policy_smoke
import tools.remote_aloha.process_record
import tools.remote_aloha.remote

blocked = {name.split('.', 1)[0] for name in sys.modules} & {'jax', 'jaxlib', 'torch', 'flax', 'openpi'}
assert not blocked, blocked
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)
