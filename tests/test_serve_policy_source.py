import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace


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
    assert "require_torch_device" in source
    assert 'policy_backend: Literal["jax", "pytorch"]' in source
    assert "pytorch_compile_mode=None" in source
    assert "compact_masked_images" in source
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
    assert "--require-torch-device=3090" in source
    assert '"--policy-backend=$backend"' in source
    assert "jax_platform=cpu" in source
    assert "${checkpoint_label}_pytorch" in source
    assert '-f "$checkpoint/model.safetensors" || -f "$checkpoint/model.safetensors.index.json"' in source
    assert "--compact-masked-images" in source
    assert '"XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_mem_fraction"' in source
    assert "0.75|0.80|0.85|0.90|0.95" in source
    assert "process_record launch" in source
    assert 'kill -TERM "$pid"' not in source
    assert "[server] loading profile=$profile backend=$backend" in source
    assert "[server] still loading; elapsed=${SECONDS}s" in source

    smoke = Path("scripts/smoke_policy.sh").read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=10s" in smoke
    assert "--query-gpu=timestamp,name,memory.used,utilization.gpu" in smoke
    assert "process_record verify" in smoke
    assert "--query-compute-apps=pid,used_gpu_memory" in smoke
    assert "host_peak_rss" in smoke
    assert '[[ "$backend" == jax ]]' in smoke
    assert "torch_model_device" in Path("scripts/serve_policy.py").read_text(encoding="utf-8")

    client = Path("tools/remote_aloha/policy_smoke.py").read_text(encoding="utf-8")
    assert '"images": {name: image for name in POLICY_CAMERA_VIEWS}' in client
    assert 'observation["prompt"] = profile.default_prompt' in client
    assert "policy.close()" in client

    policy = Path("src/openpi/policies/policy.py").read_text(encoding="utf-8")
    assert "if self._compact_masked_images:" in policy
    assert '"__openpi_benchmark_noise_seed"' in policy
    assert "torch.Generator(device=self._pytorch_device).manual_seed" in policy
    assert "generator=generator" in policy
    model = Path("src/openpi/models/pi0.py").read_text(encoding="utf-8")
    assert "image_keys=tuple(observation.images)" in model


def test_runtime_evidence_paths_are_ignored() -> None:
    for path in (".runtime/server.json", ".runtime/secret-scan.sha", "policy_records/example"):
        subprocess.run(["git", "check-ignore", "--quiet", path], check=True)


def test_secret_scan_receipt_is_symlink_safe_and_atomic() -> None:
    source = Path("scripts/secret_scan.sh").read_text(encoding="utf-8")
    assert "[[ ! -L .runtime" in source
    assert "mktemp .runtime/.secret-scan.sha" in source
    assert 'mv -f -- "$receipt_tmp" .runtime/secret-scan.sha' in source


def test_partial_bfloat16_converter_is_auto_selected_bounded_and_sharded() -> None:
    converter = Path("examples/convert_jax_model_to_pytorch.py").read_text(encoding="utf-8")
    assert 'restore_mode: Literal["full-float32", "partial-bfloat16"] = "full-float32"' in converter
    assert "ocp.PLACEHOLDER" in converter
    assert "torch.from_dlpack" in converter
    assert 'max_shard_size="1GB"' in converter
    assert "shared_tensors_to_discard=[_BASE_LM_HEAD]" in converter
    assert "Expected {expected_source_count} source leaves" in converter
    assert "Expected {expected_target_count} mapped targets" in converter
    assert converter.count("gemma_expert.lm_head = None") == 2
    assert "del mappings, source_tensor, restored" in converter

    wrapper = Path("scripts/convert_policy_checkpoint.sh").read_text(encoding="utf-8")
    assert "umask 077" in wrapper
    assert "timeout --signal=TERM --kill-after=30s" in wrapper
    assert "60 * 1024 * 1024" in wrapper
    assert "realpath -e" in wrapper
    assert "conversion.lock" in wrapper
    assert "JAX_PLATFORMS=cpu" in wrapper
    assert 'awk \'$1 == "MemAvailable:"' in wrapper
    assert "16 * 1024 * 1024" in wrapper
    assert "auto|full-float32|partial-bfloat16" in wrapper
    assert '--restore-mode "$restore_mode"' in wrapper
    assert "checkpoint.partial" in wrapper
    assert 'mv -- "$temporary_checkpoint" "$final_checkpoint"' in wrapper
    assert wrapper.index('mv -- "$temporary_checkpoint" "$final_checkpoint"') < wrapper.index("published=yes")
    assert '"$temporary_checkpoint/model.safetensors"' in wrapper
    assert "__ALOHA_REMOTE_EVIDENCE__" in wrapper
    assert "pi0_aloha_sim)" in wrapper
    assert "pi05_aloha_base)" in wrapper
    assert "eval" not in wrapper
    assert 'assets_source = pathlib.Path(checkpoint_dir) / "assets"' in converter
    assert '"restore_mode": "full-float32"' in converter

    policy_loader = Path("src/openpi/policies/policy_config.py").read_text(encoding="utf-8")
    model_loader = Path("src/openpi/models/model.py").read_text(encoding="utf-8")
    assert 'sharded_weight_index = os.path.join(checkpoint_dir, "model.safetensors.index.json")' in policy_loader
    assert "load_torch_model(model, weight_path, strict=False, safe=True)" in model_loader
    assert "with torch.device(device):" in model_loader
    assert "gemma_expert.lm_head = None" in model_loader
    assert "load_pytorch(train_config, weight_path, pytorch_device)" in policy_loader


def test_partial_bfloat16_mapping_contract_without_checkpoint() -> None:
    source = Path("examples/convert_jax_model_to_pytorch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    mapping_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_mapped_tensors"
    )

    class Symbol:
        def __init__(self, expression: str):
            self.expression = expression

        def __getitem__(self, index):
            return Symbol(f"{self.expression}[{index!r}]")

        @property
        def T(self):  # noqa: N802 - mirror torch.Tensor.T
            return Symbol(f"{self.expression}.T")

        def permute(self, *dimensions):
            return Symbol(f"{self.expression}.permute{dimensions}")

        def reshape(self, *dimensions):
            return Symbol(f"{self.expression}.reshape{dimensions}")

    expert_config = SimpleNamespace(depth=18, num_heads=8, head_dim=256, width=1024)
    namespace = {
        "torch": SimpleNamespace(Tensor=Symbol),
        "openpi": SimpleNamespace(
            models=SimpleNamespace(
                gemma=SimpleNamespace(get_config=lambda _: expert_config),
                pi0_config=SimpleNamespace(Pi0Config=object),
            )
        ),
    }
    ast.fix_missing_locations(mapping_function)
    exec(compile(ast.Module(body=[mapping_function], type_ignores=[]), "<mapping-contract>", "exec"), namespace)
    mapped_tensors = namespace["_mapped_tensors"]

    common_keys = [
        "img/embedding/kernel",
        "img/embedding/bias",
        "img/pos_embedding",
        "img/Transformer/encoder_norm/scale",
        "img/Transformer/encoder_norm/bias",
        "img/head/kernel",
        "img/head/bias",
        "llm/embedder/input_embedding",
        "llm/final_norm/scale",
        "img/Transformer/encoderblock/LayerNorm_0/scale",
        "img/Transformer/encoderblock/LayerNorm_0/bias",
        "img/Transformer/encoderblock/LayerNorm_1/scale",
        "img/Transformer/encoderblock/LayerNorm_1/bias",
        "img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel",
        "img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias",
        "img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel",
        "img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias",
        *[
            f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/{component}/{parameter}"
            for component in ("key", "value", "query", "out")
            for parameter in ("kernel", "bias")
        ],
        "llm/layers/attn/q_einsum/w",
        "llm/layers/attn/kv_einsum/w",
        "llm/layers/attn/attn_vec_einsum/w",
        "llm/layers/mlp/gating_einsum",
        "llm/layers/mlp/linear",
        "llm/layers/pre_attention_norm/scale",
        "llm/layers/pre_ffw_norm/scale",
        "llm/layers/attn/q_einsum_1/w",
        "llm/layers/attn/kv_einsum_1/w",
        "llm/layers/attn/attn_vec_einsum_1/w",
        "llm/layers/mlp_1/gating_einsum",
        "llm/layers/mlp_1/linear",
    ]
    variant_keys = {
        False: [
            "llm/layers/pre_attention_norm_1/scale",
            "llm/layers/pre_ffw_norm_1/scale",
            "llm/final_norm_1/scale",
        ],
        True: [
            "llm/layers/pre_attention_norm_1/Dense_0/bias",
            "llm/layers/pre_attention_norm_1/Dense_0/kernel",
            "llm/layers/pre_ffw_norm_1/Dense_0/bias",
            "llm/layers/pre_ffw_norm_1/Dense_0/kernel",
            "llm/final_norm_1/Dense_0/bias",
            "llm/final_norm_1/Dense_0/kernel",
        ],
    }
    projection_keys = {
        False: ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"],
        True: ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"],
    }

    def map_leaf(source_key: str, *, pi05: bool):
        return list(mapped_tensors(source_key, Symbol("x"), SimpleNamespace(pi05=pi05)))

    for pi05, expected_sources, expected_targets in ((False, 50, 776), (True, 51, 811)):
        sources = [f"PaliGemma/{key}" for key in common_keys + variant_keys[pi05]]
        sources += [f"{key}/{parameter}" for key in projection_keys[pi05] for parameter in ("kernel", "bias")]
        targets = [target for source_key in sources for target, _ in map_leaf(source_key, pi05=pi05)]
        assert len(sources) == expected_sources
        assert len(targets) == expected_targets
        assert len(set(targets)) == expected_targets

    patch = map_leaf("PaliGemma/img/embedding/kernel", pi05=False)[0]
    assert patch[0].endswith("embeddings.patch_embedding.weight")
    assert patch[1].expression == "x.permute(3, 2, 0, 1)"
    base_query = map_leaf("PaliGemma/llm/layers/attn/q_einsum/w", pi05=False)[0]
    assert base_query[0].endswith("layers.0.self_attn.q_proj.weight")
    assert base_query[1].expression == "x[0].permute(0, 2, 1).reshape(2048, 2048)"
    expert_output = map_leaf("PaliGemma/llm/layers/attn/attn_vec_einsum_1/w", pi05=False)[0]
    assert expert_output[0].endswith("layers.0.self_attn.o_proj.weight")
    assert expert_output[1].expression == "x[0].reshape(2048, 1024).T"
    adaptive_norm = map_leaf("PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel", pi05=True)[0]
    assert adaptive_norm[0].endswith("layers.0.input_layernorm.dense.weight")
    assert adaptive_norm[1].expression == "x[0].T"
