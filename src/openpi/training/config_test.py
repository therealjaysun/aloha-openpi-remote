from openpi.models import tokenizer
from openpi.training import config


def test_pi05_aloha_inference_config(monkeypatch) -> None:
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda max_len: ("tokenizer", max_len))

    policy_config = config.get_config("pi05_aloha")
    data_config = policy_config.data.create(policy_config.assets_dirs, policy_config.model)

    assert policy_config.model.pi05
    assert data_config.asset_id == "trossen"
    assert data_config.use_quantile_norm
    assert [type(transform).__name__ for transform in data_config.data_transforms.inputs] == [
        "AlohaInputs",
        "DeltaActions",
    ]
    assert [type(transform).__name__ for transform in data_config.data_transforms.outputs] == [
        "AbsoluteActions",
        "AlohaOutputs",
    ]
    assert len(data_config.model_transforms.inputs) == 4
