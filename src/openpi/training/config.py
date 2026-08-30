"""Inference policy configurations used by the policy server and checkpoint converter."""

import abc
import dataclasses
import difflib
import pathlib
from typing import Any, Protocol, TypeAlias

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Select the normalization-statistics directory inside a checkpoint."""

    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    asset_id: str | None = None
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    prompt_from_task: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a transform group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = model_config.fast_model_tokenizer or _tokenizer.FASTTokenizer
                tokenizer_kwargs = model_config.fast_model_tokenizer_kwargs or {}
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs)),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create inference transforms for a model."""

    def create_base_config(self, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(
            asset_id=self.assets.asset_id,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    data_transforms: GroupFactory = dataclasses.field(default_factory=lambda: _transforms.Group)
    model_transforms: GroupFactory = dataclasses.field(default_factory=ModelTransformFactory)
    base_config: DataConfig | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        base = self.base_config or self.create_base_config(model_config)
        return dataclasses.replace(
            base,
            asset_id=self.assets.asset_id or base.asset_id,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        return dataclasses.replace(
            self.create_base_config(model_config),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        return dataclasses.replace(
            self.create_base_config(model_config),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """The retained name matches the upstream checkpoint metadata API."""

    name: str
    model: _model.BaseModelConfig
    data: DataConfigFactory
    assets_base_dir: str = "./assets"
    policy_metadata: dict[str, Any] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()


def _droid_data(model_type: ModelType) -> SimpleDataConfig:
    return SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_type)],
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(prompt_from_task=True),
    )


_CONFIGS = [
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(assets=AssetsConfig(asset_id="trossen")),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(assets=AssetsConfig(asset_id="trossen")),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=_droid_data(ModelType.PI0),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=_droid_data(ModelType.PI0_FAST),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=_droid_data(ModelType.PI05),
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(assets=AssetsConfig(asset_id="physical-intelligence/libero")),
    ),
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="lerobot/aloha_sim_transfer_cube_human"),
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
    ),
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> TrainConfig:
    """Get an inference configuration by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
