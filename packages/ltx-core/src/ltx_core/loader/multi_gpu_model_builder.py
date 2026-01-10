import logging
from dataclasses import dataclass, field, replace
from typing import Generic

import torch
from accelerate import dispatch_model, infer_auto_device_map

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models sharded across multiple GPUs via device maps.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "MultiGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def model_config(self) -> dict:
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict

    def build(
        self,
        device_map: dict[str, int | str | torch.device] | str | None = None,
        dtype: torch.dtype | None = None,
        max_memory: dict[int | str, int | str] | None = None,
        offload_folder: str | None = None,
        no_split_module_classes: list[str] | None = None,
        device: torch.device | None = None,
    ) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]
        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device)

        lora_strengths = [lora.strength for lora in self.loras]
        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            sd = model_state_dict.sd
            if dtype is not None:
                sd = {key: value.to(dtype=dtype) for key, value in sd.items()}
            meta_model.load_state_dict(sd, strict=False, assign=True)
        else:
            lora_state_dicts = [
                self.load_sd([lora.path], sd_ops=lora.sd_ops, registry=self.registry, device=device)
                for lora in self.loras
            ]
            lora_sd_and_strengths = [
                LoraStateDictWithStrength(sd, strength)
                for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
            ]
            final_sd = apply_loras(
                model_sd=model_state_dict,
                lora_sd_and_strengths=lora_sd_and_strengths,
                dtype=dtype,
                destination_sd=model_state_dict if isinstance(self.registry, DummyRegistry) else None,
            )
            meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)

        if device_map is None:
            return meta_model.to(device)

        if device_map == "auto":
            device_map = infer_auto_device_map(
                meta_model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )

        return dispatch_model(meta_model, device_map=device_map, offload_folder=offload_folder)
