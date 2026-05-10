import os

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .base_pe import PE, PEConfig


def build_perception_encoder(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_name_or_path is None:
        return None

    if config.resume_path and os.path.exists(model_name_or_path):
        print(f"Resume PE from path {model_name_or_path}.")
        return PE.from_pretrained(model_name_or_path, config, torch_dtype=eval(config.model_dtype))

    ## build from scratch
    else:
        print("WARNING: Building PE from scratch!")
        pe_cfg = PEConfig(model_name_or_path)
        pe = PE(pe_cfg, config).to(eval(config.model_dtype))
        return pe
