import os

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .base_extractor import RegionExtractor, RegionExtractorConfig


def build_region_extractor(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path and os.path.exists(model_type_or_path):
        assert os.path.exists(model_type_or_path), f"Resume region extractor path {model_type_or_path} does not exist!"
        return RegionExtractor.from_pretrained(model_type_or_path, config, torch_dtype=eval(config.model_dtype))
    ## build from scratch
    else:
        print("WARNING: Building region extractor from scratch!")
        region_extractor_cfg = RegionExtractorConfig(model_type_or_path)
        region_extractor = RegionExtractor(region_extractor_cfg, config).to(eval(config.model_dtype))
        return region_extractor
