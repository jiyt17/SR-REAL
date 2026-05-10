# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import shutil
import json
import os

source = "/path/to/cold-start-model"
target = source + '-hf'
os.makedirs(target, exist_ok=True)

for f in ['/llm', '/mm_projector', '/perception_encoder', '/region_extractor', '/vision_tower']:
    shutil.copytree(source + f, target + f)
for f in os.listdir('SR-REAL/grpo/remote_code'):
    shutil.copy(os.path.join('SR-REAL/grpo/remote_code', f), os.path.join(target, f))
cfg_llm = json.load(open(os.path.join(target, 'llm', 'config.json'), 'r'))
cfg_llm['model_max_length'] = None
json.dump(cfg_llm, open(os.path.join(target, 'llm', 'config.json'), 'w'), indent=4)