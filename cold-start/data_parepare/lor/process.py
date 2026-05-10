# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import os
from tqdm import tqdm
from PIL import Image

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"

anno = json.load(open('spar_select_cold_cot.json', 'r'))
# anno = json.load(open('spar_multiview_select_cold_cot.json', 'r'))

for item in anno:
    ques = item['conversations'][0]['value']
    ans = item['conversations'][1]['value']
    cot = item['cot']
    ques = QUESTION_TEMPLATE_IMAGE.format(Question=ques)
    ans = f"<think>{cot}</think>\n<answer>{ans}</answer>"
    item['conversations'][0]['value'] = ques
    item['conversations'][1]['value'] = ans


with open('spar_cot_coldstart.json', 'w') as f:
    json.dump(anno, f, indent=4)
