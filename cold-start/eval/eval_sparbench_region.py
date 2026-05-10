# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import itertools
import json
import os
from collections import defaultdict

import PIL
import torch
import re
import numpy as np
from tqdm import tqdm

import llava
from llava import conversation as clib
from llava.media import Image, Mask
from llava.utils import distributed as dist
from transformers import AutoConfig, AutoModel
from datasets import load_dataset, Features, Image, Value

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"
QUESTION_TEMPLATE_IMAGE_DET = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first detect 3D centers of relevant objects, then think deeply about the question based on the given image, and finally provide the answer. The detection, reasoning and answer are enclosed within <detect> </detect>, <think> </think> and <answer> </answer> tags, respectively, i.e., <detect> detection here </detect>, <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"

def extract_ans(response):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, response, re.DOTALL) 
    if len(matches) > 0:
        return matches[0]
    else:
        return response
        

def eval_model(args):
        
    # Device setup
    dist.init()
    world_size, global_rank = dist.size(), dist.rank()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # Load SVILA model & conversation mode
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
    model.to(dtype=eval(args.dtype))
    model.config.model_dtype = eval(args.dtype)

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)
    print('generation_config: ', generation_config)

    # Read data list
    dataset = load_dataset(args.question_file)
    print(len(dataset['test']))
    print(dataset['test'].features.keys())
    data_list = []
    for i,example in tqdm(enumerate(dataset['test'])):
        data_list.append(example)
    print(len(data_list))
    data_list = data_list[global_rank::world_size]
    
    # Read region prompts
    region_prompts = json.load(open('path/to/sparbench_region_prompts.json'))
    region_prompts = {item['id']: item for item in region_prompts}

    answer_all = []
    for instance in tqdm(data_list, total=len(data_list), disable=global_rank != 0):
        img_num = len(instance['image'])
        contents = instance['image']
        id = instance['id']

        ques = region_prompts[id]["question_region"]
        region_flag = len(region_prompts[id]['visual_prompt']) > 0
        if args.reason:
            if region_flag:
                ques = QUESTION_TEMPLATE_IMAGE_DET.format(Question=ques)
            else:
                ques = QUESTION_TEMPLATE_IMAGE.format(Question=ques)
        print(ques)
        contents.append(ques)
        bbox = region_prompts[id]['bbox']
        # combine h,w to x1,y1,x2,y2 for dynamic_mask_process, which needs img_hw. pass img_hw or combine h,w with bbox
        h, w = instance['image'][0].height, instance['image'][0].width
        print(bbox, h, w)
        if img_num == 1:
            bbox = [box + [h, w] for box in bbox]
            if len(bbox) > 0:
                mask = Mask("bbox", bbox)
                contents.append(mask)
            output = model.generate_content(contents, generation_config=generation_config)
        else:
            for k,v in bbox.items():
                bbox_img = [box + [h, w] for box in v]
                mask = Mask("bbox", bbox_img)
                contents.append(mask)
            output = model.generate_content(contents, generation_config=generation_config, mask_idx=[region_prompts[id]['mask_idx']])

        print(output)
        response = extract_ans(output)
        
        answer_all.append({'question': instance['question'], 'answer': instance['answer'], 'task': instance['task'], 'response': response, 'output': output, 'format_type': instance['format_type'], 'img_type': instance['img_type']})
        print('='*20)

    if dist.size() > 1:
        answer_all_gather = dist.all_gather(answer_all)
        answer_all = []
        for answers in answer_all_gather:
            answer_all += answers

    if dist.is_main():
        print(len(answer_all))

        # save instance to json
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/output.json", "w") as f:
            json.dump(answer_all, f, indent=2)
        print(f"Saved output.json to {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="path/to/svila-verl/results/svila_spar_cold_filter_hard/global_step_115/actor/SVila",
    )
    parser.add_argument(
        "--reason",
        action="store_true", 
        default=False     
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file",
        type=str,
        default="path/to/data/SPAR_Bench",
    )
    parser.add_argument(
        "--format-type", type=str, default="select"
    )
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./svila-spar-cold-verl-hard-spar",
    )
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dtype", type=str, default="torch.bfloat16")
    args = parser.parse_args()

    eval_model(args)
    
