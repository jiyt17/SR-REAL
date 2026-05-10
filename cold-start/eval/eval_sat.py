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
import io
import random
from tqdm import tqdm

import llava
from llava import conversation as clib
from llava.utils import distributed as dist
from transformers import AutoConfig, AutoModel
from datasets import load_dataset, Features, Image, Value

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"

def extract_ans(response):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, response, re.DOTALL) 
    if len(matches) > 0:
        return matches[0]
    else:
        return response
        

def eval_model(args):
    random.seed(args.seed)

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
    dataset = load_dataset("parquet", data_files=args.question_file, split="train")
    print(len(dataset))
    print(dataset.features.keys())
    data_list = []
    for i,example in tqdm(enumerate(dataset)):
        data_list.append(example)
    print(len(data_list))
    data_list = data_list[global_rank::world_size]

    print(dist.size(), len(data_list))
    answer_all = []
    for instance in tqdm(data_list, total=len(data_list), disable=global_rank != 0):
        print('*'*10, args.reason)
        contents = []
        for img in instance['image_bytes']:
            image_stream = io.BytesIO(img) 
            image = PIL.Image.open(image_stream)
            contents.append(image)
        ques = instance["question"]
        ques = ques + "\nSelect the correct response from the given choices."
        right_ans = "A"
        answers = instance['answers']
        random.shuffle(answers)
        ans_len = len(answers)
        for i,ind in enumerate(['A', 'B', 'C', 'D'][:ans_len]):
            ques = ques + f"\n{ind}. {answers[i]}"
            if answers[i] == instance['correct_answer']:
                right_ans = ind
        if args.reason:
            ques = QUESTION_TEMPLATE_IMAGE.format(Question=ques)
        print(ques)
        contents.append(ques)
        output = model.generate_content(contents, generation_config=generation_config)
        print(output)
        response = extract_ans(output)
        
        answer_all.append({'question': instance['question'], 'answers': answers, 'answer':right_ans, 'response': response, 'output': output, 'question_type': instance['question_type']})

    if dist.size() > 1:
        answer_all_gather = dist.all_gather(answer_all)
        answer_all = []
        for answers in answer_all_gather:
            answer_all += answers

    if dist.is_main():
        print(len(answer_all))

        correct_all = []
        for instance in answer_all:
            # remove all parentheses and spaces and . and lower
            answer = (
                instance["answer"]
                .strip()
                .lower()
                .replace("(", "")
                .replace(")", "")
                .replace(" ", "")
                .replace(".", "")
            )
            response = (
                instance["response"]
                .strip()
                .lower()
                .replace("(", "")
                .replace(")", "")
                .replace(" ", "")
                .replace(".", "")
            )
            correct = answer == response
            instance["correct"] = correct
            correct_all.append(correct)

        # Step 1: Gather correct counts and totals
        counts_by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
        for instance in answer_all:
            question_type = instance["question_type"]
            counts_by_question_type[question_type]["correct"] += int(instance["correct"])
            counts_by_question_type[question_type]["total"] += 1

        # Step 2: Compute accuracy
        accuracy_by_question_type = {k: v["correct"] / v["total"] for k, v in counts_by_question_type.items()}
        accuracy_by_question_type["mean"] = sum(correct_all) / len(correct_all)

        # Step 3: Pretty print
        print("Accuracy by question type:")
        for question_type in counts_by_question_type:
            correct = counts_by_question_type[question_type]["correct"]
            total = counts_by_question_type[question_type]["total"]
            accuracy = correct / total
            print(f"{question_type}: {accuracy * 100:.2f}% ({correct} / {total})")

        # Print mean accuracy
        mean_accuracy = accuracy_by_question_type["mean"]
        print(f"Mean Accuracy: {mean_accuracy * 100:.2f}% ({sum(correct_all)} / {len(correct_all)})")

        # save instance to json
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/output.json", "w") as f:
            json.dump(answer_all, f, indent=2)
        print(f"Saved output.json to {output_dir}")

        # save accuracy to json
        with open(f"{output_dir}/accuracy.json", "w") as f:
            json.dump(accuracy_by_question_type, f, indent=2)
        print(f"Saved accuracy.json to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        # default="path/to/svila-verl/results/svila_spar_cold_v2/global_step_178/actor/SVila",
        default="path/to/pretrained/svila-8b-rvila-compatible",
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
        default="path/to/data/SAT/SAT_test.parquet",
    )
    parser.add_argument(
        "--image-dir", type=str, default=""
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./svila-rwqa",
    )
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dtype", type=str, default="torch.bfloat16")
    args = parser.parse_args()

    eval_model(args)
