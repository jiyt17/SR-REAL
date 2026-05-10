# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import base64
import io
import itertools
import json
import os
import random
import re
from collections import defaultdict

import torch
from PIL import Image
from termcolor import colored
from tqdm import tqdm
from PIL import Image, ImageDraw
from llava.media import Mask

import llava
from llava import conversation as clib
from llava.data.builder import DATASETS
from llava.utils import distributed as dist

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"
QUESTION_TEMPLATE_IMAGE_DET = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first detect 3D centers of relevant objects, then think deeply about the question based on the given image, and finally provide the answer. The detection, reasoning and answer are enclosed within <detect> </detect>, <think> </think> and <answer> </answer> tags, respectively, i.e., <detect> detection here </detect>, <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"


def extract_ans(response):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, response, re.DOTALL) 
    if len(matches) > 0:
        return matches[0]
    else:
        return response

def format_question_with_choices(question, options, correct_answer):
    # Label each option with (A), (B), ...
    labeled_options = [f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)]  # 65 = 'A'

    # Append labeled options to the question
    formatted_question = f"{question} " + "Select from the following choices.\n" + "\n".join(labeled_options)
    # formatted_question += "\nPlease answer with: " + ", ".join([f"({chr(65 + i)})" for i in range(len(options))]) + "."

    # Transform correct_answer from number to letter (mapping: 0 -> A, 1 -> B, ...)
    correct_label = chr(65 + int(correct_answer))
    return formatted_question, correct_label


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

    # Read data list
    data_list = []
    data_list = json.load(open(args.question_file))
    data_list = data_list[global_rank::world_size]

    answer_all = []
    ind = -1
    for instance in tqdm(data_list, total=len(data_list), disable=global_rank != 0):
        ind += 1
        contents = []

        # prepare image
        image_data = base64.b64decode(instance["image"])
        pil_image = Image.open(io.BytesIO(image_data))
        w, h = pil_image.size
        contents.append(pil_image)

        objects_names = []
        objects_bboxes = []
        error_flag = 0
        for obj in instance["objects"]:
            if obj["bbox"][0] + obj["bbox"][2] > w:
                error_flag = 1
                print('box annotation error')
                break
            objects_names.append(obj["name"])
            objects_bboxes.append(obj["bbox"])  # x,y,w,h
        if error_flag == 0:
            objects_names = [name + ' <mask>' for name in objects_names]
            objects_names = "Relevant objects: " + ", ".join(objects_names) + ".\n"

        # format into multi-choice question
        q_formatted, correct_letter = format_question_with_choices(
            instance["question"], instance["answer_options"], instance["answer"]
        )
        if error_flag == 0:
            q_formatted = q_formatted + "\n" + objects_names
            if args.reason:
                q_formatted = QUESTION_TEMPLATE_IMAGE_DET.format(Question=q_formatted)
        else:
            if args.reason:
                q_formatted = QUESTION_TEMPLATE_IMAGE.format(Question=q_formatted)
        contents.append(q_formatted)
                
        if error_flag == 0:
            bbox = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] + [h, w] for box in objects_bboxes]
            # print(bbox)
            mask = Mask("bbox", bbox)
            contents.append(mask)

        output = model.generate_content(contents, generation_config=generation_config)
        print(output)
        response = extract_ans(output)
        instance["response"] = response
        answer_all.append(
            {
                "relation": instance["relation"],
                "question": q_formatted,
                "answer": correct_letter,
                "output": output,
                "response": response,
            }
        )

        print(colored(q_formatted, "red", attrs=["bold"]))
        print(colored(response, "green", attrs=["bold"]))
        print(colored(correct_letter, "blue", attrs=["bold"]))
        print(colored("-" * 100, "yellow", attrs=["bold"]))

    if dist.size() > 1:
        answer_all = dist.all_gather(answer_all)
        if dist.is_main():
            answer_all = list(itertools.chain(*answer_all))
            print(len(answer_all))

            correct_all = []
            for instance in answer_all:
                # Extract first choice letter (A/B/C/D) using regex
                answer_match = re.search(r"\b([A-D])\b", instance["answer"].upper())
                response_match = re.search(r"\b([A-D])\b", instance["response"].upper())

                # Default to incorrect if cannot parse either
                correct = False
                if answer_match and response_match:
                    correct = answer_match.group(1) == response_match.group(1)

                instance["correct"] = correct
                correct_all.append(correct)

            # Step 1: Gather correct counts and totals by question_type
            counts_by_relation = defaultdict(lambda: {"correct": 0, "total": 0})
            for instance in answer_all:
                relation = instance["relation"]
                counts_by_relation[relation]["correct"] += int(instance["correct"])
                counts_by_relation[relation]["total"] += 1

            # Step 2: Compute accuracy
            accuracy_by_relation = {k: v["correct"] / v["total"] for k, v in counts_by_relation.items()}

            # Step 3: Pretty print
            print("Accuracy by relation:")
            for relation in accuracy_by_relation:
                print(f"{relation}: {accuracy_by_relation[relation] * 100:.2f}%")

            # Print mean accuracy
            mean_accuracy = sum(correct_all) / len(correct_all)
            print(f"Mean Accuracy: {mean_accuracy * 100:.2f}% ({sum(correct_all)} / {len(correct_all)})")
            accuracy_by_relation["average"] = sum(correct_all) / len(correct_all)

            # save instance to json
            output_dir = os.path.expanduser(args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/embspatial_output.json", "w") as f:
                json.dump(answer_all, f, indent=2)
            print(f"Saved embspatial_output.json to {output_dir}")

            # save accuracy to json
            with open(f"{output_dir}/metrics.json", "w") as f:
                json.dump(accuracy_by_relation, f, indent=2)
            print(f"Saved metrics.json to {output_dir}")

        # dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="path/to/svila-verl/results/svila_spar_CA_NS_cold_blend_region_v11_mix_filter/190_actor_for_test/SVila-test",
        type=str,
    )
    parser.add_argument(
        "--reason",
        action="store_true", 
        default=False     
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument(
        "--output-dir",
        type=str,
    )
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dtype", type=str, default="torch.bfloat16")
    args = parser.parse_args()

    args.question_file = "path/to/data/EmbSpatial/embspatial_bench.json"

    eval_model(args)