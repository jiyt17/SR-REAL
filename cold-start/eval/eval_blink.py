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
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm

import llava
from llava import conversation as clib
from llava.utils import distributed as dist
import re

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"
QUESTION_TEMPLATE_IMAGE_DET = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first detect 3D centers of relevant objects, then think deeply about the question based on the given image, and finally provide the answer. The detection, reasoning and answer are enclosed within <detect> </detect>, <think> </think> and <answer> </answer> tags, respectively, i.e., <detect> detection here </detect>, <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"

def extract_ans(response):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, response, re.DOTALL) 
    if len(matches) > 0:
        return matches[0]
    else:
        return response

def analyze_answer(d, gpt_answer):
    try:
        all_choices = d["choices"]
        intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
        intersect_last = list(set(all_choices).intersection(set(gpt_answer.split("\n\n")[-1].split())))
        if gpt_answer in ["A", "B", "C", "D", "E"]:
            prediction = "(" + gpt_answer + ")"
        elif gpt_answer in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
            prediction = gpt_answer
        else:
            if len(intersect_last) == 1:
                intersect = intersect_last
                gpt_answer = gpt_answer.split("\n\n")[-1]
            prediction = intersect[0]
        return prediction
    except Exception as e:
        pass


def get_prediction_file(split, model_name, args):
    save_path = f"{split}_predictions/{model_name}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    saved = {}
    for task_name in args.subtasks:
        output_path = f'{args.output_save_folder}/{model_name}/{task_name.replace("_", " ")}.json'
        outputs = json.load(open(output_path))[split]
        for d in outputs:
            saved[d["idx"]] = d["prediction"]
    json.dump(saved, open(save_path, "w"), indent=4)
    return save_path


def eval_prediction(val_results, args):
    accu_by_task = {}
    task_numbers = {}
    errors = {}
    for task_name in args.subtasks:
        accu_by_task[task_name] = 0
        task_numbers[task_name] = 0
        errors[task_name] = []
    answers = json.load(open(args.answer_file))
    for idx, gold_answer in answers.items():
        task = "_".join(idx.split("val")[1][1:].split("_")[:-1])
        if task not in task_numbers:
            continue
        task_numbers[task] += 1
        if idx in val_results and val_results[idx] == gold_answer:
            accu_by_task[task] += 1
        else:
            errors[task].append(idx)

    average_accu = 0
    for task in args.subtasks:
        accu_by_task[task] = accu_by_task[task] / task_numbers[task]
        average_accu += accu_by_task[task]
    average_accu = average_accu / len(args.subtasks)
    accu_by_task["Total"] = average_accu
    print(f"Average Accuracy on BLINK split val over all tasks is {round(100 * average_accu, 2)}%")
    return accu_by_task


def load_prompt(task_name, d, image_folder):
    need_disclaimer_tasks = []  # "Forensic_Detection", "Jigsaw", "Art_Style"
    disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"
    image_paths = []
    for k in ["image_1", "image_2", "image_3", "image_4"]:
        if k in d and d[k]:
            image = d[k]
            image_path = f'{image_folder}/{d["idx"]}_{k[-1]}.jpg'
            if not os.path.exists(image_path):
                image.save(image_path)
            image_paths.append(image_path)
    prompt = d["prompt"]
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    return image_paths, prompt


def build_dataset(task_name, data, image_folder):
    output_d = []
    for orig_d in data:
        idx = orig_d["idx"]
        gold_answer = orig_d["answer"]
        all_choices = ["(A)", "(B)", "(C)", "(D)", "(E)"][: len(orig_d["choices"])]
        image_paths, question = load_prompt(task_name, orig_d, image_folder)
        output_d.append(
            {
                "idx": idx,
                "images": image_paths,
                "question": question,
                "answer": gold_answer,
                "choices": all_choices,
            }
        )
    return output_d


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

    results = {}
    for task_name in args.subtasks:
        output_path = os.path.join(args.output_dir, f"{task_name.replace(' ', '_')}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_folder = os.path.join(args.image_dir, f"{task_name}_images")
        os.makedirs(image_folder, exist_ok=True)

        outputs = []
        data = load_dataset("BLINK-Benchmark/BLINK", task_name)[args.split]
        data_list = build_dataset(task_name, data, image_folder)[global_rank::world_size]

        for instance in tqdm(data_list, total=len(data_list), disable=global_rank != 0):
            contents = []
            for image_path in instance["images"]:
                image = PIL.Image.open(os.path.join(args.image_dir, image_path))
                contents.append(image)
            ques = instance["question"]
            if args.reason:
                ques = QUESTION_TEMPLATE_IMAGE.format(Question=ques)
            contents.append(ques)
            output = model.generate_content(contents, generation_config=generation_config)
            print(colored(instance["question"], "red", attrs=["bold"]))
            print(colored(output, "green", attrs=["bold"]))
            print(colored(instance["answer"], "blue", attrs=["bold"]))
            print(colored("-" * 100, "yellow", attrs=["bold"]))

            response = extract_ans(output)

            prediction = analyze_answer(instance, response)
            outputs.append(
                {
                    "idx": instance["idx"],
                    "question": instance["question"],
                    "answer": instance["answer"],
                    "full_prediction": output,
                    "prediction": prediction,
                }
            )

            # Wait for all processes to finish
            dist.barrier()
            if dist.size() > 1:
                outputs = dist.all_gather(outputs)
                if dist.is_main():
                    outputs = list(itertools.chain(*outputs))

                    json.dump(outputs, open(output_path, "w"), indent=4)
                    print(f"Saved {output_path}")

                    for entry in outputs:
                        results[entry["idx"]] = entry["prediction"]

        dist.barrier()
        if dist.size() > 1 and dist.is_main():
            results_save_path = f"{args.output_dir}/{args.split}_predictions.json"
            json.dump(results, open(results_save_path, "w"), indent=4)
            print(f"Saved {results_save_path}")

            if args.split == "val":
                accu_by_task = eval_prediction(results, args)
                print(accu_by_task)

                # save accu_by_task to json
                accu_by_task_save_path = f"{args.output_dir}/{args.split}_accu_by_task.json"
                json.dump(accu_by_task, open(accu_by_task_save_path, "w"), indent=4)
                print(f"Saved {accu_by_task_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="path/to/pretrained/svila-8b-rvila-compatible",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--reason",
        action="store_true", 
        default=False     
    )
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="path/to/svila-verl/eval/results/svila_blink",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default="path/to/SpatialAgent/data/BLINK/val_answers.json",
    )
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dtype", type=str, default="torch.float16")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--subtask", type=str, default="Spatial_Relation")
    args = parser.parse_args()

    args.subtasks = [
        # "Art_Style",
        # "Functional_Correspondence",
        # "Multi-view_Reasoning",
        # "Relative_Reflectance",
        # "Visual_Correspondence",
        # "Counting",
        # "IQ_Test",
        # "Object_Localization",
        # "Semantic_Correspondence",
        # "Visual_Similarity",
        # "Forensic_Detection",
        # "Jigsaw",
        # "Relative_Depth",
        # "Spatial_Relation",
        args.subtask
    ]

    eval_model(args)
