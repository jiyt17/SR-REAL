# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .flops_counter import VALID_MODLE_TYPE
from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import re
from functools import partial
from verl.utils.vila_remote_code.tokenizer_utils import tokenize_conversation
from verl.utils.vila_remote_code.auto_processor import extract_value_from_conv
from verl.utils.qwen_vl_utils import process_vision_info
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import random


MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
    "mask": "<mask>",
}

class Mask():
    def __init__(self, modality_type: str, content) -> None:
        self.modality_type = modality_type
        self.content = content

def _remove_media_tokens(text: str) -> str:
    for token in ["<image>", "<video>"]:
        text = text.replace(token + "\n", "").replace("\n" + token, "").replace(token, "")
    return text.strip()


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"
QUESTION_TEMPLATE_VIDEO = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"
QUESTION_TEMPLATE_IMAGE_DET = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first detect 3D centers of relevant objects, then think deeply about the question based on the given image, and finally provide the answer. The detection, reasoning and answer are enclosed within <detect> </detect>, <think> </think> and <answer> </answer> tags, respectively, i.e., <detect> detection here </detect>, <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"


def _get_messages_vila(example: Dict[str, Any],
                       prompt_key: str = "prompt",
                       image_key: str = "images",
                       image_dir: Optional[str] = None,
                       video_key: str = "videos",
                       video_dir: str = None,) -> Dict[str, Any]:
    if video_key in example:
        vision_key = "video"
        vision_value = example[video_key]
        if video_dir is not None and isinstance(vision_value, str):  # image paths
            vision_value = os.path.join(video_dir, vision_value)
        message_key = "video"
        question_template = QUESTION_TEMPLATE_VIDEO
    elif image_key in example:
        vision_key = "image"
        vision_value = example[image_key][0]
        if isinstance(vision_value, ImageObject):
            message_key = "image_pil"
        elif isinstance(vision_value, str):
            vision_key = "image"
        else:
            raise ValueError("Unknown image type", vision_value)
        question_template = QUESTION_TEMPLATE_IMAGE
    else:
        raise ValueError("Unsupported VILA for text only.")

    messages = [{"role": "user", "content": "<%s>" % vision_key + example[prompt_key]}]
    prompt = question_template.format(Question=messages[-1]['content'].replace("<%s>" % vision_key, ""))
    messages[-1]['content'] = [
        {"type": vision_key, message_key: vision_value},
        {"type": "text", "text": prompt},
    ]
    return messages, prompt

def _filter_overlong_prompts_vila(example: Dict[str, Any],
                                  tokenizer=None,
                                  max_prompt_length: int = 2048,
                                  prompt_key: str = "prompt",
                                  image_key: str = "images",
                                  image_dir: Optional[str] = None,
                                  video_key: str = "videos",
                                  video_dir: str = None,) -> bool:
    def apply_chat_template_vila(conversation):
        vila_conv = []
        for chat in conversation:
            vila_chat = {"from": "", "value": []}
            if chat["role"] in ("user", "system"):
                # user allows to input image and text
                vila_chat["from"] = "human" if chat["role"] == "user" else "system"
                vila_chat["value"] = extract_value_from_conv(chat)
            elif chat["role"] == "assistant":
                vila_chat["from"] = "gpt"
                vila_chat["value"] = extract_value_from_conv(chat)
            else:
                raise ValueError(f"Unsupported role: {chat['role']} in chat {chat}")
            vila_conv.append(vila_chat)
        return vila_conv

    messages, _ = _get_messages_vila(example, prompt_key, image_key, image_dir, video_key, video_dir)
    messages = apply_chat_template_vila(messages)
    messages[-1]['value'] = messages[-1]['value'][-1]
    inputs = tokenize_conversation(messages, tokenizer, add_generation_prompt=True,
                                   return_ids_only=False)
    return inputs.input_ids[0].size(-1) <= max_prompt_length

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        image_dir: Optional[str] = None,
        video_key: str = "videos",
        video_dir: str = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = False,
        vila_model: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.image_dir = image_dir
        self.video_key = video_key
        self.video_dir = video_dir
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.vila_model = vila_model

        self.video_hw = {}

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)
        # print(self.dataset, self.vila_model)
        # if data_split == "train":
        #     self.dataset = load_dataset(data_path, split="train[:99%]")
        # else:
        #     self.dataset = load_dataset(data_path, split="train[99%:]")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()
        
        if self.filter_overlong_prompts:
            if self.vila_model:
                _filter_overlong_prompts = partial(_filter_overlong_prompts_vila, tokenizer=self.tokenizer,
                                                   max_prompt_length=max_prompt_length, prompt_key=prompt_key,
                                                   image_key=image_key, image_dir=image_dir, video_key=video_key,
                                                   video_dir=video_dir)
            else:
                _filter_overlong_prompts = self._filter_overlong_prompts
            self.dataset = self.dataset.filter(
                _filter_overlong_prompts, desc="Filtering overlong prompts", num_proc=16,
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
            # print('2', prompt_str) # <image> ques prompt

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
            # print('3', content_list)
            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key] or []
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            resized_images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
            ] or None
            model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print('0', self.processor)
        example: dict = self.dataset[index]
        # print('example:', example) {problem, answer, images}
        messages = self._build_messages(example)
        # print('messages:', messages) # conversation format
        max_prompt_length = self.max_prompt_length
        if self.vila_model:
            messages, prompt = _get_messages_vila(example, self.prompt_key, self.image_key, self.image_dir, self.video_key, self.video_dir)
            # print('4', messages, prompt)
            vision_key = messages[-1]['content'][0]['type']
            messages = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # print('5', messages) # vila-format chat
            model_inputs = self.processor(text=[messages], return_tensors="pt")
            # print('model_inputs', model_inputs['media'][vision_key].shape) # 7 torch.Size([3, 448, 448]) 
            example["multi_modal_data"] = {
                vision_key: model_inputs['media'][vision_key],
            }
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["raw_prompt_ids"] = input_ids.tolist()
        else:
            if self.image_key in example:
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = example.pop(self.image_key)
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                    images = [os.path.join(self.image_dir, image) for image in images]
                resized_images = [
                    process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
                ] or None
                model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                example["multi_modal_data"] = {"images": images}
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                max_prompt_length += (input_ids==image_token_id).sum()
            elif self.video_key in example:
                video = example.pop(self.video_key)
                assert isinstance(video, str), "Only support video path input"
                if self.video_dir is not None:  # video paths
                    video = os.path.join(self.video_dir, video)
                messages = [{"role": "user", "content": [{"type": "video", "video": video, "nframes": self.processor.num_video_frames},
                                                         {"type": "text", "text": example[self.prompt_key]}]}]
                if "resized_height" in self.video_hw and "resized_width" in self.video_hw:
                    messages[0]["content"][0]["resized_height"] = self.video_hw["resized_height"]
                    messages[0]["content"][0]["resized_width"] = self.video_hw["resized_width"]

                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                if len(videos) > 0 and not "resized_height" in self.video_hw and not "resized_width" in self.video_hw :
                    self.video_hw["resized_height"], self.video_hw["resized_width"] = videos[0].size(2), videos[0].size(3)

                model_inputs = self.processor(text=prompt, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                example["multi_modal_data"] = {"video": videos}
                video_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
                max_prompt_length += (input_ids==video_token_id).sum()
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.max_prompt_length:
                if self.truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
                elif self.truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                elif self.truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
            example["raw_prompt_ids"] = raw_prompt_ids
        # print('process', example, input_ids.shape)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        # print('VF', input_ids.shape) # [2048]
        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        answer = example.pop(self.answer_key)
        if "<answer>" in answer:
            match = re.search(r"<answer>(.*?)</answer>", answer)
            example["ground_truth"] = match.group(1)
        else:
            example["ground_truth"] = answer
        return example



class SRDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        sys_prompt: str,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.enable_depth = True
        self.truncation = truncation

        self.data_path = data_path
        self.media_dir = 'path/to/data/SPAR-7M-RGBD/spar/scannet/images'
        self.instances = json.load(open(self.data_path, 'r'))

        if self.enable_depth:
            self.depth_dir = 'path/to/data/SPAR-7M-RGBD/scannet_svila/depth'

        self.resample_on_failure = True
        self.sys_prompt = sys_prompt


    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(instance["conversations"])
        # Remove media tokens from messages
        for message in messages:
            message["value"] = _remove_media_tokens(message["value"])
        ques = messages[0]['value']
        ans = messages[1]['value']
        first_main_frame = 'first image' in ques

        # Extract media from the instance
        media = defaultdict(list)
        if "image" in instance:
            for i,image_path in enumerate(instance["image"]):
                if 'path/to/dataset' in image_path:
                    media['image'].append(Image.open(image_path))
                    if self.enable_depth:
                        media['depth'].append(Image.open(instance["depth"][i]))
                else:
                    media['image'].append(Image.open(os.path.join(self.media_dir, image_path)))
                    if self.enable_depth:
                        image_path_without_extension = os.path.splitext(image_path)[0]
                        media['depth'].append(Image.open(os.path.join(self.depth_dir, f"{image_path_without_extension}.png")))
                        
            vision_key = "image"
            message_key = "image_pil"
            new_imgs = draw_visual_prompt(instance, media['image'])
            if '3d_pos' not in instance:
                pos_3d = None
                question_template = QUESTION_TEMPLATE_IMAGE
            else:
                pos_3d = instance['3d_pos']
                question_template = QUESTION_TEMPLATE_IMAGE_DET
            
        else:
            raise ValueError("Unknown image type")
            
        if "bbox" in instance and type(instance['bbox']) == dict: # multi-image mask
            for k,v in instance['bbox'].items():
                media['mask'].append(Mask('bbox', v))
        elif "bbox" in instance and len(instance['bbox']) > 0:
            media['mask'].append(Mask("bbox", instance['bbox']))
        else:
            media['mask'].append(None)

        messages = [{"role": "user", "content": "<%s>" % vision_key + ques}]
        prompt = question_template.format(Question=messages[-1]['content'].replace("<%s>" % vision_key, ""))
        messages[-1]['content'] = []
        for vision_value in new_imgs:
            messages[-1]['content'].append({"type": vision_key, message_key: vision_value})
        messages[-1]['content'].append({"type": "text", "text": prompt})
        
        return messages, prompt, media, ans, pos_3d, first_main_frame
    

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        # Process instance to conversation
        messages, prompt, media, ans, pos_3d, first_main_frame = self.process(instance)
        image_hw = instance.get("image_hw", None)
        if len(media['mask']) > 1: # multi-image mask
            mask = media['mask']
        else:
            mask = media['mask'][0] if media['mask'][0] else None

        vision_key = messages[-1]['content'][0]['type']
        messages = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.processor(text=[messages], depth=media['depth'], mask=mask, image_hw=image_hw, return_tensors="pt")

        # example = {'instance_id': instance['id']}
        example = {}
        example["multi_modal_data"] = {
            vision_key: model_inputs['media'][vision_key],
            "xyz": model_inputs['media']['xyz'],
            "mask": [None] * len(model_inputs['media'][vision_key])
        }
        if 'mask' in model_inputs['media']:
            example["multi_modal_data"]["mask"] = model_inputs['media']['mask']
        if len(media[vision_key]) > 1:
            example["multi_modal_data"]['block_lengths'] = [1] * len(media[vision_key])
        else:
            example["multi_modal_data"]['block_lengths'] = [len(model_inputs['media'][vision_key])]
        
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        example["raw_prompt_ids"] = input_ids.tolist()

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        # print('VF', input_ids.shape) # [2048]
        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        answer = ans
        if "<answer>" in answer:
            match = re.search(r"<answer>(.*?)</answer>", answer)
            gt = match.group(1)
        else:
            gt = answer

        # multi-img region
        if 'mask_idx' in instance:
            example['mask_idx'] = instance['mask_idx']
        elif 'bbox' in instance:
            example['mask_idx'] = len(instance['bbox'])
        else:
            example['mask_idx'] = 0

        example['ground_truth'] = {'gt': gt, 'pos_3d': pos_3d, 'first_main_frame': first_main_frame}
        return example


    def __len__(self):
        return len(self.instances)




def draw_visual_prompt_singleimg(line, img):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    dot_radius = 20

    if 'red_point' in line:
        for point in line['red_point']:
            x = point[0] * w / 1000
            y = point[1] * h / 1000
            draw.ellipse(
                [(x - dot_radius, y - dot_radius), 
                (x + dot_radius, y + dot_radius)], 
                fill="red",
            )

    if 'blue_point' in line:
        for point in line['blue_point']:
            x = point[0] * w / 1000
            y = point[1] * h / 1000
            draw.ellipse(
                [(x - dot_radius, y - dot_radius),
                (x + dot_radius, y + dot_radius)],
                fill="blue",
            )

    if 'green_point' in line:
        for point in line['green_point']:
            x = point[0] * w / 1000
            y = point[1] * h / 1000
            draw.ellipse(
                [(x - dot_radius, y - dot_radius),
                (x + dot_radius, y + dot_radius)],
                fill="green",
            )

    if 'red_bbox' in line:
        for bbox in line['red_bbox']:
            x1 = bbox[0] * w / 1000
            y1 = bbox[1] * h / 1000
            x2 = bbox[2] * w / 1000
            y2 = bbox[3] * h / 1000
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="red",
                width=5,
            )

    if 'blue_bbox' in line:
        for bbox in line['blue_bbox']:
            x1 = bbox[0] * w / 1000
            y1 = bbox[1] * h / 1000
            x2 = bbox[2] * w / 1000
            y2 = bbox[3] * h / 1000
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="blue",
                width=5,
            )

    if 'green_bbox' in line:
        for bbox in line['green_bbox']:
            x1 = bbox[0] * w / 1000
            y1 = bbox[1] * h / 1000
            x2 = bbox[2] * w / 1000
            y2 = bbox[3] * h / 1000
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="green",
                width=5,
            )

    if 'yellow_bbox' in line:
        for bbox in line['yellow_bbox']:
            x1 = bbox[0] * w / 1000
            y1 = bbox[1] * h / 1000
            x2 = bbox[2] * w / 1000
            y2 = bbox[3] * h / 1000
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="yellow",
                width=5,
            )

    if 'yellow_point' in line:
        for point in line['yellow_point']:
            x = point[0] * w / 1000
            y = point[1] * h / 1000
            draw.ellipse(
                [(x - dot_radius, y - dot_radius),
                (x + dot_radius, y + dot_radius)],
                fill="yellow",
            )

    return img

def draw_visual_prompt(line, imgs):
    if len(imgs) == 1:
        return [draw_visual_prompt_singleimg(line, imgs[0])]
    
    img_list = imgs
    if 'point_img_idx' in line:
        points = []
        for k,v in line.items():
            if '_point' in k:
                points.append({k: v})
        for point_id, img_id in enumerate(line['point_img_idx'][0]):
            img_list[img_id] = draw_visual_prompt_singleimg(points[point_id], img_list[img_id])
    if 'bbox_img_idx' in line:
        bboxes = []
        for k,v in line.items():
            if '_bbox' in k:
                bboxes.append({k: v})
        for bbox_id, img_id in enumerate(line['bbox_img_idx'][0]):
            img_list[img_id] = draw_visual_prompt_singleimg(bboxes[bbox_id], img_list[img_id])
    return img_list