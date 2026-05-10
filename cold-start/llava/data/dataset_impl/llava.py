# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import glob
import os
import random
from typing import Any, Dict, List, Optional

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.data.dataset_impl.utils import _remove_media_tokens
from llava.media import Depth, Image, Mask, Video
from llava.utils import io, make_list

__all__ = ["LLaVADataset", "LLaVANextDataset", "LLaVANextVideoDataset"]


class LLaVADataset(BaseDataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        media_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        is_video=False,
        data_num=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.data_path = data_path
        self.media_dir = media_dir
        self.instances = io.load(self.data_path)
        global_batch_size = kwargs.get("global_batch_size", None)
        if data_num:
            data_num = int(data_num)
            random.shuffle(self.instances)
            self.instances = self.instances[:data_num]
        self.is_video = is_video or any(["video" in instance for instance in self.instances])

        self.enable_depth = self.data_args.enable_depth
        self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic"
        self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2"

        if self.enable_depth:
            assert depth_dir is not None, f"depth_dir is required when enable_depth is True for {self.name}"
            self.depth_dir = depth_dir

        if global_batch_size is not None:
            residual = global_batch_size - len(self.instances) % global_batch_size
            if residual != global_batch_size:
                if global_batch_size // len(self.instances) >= 2:
                    self.instances = self.instances * (global_batch_size // len(self.instances))
                    residual = global_batch_size - len(self.instances) % global_batch_size
                selected_elements = random.sample(range(len(self.instances)), residual)
                additional_instance = [self.instances[i] for i in selected_elements]
                self.instances.extend(additional_instance)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(instance["conversations"])

        # Extract media from the instance
        medias = []
        if "image" in instance:
            for i,image_path in enumerate(make_list(instance["image"])):
                if 'path/to/dataset' in image_path:
                    medias.append(Image(image_path))
                    if self.enable_depth:
                        medias.append(Depth(instance["depth"][i]))
                else:
                    medias.append(Image(os.path.join(self.media_dir, image_path)))
                    if self.enable_depth:
                        image_path_without_extension = os.path.splitext(image_path)[0]
                        medias.append(Depth(os.path.join(self.depth_dir, f"{image_path_without_extension}.png")))
            

        # NOTE(ligeng): quick workaround for idefics2_sft
        if "images" in instance:
            for image_path in make_list(instance["images"]):
                medias.append(Image(os.path.join(self.media_dir, image_path)))

        if "video" in instance:
            for video_path in make_list(instance["video"]):
                medias.append(Video(os.path.join(self.media_dir, video_path)))

        # NOTE(rvila): dataset contains region input, select one modality here
        if "rle" in instance or "segmentation" in instance or "bbox" in instance:
            region_modality_list = [_ for _ in ["rle", "segmentation", "bbox"] if _ in instance.keys()]
            region_modality = random.choice(region_modality_list)
            # multi-img
            if type(instance['bbox']) == dict:
                for k,v in instance['bbox'].items():
                    medias.append(Mask('bbox', v))
            else:
                if len(instance[region_modality]) > 0:
                    medias.append(Mask(region_modality, instance[region_modality]))
            

        # Remove media tokens from messages
        for message in messages:
            message["value"] = _remove_media_tokens(message["value"])

        # Add media to the beginning of the first message
        messages[0]["value"] = medias + [messages[0]["value"]]
        return messages


class LLaVANextDataset(BaseDataset):
    def __init__(self, data_path: str, media_dir: str, is_video=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.media_dir = media_dir
        self.instances = io.load(self.data_path)
        self.is_video = is_video or any(["video" in instance for instance in self.instances])
        self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic" and not self.is_video
        self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2" and not self.is_video

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        datasource = instance.get("datasource", None)
        messages = instance["conversations"]

        if "image" in instance:
            img_list = []
            for img_path in instance["image"]:
                img_list.append(Image(os.path.join(self.media_dir, img_path)))

            # replace all <image> tokens in the messages
            for idx1, msg in enumerate(messages):
                # value = messages[0]["value"]
                value = messages[idx1]["value"]
                img_tok_len = len(DEFAULT_IMAGE_TOKEN)
                new_value = []

                while value.find(DEFAULT_IMAGE_TOKEN) >= 0:  # still has <image>
                    idx = value.find(DEFAULT_IMAGE_TOKEN)
                    if idx > 0:
                        new_value.append(value[:idx])
                    new_value.append(img_list.pop(0))
                    value = value[idx + img_tok_len :]
                new_value.append(value)
                messages[idx1]["value"] = new_value

                # FIXME(ligeng): this is an interesting bug... if we feed [{"from": "gpt"}, {"from": "user"}] to the model, it will throw errors.
                if datasource == "twitter_post":
                    # warnings.warn(f"{index} {datasource} enforcing the role for twitter_post datasource")
                    role = "human" if idx1 % 2 == 0 else "gpt"
                    messages[idx1]["from"] = role

            assert (
                len(img_list) == 0
            ), f"#Num of <images> does not match the number of images in the instance. {instance}"
        return messages


class LLaVANextVideoDataset(BaseDataset):
    def __init__(self, data_path: str, media_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.media_dir = media_dir
        self.instances = io.load(self.data_path)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = instance["conversations"]

        if "video" in instance:
            img_flist = glob.glob(os.path.join(self.media_dir, instance["video"]) + "/*.jpeg")
            vpath = os.path.join(self.media_dir, instance["video"])

            assert len(img_flist) > 0, f"no images found in {vpath}"
            value = messages[0]["value"]
            img_list = [Image(img_path) for img_path in img_flist]
            new_value = [*img_list, value.replace(DEFAULT_IMAGE_TOKEN, "").strip()]
            messages[0]["value"] = new_value
        return messages
