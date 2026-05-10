import random
from typing import Any, Dict, List

import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llava.mm_utils import (
    dynamic_process_images_and_prompt,
    dynamic_process_masks,
    dynamic_process_xyzs,
    dynamic_s2_process_images_and_prompt,
    process_depth,
    process_images,
    process_masks,
    process_xyzs,
)
from llava.train.args import DataArguments
from llava.utils.logging import logger
from llava.utils.media import extract_media, draw_visual_prompt
from llava.utils.tokenizer import preprocess_conversation


__all__ = ["BaseDataset"]


def _process_mask(masks: List[Any], image_hw: tuple, data_args: DataArguments) -> torch.Tensor:
    return process_masks(masks, image_hw, data_args.image_processor, data_args)


def _process_xyz(xyzs: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_xyzs(xyzs, data_args.image_processor, data_args)


def _process_image(images: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_images(images, data_args.image_processor, data_args)


def _process_video(videos: List[Any], data_args: DataArguments) -> torch.Tensor:
    return [_process_image(video, data_args) for video in videos]


class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        no_system_prompt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.no_system_prompt = no_system_prompt
        self.instances = []
        self.enable_depth = False
        self.enable_dynamic_res = False
        self.enable_dynamic_res_s2 = False
        # global_batch_size: int,
        self.global_batch_size = kwargs.get("global_batch_size", 1)

        # by default, dataset cls will resample on failure
        self.resample_on_failure = kwargs.get("resample_on_failure", True)

        # by default, dataset cls will resample on failure
        self.resample_on_failure = kwargs.get("resample_on_failure", True)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        try:
            # Process instance to conversation
            conversation = self.process(instance)

            # Extract media from conversation
            media = extract_media(conversation, self.data_args)

            block_sizes = []
            # Process media
            if "image" in media:
                # print('get item img', media["image"])
                if type(media["image"]) is list: # SPAR
                    new_img = draw_visual_prompt(instance, media['image'])
                    media['image'] = new_img
                if self.enable_dynamic_res_s2:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                # multi img not tiling
                elif len(media["image"]) == 1 and self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic": # True
                    processed_images, processed_prompt, block_lengths = dynamic_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                    conversation[0]["value"] = processed_prompt
                else:
                    processed_images = _process_image(media["image"], self.data_args)
                    block_lengths = [1] * processed_images.size(0)

            if ("depth" in media) and self.enable_depth:
                xyzs = [process_depth(depth) for depth in media["depth"]]
                if self.enable_dynamic_res_s2:
                    raise NotImplementedError("Dynamic res s2 not implemented for depth")
                # multi img not tiling
                elif len(media["depth"]) == 1 and self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic":
                    processed_xyzs = dynamic_process_xyzs(xyzs, self.data_args)  # processed_xyzs = (x,448,448,3)
                else:
                    processed_xyzs = _process_xyz(xyzs, self.data_args)

            if "mask" in media:
                if "image_hw" in instance:
                    image_hw = instance["image_hw"]
                else:
                    raise KeyError("image_hw not found in instance")

                if self.enable_dynamic_res_s2:
                    raise NotImplementedError("Dynamic res s2 not implemented for mask")
                elif len(media["mask"]) == 1 and self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic":
                    processed_masks = dynamic_process_masks(media["mask"], image_hw, self.data_args)
                    # print('mask', len(processed_masks), processed_masks[0].shape) # 13 torch.Size([region_num, 448, 448])
                else:
                    processed_masks = _process_mask(media["mask"], image_hw, self.data_args)
                    # print('mask', len(processed_masks), processed_masks[0].shape)

            if "video" in media:
                if self.enable_dynamic_res_s2 and self.data_args.video_max_tiles > 1:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        media["video"][0],
                        conversation[0]["value"],
                        self.data_args,
                        max_tiles=self.data_args.video_max_tiles,
                    )
                    # For HighRes video training, we use <image> token instead of <vila/video>
                    conversation[0]["value"] = processed_prompt.replace("<vila/video>", "")
                elif (
                    self.enable_dynamic_res
                    and self.data_args.image_aspect_ratio == "dynamic"
                    and self.data_args.video_max_tiles > 1
                ):
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["video"][0],
                        conversation[0]["value"],
                        self.data_args,
                        max_tiles=self.data_args.video_max_tiles,
                    )
                    # For HighRes video training, we use <image> token instead of <vila/video>
                    conversation[0]["value"] = processed_prompt.replace("<vila/video>", "")
                else:
                    processed_images = _process_video(media["video"], self.data_args)

            # Prepare "input_ids" and "labels" for training
            data = preprocess_conversation(conversation, self.tokenizer, no_system_prompt=self.no_system_prompt)

            if self.enable_dynamic_res_s2 and ("image" in media or "video" in media):
                data["block_sizes"] = block_sizes

            if "image" in media:
                num_images = processed_images.size(0)
                data["image"] = processed_images
                data["block_lengths"] = [block_lengths]
            else:
                num_images = 0
                data["block_lengths"] = [[0]]

            if "video" in media:
                if (
                    self.enable_dynamic_res_s2 == True or self.enable_dynamic_res == True
                ) and self.data_args.video_max_tiles > 1:
                    # HighRes video training
                    data["image"] = processed_images
                else:
                    data["video"] = processed_images

            if "depth" in media and self.enable_depth:
                data["xyz"] = processed_xyzs

            if "mask" in media:
                data["mask"] = processed_masks
                assert len(data["mask"]) == len(data["image"])
            else:
                data["mask"] = [None] * num_images  # handle multi image inputs

            assert (
                int(np.sum(data["block_lengths"])) == num_images
            ), f"{int(np.sum(data['block_lengths']))}, {num_images}"

            if "image" in media and self.enable_depth:
                assert processed_images.size(0) == processed_xyzs.size(
                    0
                ), f"image/depth size mismatch: image {processed_images.size(0)}, {processed_xyzs.size(0)}"

        except Exception as e:
            if not self.resample_on_failure:
                raise e
            else:
                logger.exception(f"Error processing instance '{instance}': '{e}'. Resampling.")
                return self.__getitem__(random.randint(0, len(self.instances) - 1))

        # multi-img region
        if 'mask_idx' in instance:
            data['mask_idx'] = instance['mask_idx']
        elif 'bbox' in instance:
            data['mask_idx'] = len(instance['bbox'])
        else:
            data['mask_idx'] = 0
            
        return data

    def __len__(self) -> int:
        return len(self.instances)
