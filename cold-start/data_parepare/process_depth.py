# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

# import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils import data
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# DATA LIST
IMAGE_DIR = "/path/to/SPAR-7M-RGBD/spar/scannet/images"
DEPTH_DIR = "/path/to/SPAR-7M-RGBD/scannet_svila/depth"
JSON_PATH = (
    "/path/to/SPAR-7M-RGBD/spar_scannet_multiimg_fill_r1.json"
)


def load_image_list():
    images_list = set()
    with open(JSON_PATH) as f:
        raw_samples = json.load(f)
    img_list = []
    for sample in raw_samples:
        img_list.extend(sample["image"])
    images_list = list(OrderedDict.fromkeys(img_list))
    print(f"Loaded {len(images_list)} unique images, from {len(raw_samples)} samples")
    return list(images_list)


def save_depth_image(save_path, depth_map):
    normalized_depth = (1 - (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())) * 255.0
    normalized_depth = normalized_depth.astype(np.uint8)
    depth_image = Image.fromarray(normalized_depth)
    # print(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth_image.save(save_path)
    if not os.path.exists(save_path):
        raise ValueError(f"Depth image not found at {save_path}")


class CustomDataset(data.Dataset):
    def __init__(self, root_dir, images_list, image_processor):
        self.root_dir = root_dir
        self.images_list = images_list
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images_list[idx])
        image = Image.open(image_path).convert("RGB")
        processed = self.image_processor(images=image, return_tensors="pt")

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "image_size": (image.height, image.width),
            "path": self.images_list[idx],
        }


def custom_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    image_sizes = [item["image_size"] for item in batch]
    paths = [item["path"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "image_size": image_sizes,
        "path": paths,
    }


def main():
    # # 1. Init distributed
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # # 2. Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 3. Load model
    image_processor = AutoImageProcessor.from_pretrained("/path/to/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("/path/to/Depth-Anything-V2-Large-hf")
    model.to(device)
    model.eval()

    # 4. Load dataset
    images_list = load_image_list()[rank::world_size]
    dataset = CustomDataset(IMAGE_DIR, images_list, image_processor)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, sampler=sampler)

    # 5. Process
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Rank {rank}"):
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values)
            post_processed = image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=batch["image_size"],
            )

            for instance, path in zip(post_processed, batch["path"]):
                depth_map = instance["predicted_depth"].detach().cpu().numpy()
                sample_name = os.path.splitext(path)[0]
                save_path = os.path.join(DEPTH_DIR, f"{sample_name}.png")
                save_depth_image(save_path, depth_map)

    # 6. Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
