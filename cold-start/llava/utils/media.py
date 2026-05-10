import glob
import os
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import requests
from transformers import PretrainedConfig

from llava.constants import MEDIA_TOKENS, MEDIA_TOKENS_RESERVED
from llava.media import Depth, Image, Mask, Video
from llava.utils import make_list
from llava.utils.logging import logger
from PIL import ImageDraw

__all__ = ["extract_media"]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image


def _extract_depth(depth: Union[Depth, PIL.Image.Image]) -> np.ndarray:
    if isinstance(depth, Depth):
        if depth.path.startswith("http://") or depth.path.startswith("https://"):
            depth = PIL.Image.open(requests.get(depth.path, stream=True).raw)
        else:
            depth = PIL.Image.open(depth.path)
    return depth


def _load_video_bytesio(video_bytesio: BytesIO, *, num_frames: int) -> List[PIL.Image.Image]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(video_bytesio.read())
        temp_video_name = temp_video.name
        return _load_video(temp_video_name, num_frames=num_frames)


def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            logger.warning(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        logger.warning("Extracting frames from video with specified FPS is not supported yet. Ignored.")
    if isinstance(video.path, BytesIO):
        frames = _load_video_bytesio(video.path, num_frames=num_frames)
    else:
        frames = _load_video(video.path, num_frames=num_frames)
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    # NOTE(rvila): keep <mask> tokens in the text
                    if token in part and token not in MEDIA_TOKENS_RESERVED.values():
                        logger.warning(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    media["video"].append(_extract_video(part, config))
                text += MEDIA_TOKENS["video"]
            elif isinstance(part, Mask):
                if draft:
                    media["mask"].append(part)
                else:
                    # NOTE(anjie): mask already extracted and stored inside json, no need to extract here
                    media["mask"].append(part)
            elif isinstance(part, Depth):
                if draft:
                    media["depth"].append(part)
                else:
                    media["depth"].append(_extract_depth(part))
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media

# for spar, draw visual prompts
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
