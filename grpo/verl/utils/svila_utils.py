
import base64
import os
import tempfile
from io import BytesIO

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
# from pycocotools import mask as cocomask
from transformers import StoppingCriteria



def dynamic_preprocess(image, min_num=1, max_num=12, image_size=384, use_thumbnail=True):

    ########################################################################
    # NOTE(rvila): this is a modified version of the dynamic_preprocess function, supporting also np masks
    ########################################################################

    def get_dimensions(img):
        """Return (width, height) for either NumPy or PIL image."""
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]  # shape = (H, W, C)
            return w, h
        elif isinstance(img, Image.Image):
            return img.size  # (width, height)
        else:
            raise TypeError("Input image must be either a NumPy array or a PIL Image.")

    def resize_image(img, new_width, new_height):
        """Resize using cv2 for NumPy arrays, or .resize() for PIL."""
        if isinstance(img, np.ndarray):
            # cv2.resize expects (width, height) in (cols, rows) format
            # use nearest for binary mask
            if img.dtype == np.uint8:
                return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            else:
                return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        elif isinstance(img, Image.Image):
            return img.resize((new_width, new_height))
        else:
            raise TypeError("Unsupported image type for resizing.")

    def crop_image(img, x1, y1, x2, y2):
        """Crop using slicing for NumPy, or .crop() for PIL."""
        if isinstance(img, np.ndarray):
            return img[y1:y2, x1:x2, ...]
        elif isinstance(img, Image.Image):
            return img.crop((x1, y1, x2, y2))
        else:
            raise TypeError("Unsupported image type for cropping.")

    orig_width, orig_height = get_dimensions(image)
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = resize_image(image, target_width, target_height)
    processed_images = []
    for i in range(blocks):
        x1, y1, x2, y2 = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = crop_image(resized_img, x1, y1, x2, y2)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = resize_image(image, image_size, image_size)
        processed_images.append(thumbnail_img)

    return processed_images


def process_depth(depth_file, depth_folder=None):
    # this is for processing single-view relative depth w/o camera into xyzs
    if isinstance(depth_file, str):
        if depth_folder is not None:
            depth = Image.open(os.path.join(depth_folder, depth_file))
        else:
            depth = Image.open(depth_file)
    else:
        depth = depth_file

    # Depth is stored in PIL, trun to np and range [0,1]
    depth_max = np.max(np.array(depth))
    # print('depth max', depth_max)
    if depth_max <= 255:
        depth = np.array(depth) / 255  # (h,w)
    else:
        depth = np.array(depth) / depth_max  # (h,w)
    depth = depth_to_points_np(depth)
    return depth



def depth_to_points_np(depth, R=None, t=None):
    if depth.ndim == 2:
        depth = depth[np.newaxis, ...]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth to have 2 or 3 dims, got {depth.ndim}")

    if R is None:
        R = np.eye(3, dtype=depth.dtype)
    if t is None:
        t = np.zeros(3, dtype=depth.dtype)

    _, H, W = depth.shape  # depth has shape (1, H, W)
    # Create row and column indices
    y = np.arange(H, dtype=depth.dtype)
    x = np.arange(W, dtype=depth.dtype)
    # Use meshgrid with indexing='ij' to get shape (H, W)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Normalize x, y to range [-1..1]
    u_norm = 2.0 * (xx / (W - 1)) - 1.0  # shape (H, W)
    v_norm = 2.0 * (yy / (H - 1)) - 1.0  # shape (H, W)

    z_vals = -1.0 + 2.0 * depth  # shape (1, H, W)
    u_norm_3d = u_norm[np.newaxis, ...]  # (1, H, W)
    v_norm_3d = v_norm[np.newaxis, ...]  # (1, H, W)

    X = u_norm_3d * z_vals  # (1, H, W)
    Y = v_norm_3d * z_vals  # (1, H, W)
    Z = z_vals  # (1, H, W)

    points = np.concatenate([X, Y, Z], axis=0)  # (3, H, W)
    points = points.transpose(1, 2, 0)  # (H, W, 3)

    points_2d = points.reshape(-1, 3)  # (H*W, 3)
    points_2d = points_2d @ R.T + t  # apply rotation & translation
    points = points_2d.reshape(H, W, 3)  # back to (H, W, 3)
    return points


def dynamic_process_xyzs(xyzs, data_args, image_folder=None, max_tiles=None):
    all_xyzs = []
    for xyz in xyzs:
        processed_xyzs = process_xyz(xyz, data_args, enable_dynamic_res=True, max_tiles=max_tiles)
        all_xyzs.append(processed_xyzs)
    if all_xyzs:
        all_xyzs = torch.cat(all_xyzs)
    else:
        all_xyzs = None
    return all_xyzs


def process_xyz(xyz, data_args, enable_dynamic_res=False, enable_dynamic_s2=False, max_tiles=None):
    if hasattr(data_args.image_processor, "crop_size"):
        # CLIP vision tower
        crop_size = data_args.image_processor.crop_size
    else:
        # SIGLIP vision tower
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        xyzs, block_size = dynamic_s2_preprocess(
            xyz, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        xyzs = [torch.tensor(_xyz) for _xyz in xyzs]
        return torch.stack(xyzs), block_size
    if "dynamic" in data_args.image_aspect_ratio and enable_dynamic_res:
        assert crop_size["height"] == crop_size["width"]
        if max_tiles is not None:
            max_num = max_tiles
        else:
            max_num = data_args.max_tiles
        xyzs = dynamic_preprocess(xyz, min_num=data_args.min_tiles, max_num=max_num, image_size=crop_size["height"])
        xyzs = [torch.tensor(_xyz) for _xyz in xyzs]
        return torch.stack(xyzs)

    # if data_args.image_aspect_ratio == "resize":
    xyz = cv2.resize(xyz, (crop_size["width"], crop_size["height"]), interpolation=cv2.INTER_AREA)
    if data_args.image_aspect_ratio == "pad":
        raise NotImplementedError

    return torch.tensor(xyz)


def dynamic_process_images_and_prompt(images, prompt, data_args, image_folder=None, max_tiles=None):
    prompt = prompt.split(DEFAULT_IMAGE_TOKEN)
    idx = 0
    all_images = []
    block_lengths = []
    for img in images:
        processed_images = process_image(img, data_args, image_folder, enable_dynamic_res=True, max_tiles=max_tiles)
        all_images.append(processed_images)
        block_lengths.append(len(processed_images))
        prompt.insert(idx + 1, f"{DEFAULT_IMAGE_TOKEN}\n" * processed_images.shape[0])
        idx += 2
    prompt = "".join(prompt)
    if all_images:
        all_images = torch.cat(all_images)
    else:
        all_images = None
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, "")
    return all_images, prompt, block_lengths


def process_image(
    image_file, data_args, image_folder, enable_dynamic_res=False, enable_dynamic_s2=False, max_tiles=None
):
    processor = data_args.image_processor
    
    if isinstance(image_file, str):
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    image = image.convert("RGB")
    if hasattr(data_args.image_processor, "crop_size"):
        # CLIP vision tower
        crop_size = data_args.image_processor.crop_size
    else:
        # SIGLIP vision tower
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        images, block_size = dynamic_s2_preprocess(
            image, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images), block_size
    if "dynamic" in data_args.image_aspect_ratio and enable_dynamic_res:
        assert crop_size["height"] == crop_size["width"]
        if max_tiles is not None:
            max_num = max_tiles
        else:
            max_num = data_args.max_tiles
        images = dynamic_preprocess(image, min_num=data_args.min_tiles, max_num=max_num, image_size=crop_size["height"])
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images)

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    if data_args.image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        # Using default behavior of the vision encoder
        # For CLIP, default is central crop
        # For Radio, default is central crop
        # For Siglip, default is resize
        # For InternVIT, default is resize
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image



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
            elif isinstance(part, (PIL.Image.Image)):
                media["image"].append(part)
                text += MEDIA_TOKENS["image"]
            # elif isinstance(part, Video):
            #     if draft:
            #         media["video"].append(part)
            #     else:
            #         media["video"].append(_extract_video(part, config))
            #     text += MEDIA_TOKENS["video"]
            # elif isinstance(part, Mask):
            #     if draft:
            #         media["mask"].append(part)
            #     else:
            #         # NOTE(anjie): mask already extracted and stored inside json, no need to extract here
            #         media["mask"].append(part)
            elif isinstance(part, Depth):
                if draft:
                    media["depth"].append(part)
                else:
                    media["depth"].append(_extract_depth(part))
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media


def draw_visual_prompt(line, img):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    dot_radius = 20

    if 'red_point' in line:
        for point in line['red_point']:
            x = point[0] * w / 1000
            y = point[1] * h / 1000
            draw.ellipse(
                [(x - dot_radius, y - dot_radius),  # 左上角坐标
                (x + dot_radius, y + dot_radius)], # 右下角坐标
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
