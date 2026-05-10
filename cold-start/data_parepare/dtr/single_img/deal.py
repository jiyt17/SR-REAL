# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Object tracking across frames - based on fixed region_vg_grounding.py
Generate questions asking to locate objects and provide frame-by-frame bounding boxes
"""
import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
from tqdm import tqdm
import pdb
try:
    import torch
except ImportError:
    torch = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util.conv_util import *
from util.depth_util import *
from util.io_util import *
from util.mask_util import *
from util.math_util import *
from util.scene_util import *

def _9dof_to_box(box, label=None, color_selector=None, color=None):
    if isinstance(box, list):
        box = np.array(box)
    if torch and isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    
    if o3d is None:
        raise ImportError("open3d is required for oriented bounding box operations")
    
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
    return geo

def project_3d_bbox_to_2d(center, extent, extrinsic, intrinsic, image_size):
    """Project 3D bounding box to 2D using camera parameters"""
    # CRITICAL FIX: EmbodiedScan uses 9-DOF bounding boxes with rotation!
    # Use the proven _9dof_to_box function from the working codebase
    
    # If this is called with 9-DOF box_3d, use oriented bbox
    if len(center) == 9:
        box_3d = center  # Rename for clarity
        
        if o3d is None:
            return None
        
        def _9dof_to_box_local(box):
            if isinstance(box, list):
                box = np.array(box)
            if torch and isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            center = box[:3].reshape(3, 1)
            scale = box[3:6].reshape(3, 1)
            rot = box[6:].reshape(3, 1)
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
            geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
            return geo, center.reshape(-1)
        
        # Create oriented bounding box using the working method
        box_obb, box_center = _9dof_to_box_local(box_3d)
        
        # Get the 8 corners of the oriented bounding box
        corners_3d = np.asarray(box_obb.get_box_points())
        all_points_3d = np.vstack([corners_3d, box_center])
    else:
        # Fallback to axis-aligned approach (for backward compatibility)
        corners_3d = np.array([
            [-extent[0]/2, -extent[1]/2, -extent[2]/2],
            [+extent[0]/2, -extent[1]/2, -extent[2]/2],
            [+extent[0]/2, +extent[1]/2, -extent[2]/2],
            [-extent[0]/2, +extent[1]/2, -extent[2]/2],
            [-extent[0]/2, -extent[1]/2, +extent[2]/2],
            [+extent[0]/2, -extent[1]/2, +extent[2]/2],
            [+extent[0]/2, +extent[1]/2, +extent[2]/2],
            [-extent[0]/2, +extent[1]/2, +extent[2]/2],
        ]) + center
    
    # Add homogeneous coordinate
    all_points_3d_homo = np.hstack([all_points_3d, np.ones((9, 1))])
    
    # Transform to camera coordinate system
    # CRITICAL FIX: Invert extrinsic matrix to get world-to-camera transformation
    world_to_cam = np.linalg.inv(extrinsic)
    # print(world_to_cam.shape, corners_3d_homo.shape)
    all_points_cam = (world_to_cam @ all_points_3d_homo.T).T[:, :3]

    center_cam = all_points_cam[-1]
    corners_cam = all_points_cam[:-1]
    
    # Only keep points in front of camera
    valid_mask = corners_cam[:, 2] > 0.1  # Small threshold to avoid near-zero depth
    if not np.any(valid_mask):
        return None
    
    corners_cam_valid = corners_cam[valid_mask]
    
    # Project to image plane
    corners_2d = (intrinsic @ corners_cam_valid.T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
    
    # Get bounding box in 2D
    min_x, min_y = corners_2d.min(axis=0)
    max_x, max_y = corners_2d.max(axis=0)
    
    # Check if projection is within image bounds
    width, height = image_size
    if max_x < 0 or min_x > width or max_y < 0 or min_y > height:
        return None
    
    # Clip to image bounds and normalize to [0, 1]
    bbox_2d = [
        max(0, min_x) / width,
        max(0, min_y) / height,
        min(width, max_x) / width,
        min(height, max_y) / height
    ]
    
    # Check if bbox is valid and has reasonable size
    bbox_width = bbox_2d[2] - bbox_2d[0]
    bbox_height = bbox_2d[3] - bbox_2d[1]
    
    # Relax the size constraints to be more permissive for very large objects
    if bbox_width > 0.001 and bbox_height > 0.001 and bbox_width <= 1.0 and bbox_height <= 1.0:
        return bbox_2d, center_cam
    else:
        return None


def get_all_visible_frames_for_object(obj_id, scene_data, sampled_indices):
    """Get all frames where the object is visible with their frame indices"""
    visible_frames = []
    
    for sampled_idx, frame_idx in enumerate(sampled_indices):
        frame_data = scene_data["images"][frame_idx]
        if obj_id in frame_data.get("new_visible_instance_ids", []):
            visible_frames.append((sampled_idx, frame_idx))
    
    return visible_frames

def track_object_across_frames(obj_id, scene_data, instance_dict, dataset_path, sampled_indices):
    """Track object across all visible frames and return frame-by-frame bounding boxes"""
    tracking_results = []
    
    # Get 3D bbox info from instance
    instance = instance_dict[obj_id]
    box_3d = instance["bbox_3d"]
    print(f"Tracking object {obj_id} with 3D box {box_3d}")
    
    # Get all visible frames
    visible_frames = get_all_visible_frames_for_object(obj_id, scene_data, sampled_indices)
    
    if not visible_frames:
        return [], "not_visible_in_sampled_frames"
    
    projection_failures = 0
    
    # Process each visible frame
    for sampled_idx, frame_idx in visible_frames:
        print(f"Processing frame {frame_idx} (sampled idx {sampled_idx}) for object {obj_id}")
        frame_data = scene_data["images"][frame_idx]
        
        try:
            # Get camera parameters
            cam2global = np.array(frame_data["cam2global"]).reshape(4, 4)
            axis_align_matrix = np.array(scene_data.get("axis_align_matrix"))
            extrinsic = axis_align_matrix @ cam2global
            
            # Get image size
            image_path = os.path.join(dataset_path, frame_data["img_path"])
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Get camera intrinsics from scene data
            if "cam2img" not in scene_data:
                continue
            
            intrinsic = np.array(scene_data["cam2img"])[:3, :3]
            image_size = (width, height)
            
            # Project 3D bbox to 2D using the full 9-DOF box for oriented projection
            bbox_2d = project_3d_bbox_to_2d(box_3d, None, extrinsic, intrinsic, image_size)
            print(f"Projected 2D bbox: {bbox_2d}")
            
            if bbox_2d is not None:
                tracking_results.append({
                    "sampled_frame_idx": sampled_idx,
                    "original_frame_idx": frame_idx,
                    "bbox_2d": bbox_2d
                })
            else:
                projection_failures += 1
                
        except Exception as e:
            print(f"Error processing frame {frame_idx} for object {obj_id}: {e}")
            projection_failures += 1
            continue
    
    if not tracking_results:
        if projection_failures > 0:
            return [], f"projection_failed_in_all_{len(visible_frames)}_visible_frames"
        else:
            return [], "no_valid_projections"
    
    return tracking_results, "success"

def draw_bbox(bbox, number, img):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    x1 = bbox[0] * w 
    y1 = bbox[1] * h 
    x2 = bbox[2] * w 
    y2 = bbox[3] * h 
    draw.rectangle(
        [(x1, y1), (x2, y2)],
        outline="red",
        width=4,
    )
    font = ImageFont.load_default()
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 根据位置参数确定数字位置
    padding = 5
    text_x = x1 + padding
    text_y = y1 + padding
    
    # 绘制背景矩形
    bg_rect = [
        text_x - padding, 
        text_y - padding, 
        text_x + text_width + padding, 
        text_y + text_height + padding
    ]
    draw.rectangle(bg_rect, fill="green")
    
    # 绘制数字
    draw.text((text_x, text_y), text, fill="white", font=font)

    return img

def draw_visual_prompt(line, img):
    dot_radius = 20
    w, h = img.size
    draw = ImageDraw.Draw(img)

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


def id_to_obb_corners(instance_dict, obj_id):
    """Get oriented bounding box corners"""
    instance = instance_dict[obj_id]
    box_obb = _9dof_to_box(instance["bbox_3d"])
    return np.asarray(box_obb.get_box_points()).tolist()

def compute_iou(boxA, boxB):
    # box = [x1, y1, x2, y2] in normalized coordinates
    # Convert to absolute coordinates for IoU calculation
    boxA = [boxA[0], boxA[1], boxA[2], boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[2], boxB[3]]

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def generate_3D_data_spar(
    vg_path: str, pkl_path: str,
):
    """Generate object tracking data with frame-by-frame bounding boxes"""
    
    # Load data
    # vg_data = load_vg_file(vg_path)
    pkl_data = load_pkl_file(pkl_path)
    # print(vg_data[0])
    
    # Get class labels
    meta_info = pkl_data["metainfo"]
    # classes = list(meta_info["categories"].keys())
    print(meta_info)
    id_cate = {}
    for k,v in meta_info["categories"].items():
        id_cate[v] = k

    scene_dict = {}
    for scene in pkl_data['data_list']:
        scene_dict[scene['sample_idx']] = scene
    
    dataset_image_size = {}
    conversations = []
    spar = json.load(open('data/SPAR-3D/spar-scannet-singleimg-select.json'))
    print(len(spar))
    res = []
    box_num = 0
    point_num = 0
    box_num_org = 0
    point_num_org = 0

    for item in tqdm(spar):
        scan_id = 'scannet/' + item['image'][0].split('/')[0]
        print('*'*10, scan_id)
        scan_list = []
        try:
            scene_data = scene_dict[scan_id]
        except:
            print('skip due to no scene data')
            continue
        instance_dict = build_instance_dict(scene_data["instances"])
        with open('data/SPAR-7M-RGBD/spar/scannet/images/' + item['image'][0].replace('jpg', 'txt').replace('image_color', 'pose')) as f:
            cam2global = [[float(x) for x in line.split()] for line in f.readlines()]
        # print(cam2global)

        image_path = os.path.join('data/SPAR-7M-RGBD/spar/scannet/images', item['image'][0])
        img = Image.open(image_path)
        width, height = img.size

        cam2global = np.array(cam2global).reshape(4, 4)
        axis_align_matrix = np.array(scene_data.get("axis_align_matrix"))
        extrinsic = axis_align_matrix @ cam2global
        
        intrinsic = np.array(scene_data["cam2img"])[:3, :3]
        image_size = (width, height)
        obj_3d_2d = {}

        # img = draw_visual_prompt(item, img)

        for obj_id, instance in instance_dict.items():
            box_3d = instance["bbox_3d"]
            
            # Project 3D bbox to 2D using the full 9-DOF box for oriented projection
            out = project_3d_bbox_to_2d(box_3d, None, extrinsic, intrinsic, image_size)
            if out is not None:
                bbox_2d, center_cam = out
                bbox_2d = [int(x * 1000) for x in bbox_2d]
                obj_3d_2d[obj_id] = {
                    'bbox_2d': bbox_2d,
                    'center_cam': center_cam.tolist(),
                    'label': id_cate[instance['bbox_label_3d']]
                }
        #     print(f"Projected 2D bbox: {bbox_2d}")
        #         if bbox_2d:
        #             img = draw_bbox(bbox_2d, instance['bbox_label_3d'], img)
        
        # img.save('test.png')
        visual_prompts = {}
        box_flag = 0
        for k,v in item.items():
            if 'point' in k or 'bbox' in k:
                visual_prompts[k] = v
            if 'bbox' in k:
                box_flag = 1
        visual_prompt_3d = {}
        flag = 1
        if box_flag:
            box_num_org += 1
            for k,v in visual_prompts.items():
                if 'bbox' in k:
                    obj_num = 0
                    for obj_id, obj in obj_3d_2d.items():
                        iou = compute_iou(v[0], obj['bbox_2d'])
                        if iou > 0.74:
                            visual_prompt_3d[k] = obj
                            obj_num += 1
                    if obj_num != 1:
                        flag = 0
                        break
            if flag == 0:
                print('skip due to multiple or zero box match')
                continue
            box_num += 1
        else:
            point_num_org += 1
            if point_num >= 3400:
                continue
            for k,v in visual_prompts.items():
                if 'point' in k:
                    min_dist = 40
                    obj_num = 0
                    for obj_id, obj in obj_3d_2d.items():
                        cx = (obj['bbox_2d'][0] + obj['bbox_2d'][2]) / 2
                        cy = (obj['bbox_2d'][1] + obj['bbox_2d'][3]) / 2
                        dist = math.sqrt((v[0][0]-cx)**2 + (v[0][1]-cy)**2)
                        if dist < min_dist:
                            visual_prompt_3d[k] = obj
                            obj_num += 1
                    if obj_num != 1:
                        flag = 0
                        break
            if flag == 0:
                print('skip due to multiple or no point match')
                continue
            point_num += 1
        
        print(visual_prompt_3d)
        item['3d_pos'] = visual_prompt_3d
        res.append(item)

    print(len(res))
    print('box_num:', box_num, box_num_org)
    print('point_num:', point_num, point_num_org)
    random.shuffle(res)
    with open('./spar-scannet-singleimg-select-cold-3d.json', 'w') as f:
        json.dump(res, f, indent=4)

   

def main():

    vg_path = f"SVILA-3D-Data/embodiedscan/embodiedscan_train_vg.json"
    pkl_path = f"SVILA-3D-Data/embodiedscan/embodiedscan_infos_train.pkl"

    generate_3D_data_spar(
        vg_path=vg_path,
        pkl_path=pkl_path,
    )

if __name__ == "__main__":
    main()