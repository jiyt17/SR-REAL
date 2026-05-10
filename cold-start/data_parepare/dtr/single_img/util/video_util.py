import copy
import json
import os
import pickle
import random

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers.image_utils import to_numpy_array


def convert_from_uvd(u, v, d, intr, pose):
    # extr = np.linalg.inv(pose)

    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000

    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    world = pose @ np.array([x, y, z, 1])
    return world[:3] / world[3]


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = "".join(txt).replace("\n", " ")
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject(intrinsics, poses, depths, depth_scale):
    """
    intrinsics: (V, 4, 4)
    poses: (V, 4, 4)
    depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H * W)  # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H * W)  # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H * W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H * W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H * W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H * W)

    z = depths.view(V, H * W) / depth_scale  # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([x, y, z, torch.ones_like(x)], -1)  # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(
        -1
    )  # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords


class VideoProcessor:
    def __init__(
        self,
        video_folder,
        voxel_size=None,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy="uniform",
        val_box_type="pred",
    ):
        self.video_folder = video_folder
        self.frame_folder = os.path.join(video_folder, "data")
        self.annotation_dir = os.path.join(video_folder, "embodiedscan")
        self.voxel_size = voxel_size
        self.min_xyz_range = (
            torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        )
        self.max_xyz_range = (
            torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        )
        self.frame_sampling_strategy = frame_sampling_strategy
        self.scene = {}
        print(
            f"============frame sampling strategy: {self.frame_sampling_strategy}============="
        )

        for split in ["train", "val", "test"]:
            with open(
                os.path.join(self.annotation_dir, f"embodiedscan_infos_{split}.pkl"),
                "rb",
            ) as f:
                data = pickle.load(f)["data_list"]
                for item in data:
                    self.scene[item["sample_idx"]] = item
                    # item["sample_idx"]: "scannet/scene0415_00"
                    # if item["sample_idx"].startswith("scannet"):
                    #     self.scene[item["sample_idx"]] = item

        self.scan2obj = {}

        for split in ["train", "val"]:
            box_type = "gt" if split == "train" else val_box_type
            filename = os.path.join(
                self.video_folder, "metadata", f"scannet_{split}_{box_type}_box.json"
            )
            with open(filename) as f:
                data = json.load(f)
                self.scan2obj.update(data)

        if "mc" in self.frame_sampling_strategy:
            sampling_file = os.path.join(
                self.video_folder, "metadata", "scannet_select_frames.json"
            )
            self.mc_sampling_files = {}
            with open(sampling_file) as f:
                data = json.load(f)
                for dd in data:
                    self.mc_sampling_files[dd["video_id"]] = dd

            with open("data/metadata/pcd_discrete_0.1.pkl", "rb") as f:
                pc_data = pickle.load(f)
            self.pc_min = {}
            self.pc_max = {}
            for scene_id in pc_data:
                pc_points = pc_data[scene_id]
                min_xyz = [1000, 1000, 1000]
                max_xyz = [-1000, -1000, -1000]
                for data in pc_points:
                    min_xyz = [min(v1, v2) for v1, v2 in zip(min_xyz, data)]
                    max_xyz = [max(v1, v2) for v1, v2 in zip(max_xyz, data)]
                self.pc_min[scene_id] = torch.Tensor(min_xyz) / 10
                self.pc_max[scene_id] = torch.Tensor(max_xyz) / 10

    def sample_frame_files_mc(
        self, video_id: str, frames_upbound: int = 32, do_shift=False
    ):
        mc_files = self.mc_sampling_files[video_id]
        frame_files = mc_files["frame_files"][:frames_upbound]
        voxel_nums = mc_files["voxel_nums"][:frames_upbound]

        ratio = 1.0
        if "ratio95" in self.frame_sampling_strategy:
            ratio = 0.95
        elif "ratio90" in self.frame_sampling_strategy:
            ratio = 0.9

        if ratio != 1.0:
            num_all_voxels = mc_files["num_all_voxels"]
            out = []
            cc = 0
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                out.append(frame_file)
                cc += voxel_num
                if cc >= num_all_voxels * ratio:
                    break
            frame_files = out

        frame_files.sort(key=lambda file: int(file.split("/")[-1].split(".")[0]))
        # if do_shift:
        #     ori_len = len(frame_files)
        #     i = random.randint(0, len(frame_files)-1)
        #     frame_files = frame_files[-i:] + frame_files[:-i]
        #     assert len(frame_files) == ori_len
        return frame_files

    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        # video_file: scannet/scene00000_01

        # since the color images have the suffix .jpg
        # frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and os.path.join(video_file, f).endswith(".jpg")]
        # frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        meta_info = self.scene[video_id]
        # frame_files = [os.path.join(self.frame_folder, img["img_path"]) for img in meta_info["images"]]
        frame_files = meta_info["images"]

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        # For scannet, the RGB camera data is temporally synchronized with the depth sensor via hardware, providing synchronized depth and color capture at 30Hz
        # We follow embodiedscan by sampling one out of every ten images.
        avg_fps = 3

        total_frames = len(frame_files)
        sampled_indices = np.linspace(
            0, total_frames - 1, num_frames_to_sample, dtype=int
        )

        return [frame_files[i] for i in sampled_indices]

    def calculate_world_coords(
        self,
        video_id: str,
        frame_files,
        do_normalize=False,
    ):
        meta_info = self.scene[video_id]
        dataset_name = video_id.split("/")[0]
        scene_id = video_id.split("/")[-1]

        axis_align_matrix = torch.from_numpy(np.array(meta_info["axis_align_matrix"]))

        depths = []
        poses = []

        # Read and store the sampled frames
        for frame in frame_files:

            # depth image
            depth_path = os.path.join(self.frame_folder, frame["depth_path"])
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img).astype(np.int32)
                depths.append(torch.from_numpy(depth))

            # pose
            poses.append(torch.from_numpy(frame["cam2global"]))

        depths = torch.stack(depths)  # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses])  # (V, 4, 4)

        if dataset_name == "scannet":
            if "depth_cam2img" in meta_info:
                depth_intrinsic = np.array(meta_info.get("depth_cam2img"))[
                    :3, :3
                ]  # Use 3x3 intrinsic matrix
            else:
                depth_intrinsic = np.array(frame.get("depth_cam2img"))[:3, :3]
        else:
            if "cam2img" in meta_info:
                depth_intrinsic = np.array(meta_info.get("cam2img"))[
                    :3, :3
                ]  # Use 3x3 intrinsic matrix
            else:
                depth_intrinsic = np.array(frame.get("cam2img"))[:3, :3]

        depth_intrinsic = torch.from_numpy(depth_intrinsic)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)

        depth_scale = 4000 if dataset_name == "matterport3d" else 1000
        world_coords = unproject(
            depth_intrinsic.float(), poses.float(), depths.float(), depth_scale
        )  # (V, H, W, 3)

        if do_normalize:
            world_coords = torch.maximum(
                world_coords, self.pc_min[scene_id].to(world_coords.device)
            )
            world_coords = torch.minimum(
                world_coords, self.pc_max[scene_id].to(world_coords.device)
            )

        if "cam2img" in meta_info:
            image_intrinsic = np.array(meta_info.get("cam2img"))[
                :3, :3
            ]  # Use 3x3 intrinsic matrix
        else:
            image_intrinsic = np.array(frame.get("cam2img"))[:3, :3]

        return {
            "world_coords": world_coords,
            "poses": poses.numpy(),
            "image_intrinsic": image_intrinsic,
        }

    def preprocess(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):

        dataset_name = video_id.split("/")[0]
        assert dataset_name in ["scannet", "matterport3d", "3rscan", "arkitscenes"]

        if "mc" in self.frame_sampling_strategy:
            frame_files = self.sample_frame_files_mc(
                video_id,
                frames_upbound=frames_upbound,
                do_shift=("shift" in self.frame_sampling_strategy),
            )
        else:
            frame_files = self.sample_frame_files(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )

        video_dict = self.calculate_world_coords(
            video_id,
            frame_files,
            do_normalize=("norm" in self.frame_sampling_strategy),
        )
        world_coords = video_dict["world_coords"]
        V, H, W, _ = world_coords.shape
        world_coords_list = [world_coords[v].numpy() for v in range(V)]

        # boundry
        world_coords_flat = world_coords.reshape(-1, 3)
        x_min, x_max = (
            world_coords_flat[:, 0].min().item(),
            world_coords_flat[:, 0].max().item(),
        )
        y_min, y_max = (
            world_coords_flat[:, 1].min().item(),
            world_coords_flat[:, 1].max().item(),
        )
        z_min, z_max = (
            world_coords_flat[:, 2].min().item(),
            world_coords_flat[:, 2].max().item(),
        )
        boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        # Rotate to our coordinate system
        R_xup = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        rotated_coords = []
        for coords in world_coords_list:
            new_coords = coords.copy()
            rotated_coords.append(new_coords @ R_xup)
        flattened_rotated_coords = np.vstack(
            [pm.reshape(-1, 3) for pm in rotated_coords]
        )

        # Find min and max values across all points
        min_vals = flattened_rotated_coords.min(axis=0)
        max_vals = flattened_rotated_coords.max(axis=0)

        rotated_coords_norm = []
        for coords in rotated_coords:
            new_coords = coords.copy()
            new_coords = 2 * (new_coords - min_vals) / (max_vals - min_vals) - 1
            rotated_coords_norm.append(new_coords)

        rgb_frame_files = [
            os.path.join(self.frame_folder, frame["img_path"]) for frame in frame_files
        ]

        return {
            "frame_files": rgb_frame_files,
            "world_coords": world_coords_list,
            "video_size": len(frame_files),
            "boundry": boundry,
            "objects": self.scan2obj[video_id] if dataset_name == "scannet" else None,
            "world_coords_norm": rotated_coords_norm,
            "poses": video_dict["poses"],
            "image_intrinsic": video_dict["image_intrinsic"],
        }

    def process_3d_video(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        video_dict = self.preprocess(
            video_id,
            force_sample,
            frames_upbound,
        )
        return video_dict

    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = xyz - self.min_xyz_range.to(xyz.device)

        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()


def merge_video_dict(video_dict_list):
    new_video_dict = {}
    new_video_dict["box_input"] = []
    for k in video_dict_list[0]:
        if k in ["world_coords", "images", "objects"]:
            new_video_dict[k] = torch.stack(
                [video_dict[k] for video_dict in video_dict_list]
            )
        elif k in ["box_input"]:
            for video_dict in video_dict_list:
                if video_dict[k] is not None:
                    new_video_dict["box_input"].append(video_dict[k])

    new_video_dict["box_input"] = torch.Tensor(new_video_dict["box_input"])
    return new_video_dict
