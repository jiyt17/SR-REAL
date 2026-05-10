import json
import os
import pickle
import sys
from typing import Dict, List

import cv2
import numpy as np
from open3d.cuda.pybind.geometry import OrientedBoundingBox
from PIL import Image


def project_3d_box_to_2d(
    box: OrientedBoundingBox,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_size: tuple,
) -> np.ndarray:
    """Project a 3D bounding box onto a 2D plane to create a mask."""
    mask = np.zeros(image_size, dtype=np.uint8)
    h, w = image_size

    # Get box corners in 3D space
    corners = np.asarray(box.get_box_points())
    corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]

    # Convert to homogeneous coordinates
    corners_homogeneous = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1
    )

    # Transform corners to camera coordinates
    extrinsic_w2c = np.linalg.inv(extrinsic)  # World-to-camera transformation
    corners_camera = (extrinsic_w2c @ corners_homogeneous.T).T

    # Project valid corners onto the image plane
    corners_2d = (intrinsic @ corners_camera[:, :3].T).T
    corners_2d = corners_2d[:, :2] / np.abs(
        corners_2d[:, 2:3]
    )  # Normalize by abs depth

    # Clip projected points to image bounds
    corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, w - 1)
    corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, h - 1)

    # Define box faces for polygon filling (check if valid)
    box_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]

    # Fill each face in the mask
    for face in box_faces:
        face_points = corners_2d[face].astype(np.int32)
        cv2.fillPoly(mask, [face_points], 1)

    return mask


def project_3d_box_to_2d_box(
    box: OrientedBoundingBox,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_size: tuple,
    instance_id: int,
) -> np.ndarray:
    """
    Project a 3D bounding box onto a 2D plane to create a bounding box.
    Returns the 2D bounding box as [x_min, y_min, x_max, y_max].
    """
    h, w = image_size
    # Get box corners in 3D space
    corners = np.asarray(box.get_box_points())
    # Convert to homogeneous coordinates
    corners_homogeneous = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1
    )
    # Transform corners to camera coordinates
    extrinsic_w2c = np.linalg.inv(extrinsic)  # World-to-camera transformation
    corners_camera = (extrinsic_w2c @ corners_homogeneous.T).T

    # Project all corners, handling those behind camera
    corners_2d = np.zeros((corners_camera.shape[0], 2))
    for i in range(corners_camera.shape[0]):
        if corners_camera[i, 2] <= 0:  # Behind or on camera plane
            # Project from a small positive z value instead
            corners_camera[i, 2] = 0.1

        # Project to image plane
        point = intrinsic @ corners_camera[i, :3].T
        corners_2d[i] = point[:2] / point[2]

    # Get 2D bounding box coordinates
    x_min = np.clip(np.min(corners_2d[:, 0]), 0, w - 1)
    y_min = np.clip(np.min(corners_2d[:, 1]), 0, h - 1)
    x_max = np.clip(np.max(corners_2d[:, 0]), 0, w - 1)
    y_max = np.clip(np.max(corners_2d[:, 1]), 0, h - 1)

    return np.array([x_min, y_min, x_max, y_max]), corners_2d
