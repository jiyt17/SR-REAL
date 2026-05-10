# from matplotlib.path import Path
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_video_from_images(vis_frames, output_path, fps=30):
    """
    Create a video from a list of image paths.

    Args:
        vis_frames (list): List of paths to image files in the visualization directory
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
    """
    if not vis_frames:
        raise ValueError("Image frames list is empty")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read the first image to get dimensions
    first_image = cv2.imread(vis_frames[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {vis_frames[0]}")

    height, width = first_image.shape[:2]

    # Use mp4v codec instead of avc1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for frame_path in vis_frames:
            # Read image
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue

            # Ensure frame has the same dimensions as the first image
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            # Write frame to video
            out.write(frame)

    finally:
        # Release the VideoWriter
        out.release()

    print(f"Video saved to: {output_path}")


def get_image_size(image_path: str) -> tuple:
    """Determine the image size based on the provided path."""
    img = Image.open(image_path)
    return img.size[::-1]  # PIL returns (width, height), we need (height, width)


def load_vg_file(vg_path: str) -> List[Dict]:
    """Load the visual grounding annotation JSON file."""
    with open(vg_path) as f:
        return json.load(f)


def load_pkl_file(pkl_path: str) -> Dict:
    """Load the pkl file containing 3D information."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_mask_as_png(mask: np.ndarray, output_path: str):
    """Save the mask as a uint8 PNG file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(output_path)
    print(f"Saved mask to {output_path}")


def draw_and_save_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    corners_2d: np.ndarray,
    instance_id: int,
    label: str,
    save_path: str,
):
    """
    Draw 2D bounding box and projected corners on the image and save it.

    Args:
        image: Input image (H, W, 3)
        bbox: 2D bounding box coordinates [x_min, y_min, x_max, y_max]
        corners_2d: Projected 3D box corners in 2D (8, 2)
        instance_id: Instance ID for color selection
        save_path: Path to save the visualization
    """
    # Make a copy of the image
    vis_image = image.copy()

    # Generate a unique color for this instance
    color = plt.cm.rainbow(instance_id / 10.0)[:3]  # Get RGB from colormap
    color = tuple(int(c * 255) for c in color)  # Convert to OpenCV BGR format

    # Draw the 2D bounding box
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

    # Draw the projected corners
    for point in corners_2d.astype(np.int32):
        cv2.circle(vis_image, tuple(point), 3, color, -1)

    # Draw connections between corners (box edges)
    edge_indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]

    for start_idx, end_idx in edge_indices:
        start_point = tuple(corners_2d[start_idx].astype(np.int32))
        end_point = tuple(corners_2d[end_idx].astype(np.int32))
        cv2.line(vis_image, start_point, end_point, color, 1)

    # Add instance ID text
    cv2.putText(
        vis_image,
        f"ID: {instance_id} Label: {label}",
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    # Save the visualization
    cv2.imwrite(save_path, vis_image)


def debug_visualization(image_path, intrinsic, extrinsic, box, output_path):
    """
    Save a debug visualization of projected 3D box corners on the 2D image.

    Args:
        image_path (str): Path to the input image.
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
        extrinsic (np.ndarray): 4x4 camera extrinsic matrix.
        box (OrientedBoundingBox): 3D box to be visualized.
        output_path (str): Path to save the debug visualization.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    h, w, _ = image.shape

    # Get box corners
    corners = np.asarray(box.get_box_points())
    corners_homogeneous = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1
    )

    # Transform corners to camera coordinates
    extrinsic_w2c = np.linalg.inv(extrinsic)
    corners_camera = (extrinsic_w2c @ corners_homogeneous.T).T

    # Project to image plane
    corners_2d = (intrinsic @ corners_camera[:, :3].T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

    # Draw projected corners
    for pt in corners_2d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green dots for corners

    # Save the debug visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Debug visualization saved to {output_path}")


def load_image(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load and convert various image formats to numpy array in BGR format.

    Args:
        image: Input image as file path, PIL Image, or numpy array

    Returns:
        numpy array in BGR format
    """
    if isinstance(image, (str, Path)):
        # Load image from file path
        return cv2.imread(str(image))
    elif isinstance(image, Image.Image):
        # Convert PIL Image to BGR numpy array
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Handle numpy array input
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # Convert RGBA to BGR
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                # Assume BGR if 3 channels
                return image.copy()
        raise ValueError(f"Unsupported numpy array shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def plot_mask(
    image: Union[str, Path, np.ndarray, Image.Image],
    mask: Union[np.ndarray, Image.Image],
    save_path: Union[str, Path],
    color: Union[Tuple[int, int, int], str] = (0, 255, 0),
    opacity: float = 0.5,
    outline_thickness: int = 2,
    label: Optional[str] = None,
    label_position: Optional[Tuple[int, int]] = None,
    font_scale: float = 1.0,
    font_thickness: int = 2,
) -> None:
    """
    Plot a binary mask on an image with customizable appearance and save to file.

    Args:
        image: Input image as file path, PIL Image, or numpy array
        mask: Binary mask as numpy array or PIL Image (same size as image)
        save_path: Path where the output image will be saved
        color: Color of the mask as BGR tuple or color name string
        opacity: Opacity of the mask overlay (0.0 to 1.0)
        outline_thickness: Thickness of the mask outline (-1 for filled)
        label: Optional text label to display
        label_position: Optional custom position for label (x, y)
        font_scale: Scale of the font for the label
        font_thickness: Thickness of the font for the label
    """
    # Input validation
    assert 0.0 <= opacity <= 1.0, "Opacity must be between 0.0 and 1.0"
    assert outline_thickness >= -1, "Outline thickness must be >= -1"

    # Load and convert image
    img_array = load_image(image)
    if img_array is None:
        raise ValueError("Failed to load image")

    # Convert mask to numpy array if needed
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Ensure mask is binary and same size as image
    mask = mask.astype(bool)
    if mask.shape[:2] != img_array.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {img_array.shape[:2]}"
        )

    # Convert color name to BGR if string is provided
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
        "cyan": (255, 255, 0),
        "white": (255, 255, 255),
    }
    if isinstance(color, str):
        color = color_map.get(color.lower(), (0, 255, 0))

    # Create a copy of the input image
    result = img_array.copy()

    # Create mask overlay
    mask_overlay = np.zeros_like(img_array)
    mask_overlay[mask > 0] = color

    # Blend the mask with the image
    cv2.addWeighted(mask_overlay, opacity, result, 1 - opacity, 0, result)

    # Draw outline if thickness > 0
    if outline_thickness > 0:
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, outline_thickness)

    # Add label if provided
    if label:
        # If position not specified, place label at top-left of mask
        if label_position is None:
            if len(contours) > 0:
                # Use the top-left point of the bounding box
                x, y, w, h = cv2.boundingRect(contours[0])
                label_position = (x, max(y - 10, 20))  # Place above the mask
            else:
                label_position = (20, 20)  # Default position if no contours

        # Draw text with background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # Draw background rectangle
        cv2.rectangle(
            result,
            (label_position[0] - 5, label_position[1] - text_size[1] - 5),
            (label_position[0] + text_size[0] + 5, label_position[1] + 5),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            result,
            label,
            label_position,
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    # Save the result
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    cv2.imwrite(str(save_path), result)
