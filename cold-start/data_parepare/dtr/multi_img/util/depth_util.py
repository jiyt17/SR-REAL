import numpy as np
from PIL import Image


def preprocess_instrinsic(
    intrinsic, ori_size, target_size
):  # (V, 4, 4) (resize_shape) (h, w)

    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)

    intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
    intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

    # for crop transform
    intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

    if intrinsic.shape[0] == 1:
        intrinsic = intrinsic.squeeze(0)

    return intrinsic


def preprocess_depth_image(
    depth_image, new_image_size, do_depth_scale=True, depth_scale=1000
):
    # resize to new_image_size (w,h)
    width, height = depth_image.size
    resized_depth_image = depth_image.resize((new_image_size), Image.NEAREST)

    # rescale the depth image
    img = np.array(resized_depth_image)
    if do_depth_scale:
        img = img / depth_scale

    return img
