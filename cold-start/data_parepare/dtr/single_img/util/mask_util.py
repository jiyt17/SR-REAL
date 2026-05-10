import numpy as np
from pycocotools import mask as mask_utils


def mask_to_rle_numpy(mask: np.ndarray):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    h, w = mask.shape

    # Put in fortran order and flatten h,w
    mask = np.transpose(mask).flatten()

    # Compute change indices
    diff = mask[1:] ^ mask[:-1]
    change_indices = np.where(diff)[0]

    # Encode run length
    cur_idxs = np.concatenate(([0], change_indices + 1, [h * w]))
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if mask[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())

    return {"size": [h, w], "counts": counts}


def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle
