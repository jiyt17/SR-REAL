# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
# from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def find_all_positions(text, pattern):
    positions = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1  
    return positions

def get_bounding_box_from_point(xy, wh=50):
    x = xy[0]
    y = xy[1]
    half_size = wh / 2  # since the width and height are x, half of it is 2
    box = [x - half_size, y - half_size, x + half_size, y + half_size]
    if box[0] < 0:
        box[0] = 0
    if box[1] < 0:
        box[1] = 0
    if box[2] > 1000:
        box[2] = 999
    if box[3] > 1000:
        box[3] = 999
    return box

QUESTION_TEMPLATE_IMAGE = "You are a helpful assistant. The user asks a question, and then you solve it.\n\nPlease first detect 3D centers of relevant objects, then think deeply about the question based on the given image, and finally provide the answer. The detection, reasoning and answer are enclosed within <detect> </detect>, <think> </think> and <answer> </answer> tags, respectively, i.e., <detect> detection here </detect>, <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {Question}"


data = json.load(open('data/SPAR-3D/spar-scannet-singleimg-select-ground-coldstart.json', 'r'))
visual_prompts = ['red bbox', 'blue bbox', 'green bbox', 'yellow bbox', 'red point', 'blue point', 'green point', 'yellow point']
for item in tqdm(data):
    ques = item['conversations'][0]['value']
    img = 'data/SPAR-7M-RGBD/spar/scannet/images/' + item['image'][0]
    img = Image.open(img).convert('RGB')
    h = img.height
    w = img.width
    item['image_hw'] = [h, w]
    # print(h, w)
    pos2region = {}
    for vp in visual_prompts:
        if vp in ques:
            pos = find_all_positions(ques, vp)
            for p in pos:
                pos2region[p] = vp
    # print(ques)
    # print(pos2region)
    # sort by key
    pos2region = dict(sorted(pos2region.items()))
    # print(pos2region)
    bbox = []
    for p,vp in pos2region.items():
        if 'point' in vp:
            point_xy = item['_'.join(vp.split())][0]
            box = get_bounding_box_from_point(point_xy)
        else:
            box = item['_'.join(vp.split())][0].copy()
        # convert to absolute coordinates
        box[0] = int(box[0] * w / 1000)
        box[1] = int(box[1] * h / 1000)
        box[2] = int(box[2] * w / 1000)
        box[3] = int(box[3] * h / 1000)
        bbox.append(box)
    item['bbox'] = bbox
    for vp in visual_prompts:
        if vp in ques:
            ques = ques.replace(vp, vp + ' <mask>')
    ques = QUESTION_TEMPLATE_IMAGE.format(Question=ques)
    item['conversations'][0]['value'] = ques
    item['conversations'][1]['value'] = item['cot']

    # regionx2vp = []
    # for p,vp in pos2region.items():
    #     if vp not in regionx2vp:
    #         regionx2vp.append(vp)
    # vp2regionx = {}
    # for i,vp in enumerate(regionx2vp):
    #     vp2regionx[vp] = f"Region [{i+1}]"

    # item['vp2regionx'] = vp2regionx

with open('data/SPAR-3D/spar-scannet-singleimg-select-ground-coldstart-region.json', 'w') as f:
    json.dump(data, f, indent=4)
