# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from sys_prompt import SPAR_3dcot_multiview_generate_prompt

import os
from google import genai
from google.genai import types
from tqdm import tqdm
import json

from PIL import Image, ImageDraw
from io import BytesIO
import time

dot_radius = 20
image_root = "/path/to/SPAR-7M-RGBD/spar/scannet/images/"

def draw_visual_prompt(line, img):
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


client = genai.Client(
    api_key="xxxxx",
)

model = "expert_model_api"

spar = json.load(open('/path/to/spar-scannet-multiimg-select-3d-cot-source.json', 'r'))
print(f"Total samples: {len(spar)}")

res = []
    
for line in tqdm(spar):
# for i in range(4):
    img_list = []
    for img in line["image"]:
        image_path = image_root + img
        img = Image.open(image_path)
        img_list.append(img)
    if 'point_img_idx' in line:
        points = []
        for k,v in line.items():
            if '_point' in k:
                points.append({k: v})
        # print(points)
        for point_id, img_id in enumerate(line['point_img_idx'][0]):
            img_list[img_id] = draw_visual_prompt(points[point_id], img_list[img_id])
    if 'bbox_img_idx' in line:
        bboxes = []
        for k,v in line.items():
            if '_bbox' in k:
                bboxes.append({k: v})
        # print(bboxes)
        for bbox_id, img_id in enumerate(line['bbox_img_idx'][0]):
            img_list[img_id] = draw_visual_prompt(bboxes[bbox_id], img_list[img_id])

    ques = line['conversations'][0]['value']
    ans = line['conversations'][1]['value']
    main_frame_idx = line['main_frame_idx']
    pos_3d = line['3d_pos']
    visual_prompt_3d = {}
    for k,v in pos_3d.items():
        bbox_2d = [coord / 1000 for coord in v['main_frame_3d']['bbox_2d']]
        center_3d = [round(coord, 2) for coord in v['main_frame_3d']['center_cam']]
        visual_prompt_3d[k] = {'bbox2d': bbox_2d, 'center3d': center_3d}
    # print(visual_prompt_3d)

    binary_img_list = []
    for img in img_list:
        byte_stream = BytesIO()          # 创建内存二进制流
        img.save(byte_stream, format='JPEG')  # 将图片以PNG格式存入流（格式需匹配原文件）
        binary_img = byte_stream.getvalue() 
        binary_img = types.Part.from_bytes(data=binary_img, mime_type="image/png")
        binary_img_list.append(binary_img)

    contents = [
        types.Content(
            role="user",
            parts= binary_img_list + [
                types.Part.from_text(text=f"""Reference Frame Index: {main_frame_idx}"""),
                types.Part.from_text(text=f"""Question: {ques}"""),
                types.Part.from_text(text=f"""Answer: {ans}"""),
                types.Part.from_text(text=f"""Scene Graph: {visual_prompt_3d}"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        system_instruction=SPAR_3dcot_multiview_generate_prompt,
        response_mime_type="text/plain", # "application/json", "text/plain"
    )
    while True:
        try:
            SG = ''''''
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                SG = SG + chunk.text
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    print('question:')
    print(ques)
    print('cot:')
    print(SG)
    line['cot'] = SG
    res.append(line)

    with open('spar-scannet-multiimg-select-cold-3d-cot.json', 'w') as f:
        json.dump(res, f, indent=4)