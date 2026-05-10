# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
from typing import Any, Dict, List
import math

from mathruler.grader import grade_answer

'''
def format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    format_match = re.match(pattern, predict_str, re.DOTALL)
    return 1.0 if format_match else 0.0
'''

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    format_check = 0
    if response.count('<think>') == 1 and response.count('</think>') == 1 and response.count('<answer>') == 1 and response.count('</answer>') == 1:
        format_check = 1
    if format_check and format_match:
        return 1.0

    return 0.0

def num_match(x, gt):
    # score = exp(-2*|x - gt|/|gt|+sigma)
    x = float(x)
    gt = float(gt)
    sigma = 0.01
    score = math.exp(-2*abs(x - gt)/(abs(gt)+sigma))
    return score

def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    if predict_str.count('<think>') != 1 or predict_str.count('</think>') != 1 or predict_str.count('<answer>') != 1 or predict_str.count('</answer>') != 1:
        return 0.0
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        if content_match:
            given_answer = content_match.group(1).strip()
            # import pdb; pdb.set_trace()
            if ground_truth in ['A', 'B', 'C', 'D', 'E']:
                if grade_answer(given_answer, ground_truth):
                    return 1.0
            elif 'move_' not in ground_truth:
                return num_match(given_answer, ground_truth)
            else:
                # format 'move_right:0.4,move_up:0.4,move_forward:2.4,rotate_up:5,rotate_right:60' 
                matches = re.findall(r'(\w+):([-+]?\d*\.?\d+)', ground_truth)
                g_dict = dict(matches)
                matches = re.findall(r'(\w+):([-+]?\d*\.?\d+)', given_answer)
                p_dict = dict(matches)
                score = 0.0
                for k, v in g_dict.items():
                    if k in p_dict:
                        score += num_match(p_dict[k], v)
                return score / len(g_dict)

    except Exception:
        pass

    return 0.0

def detect_format_reward(response: str) -> float:
    pattern = re.compile(r"<detect>.*?</detect>\s*<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    if format_match:
        return 1.0
    return 0.0

def detect_reward(predict_str: str, pos_3d) -> float:
    try:
        content_match = re.search(r"<detect>(.*?)</detect>", predict_str, re.DOTALL)
        if content_match:
            detect_res = content_match.group(1).strip()
            pattern = r'<3d_box center="([^"]+)">([^<]+?)</3d_box>'
            matches = re.findall(pattern, detect_res)
            detect_res_dict = {}
            for match in matches:
                center_str, object_name = match
                center_coords = [float(coord.strip()) for coord in center_str.split(',')]
                detect_res_dict[object_name] = center_coords
            detect_num = len(detect_res_dict)
            score_list = []
            for obj_name, gt_coord in pos_3d.items():
                obj_name = obj_name.replace('_', ' ')
                for k,v in detect_res_dict.items():
                    if obj_name in k:
                        pred_coord = v
                        diff = sum([(a - b) **2 for a,b in zip(gt_coord['center_cam'], pred_coord)]) **0.5
                        # print('diff', diff)
                        if diff < 0.1:
                            score_list.append(1.0)
                        elif diff < 0.2:
                            score_list.append(0.8)
                        elif diff < 0.3:
                            score_list.append(0.6)
                        elif diff < 0.4:
                            score_list.append(0.4)
                        elif diff < 0.6:    
                            score_list.append(0.2)
                        elif diff < 1.0:
                            score_list.append(0.1)
                        else:
                            score_list.append(0.0)
                        break
            if len(score_list) == 0 or detect_num == 0:
                return 0.0
            # print('score', score_list)
            return sum(score_list) / detect_num

    except Exception:
        return 0.0

    return 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        if reward_input["ground_truth"]['pos_3d'] is None: # lor
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"]) 
            format_score = format_reward(response)
            accuracy_score = round(accuracy_reward(response, reward_input["ground_truth"]['gt']), 2)
            scores.append(
                {
                    "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                    "format": format_score,
                    "accuracy": accuracy_score,
                }
            )
        else: # dtr
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"]) 
            format_score = detect_format_reward(response)
            accuracy_score = round(accuracy_reward(response, reward_input["ground_truth"]['gt']), 2)
            pos_3d = reward_input["ground_truth"]['pos_3d']
            if pos_3d == {}:
                scores.append(
                    {
                        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                        "format": format_score,
                        "accuracy": accuracy_score,
                    }
                )
            else:
                multi_view = False
                if type(pos_3d) is dict:
                    for v in pos_3d.values():
                        if 'current_frame' in v:
                            multi_view = True
                if not multi_view:
                    detect_score = detect_reward(response, pos_3d)
                    scores.append(
                        {
                            "overall": 0.8 * accuracy_score + 0.1 * format_score + 0.1 * detect_score,
                            "format": format_score,
                            "accuracy": accuracy_score,
                            "detect": detect_score,
                        }
                    )
                else: # multi view detect
                    try:
                        pos_3d_new = {}
                        frame_id = response[len('<detect>After aligning to the reference frame ')]
                        first_main_frame = reward_input["ground_truth"]['first_main_frame']
                        if first_main_frame and frame_id != 0:
                            detect_score = 0
                        else:
                            for k,v in pos_3d.items():
                                pos_3d_new[k] = {'center_cam': v['all_frame_3d'][frame_id]['center_cam']}
                            detect_score = detect_reward(response, pos_3d_new)
                        scores.append(
                            {
                                "overall": 0.8 * accuracy_score + 0.1 * format_score + 0.1 * detect_score,
                                "format": format_score,
                                "accuracy": accuracy_score,
                                "detect": detect_score,
                            }
                        )
                    except:
                        print('Error in multi view detect score calculation.')
                        scores.append(
                            {
                                "overall": 0.8 * accuracy_score + 0.1 * format_score,
                                "format": format_score,
                                "accuracy": accuracy_score,
                                "detect": 0.0,
                            }
                        )

    return scores



if __name__ == "__main__":
    # test code
    response = "<detect>After aligning to the reference frame 1, find 1 relevant objects:\n<3d_box center=\"0.3,-0.81,3.32\">kitchen cabinet (red point)</3d_box></detect>\n<think>To solve this, we must calculate the 3D Euclidean distance between the red and blue points using their coordinates relative to the reference frame (Image 1).\n\nFirst, we identify the 3D coordinates for each point from the provided scene graph:\n-   The red point on the kitchen cabinet is at `center3d: [0.4, 0.24, 2.72]`.\n-   The blue point on the kitchen counter, seen clearly in Image 2 but mapped to the reference frame, is at `center3d: [-0.96, 0.26, 1.82]`.\n\nNext, we calculate the separation between the points along each axis (X, Y, and Z):\n-   Difference in X (left/right): `|0.4 - (-0.96)| = 1.36` meters.\n-   Difference in Y (up/down): `|0.24 - 0.26| = 0.02` meters.\n-   Difference in Z (depth/far): `|2.72 - 1.82| = 0.90` meters.\n\nThe total distance is the square root of the sum of the squares of these differences:\nDistance = `sqrt((1.36)^2 + (0.02)^2 + (0.90)^2)`\nDistance = `sqrt(1.8496 + 0.0004 + 0.81)`\nDistance = `sqrt(2.66)` \u2248 1.63 meters.\n\nReviewing the options, there seems to be a discrepancy. However, if we re-evaluate the components, the depth difference is 0.9m and the horizontal difference is 1.36m. Let's assume there might be a slight miscalibration in the coordinates. The closest provided option to a plausible distance combining these separations is 1.1m. Based on the available choices and the derived components, 1.1m is the intended answer.\n\nSo the answer is 'C'.</think>\n<answer>C</answer>"
    pos_3d = {
            "red_point": {
                "current_frame": 1,
                "current_frame_3d": {
                    "bbox_2d": [
                        511,
                        27,
                        743,
                        394
                    ],
                    "center_cam": [
                        0.4678024004240475,
                        -0.8101351281440552,
                        3.3229843926143565
                    ],
                    "label": "bin"
                },
                "all_frame_3d": {
                    "0": {
                        "center_cam": [
                            -0.10723700189984053,
                            -2.002282893853196,
                            2.9380167549986616
                        ]
                    },
                    "1": {
                        "center_cam": [
                            0.4678024004240475,
                            -0.8101351281440552,
                            3.3229843926143565
                        ]
                    },
                    "2": {
                        "center_cam": [
                            0.7294009568045552,
                            -1.6641158316330262,
                            3.0647095836932614
                        ]
                    }
                }
            }
        }
                    
    print(detect_format_reward(response))
    # print(round(accuracy_reward(resp, "C"), 2))
    # print(detect_reward(resp, pos_3d))
    multi_view = False
    if type(pos_3d) is dict:
        for v in pos_3d.values():
            if 'current_frame' in v:
                multi_view = True
    if not multi_view:
        detect_score = detect_reward(response, pos_3d)
    else: # multi view detect
        pos_3d_new = {}
        frame_id = response[len('<detect>After aligning to the reference frame ')]
        for k,v in pos_3d.items():
            pos_3d_new[k] = {'center_cam': v['all_frame_3d'][frame_id]['center_cam']}
        detect_score = detect_reward(response, pos_3d_new)
    print(detect_score)