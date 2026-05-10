# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json

instance = json.load(open('spar-scannet-singleimg-select-cold-3d-instance.json'))
cot = json.load(open('spar-scannet-singleimg-select-cold-3d-cot.json'))

print(len(instance), len(cot))
res = []
for i in range(len(instance)):
    assert instance[i]['id'] == cot[i]['id']
    instances_list = []
    for k,v in instance[i]['3d_pos'].items():
        vp = k.replace('_', ' ')
        name = v['instance_name'] if v['instance_name'] != 'unknown' else v['label']
        center = [round(x, 2) for x in v['center_cam']]
        # <3d_box center=\"0.02,1.06,2.03\">cabinet</3d_box>
        name = name + f' ({vp})'
        pos = ','.join([str(c) for c in center])
        instances_list.append(f"<3d_box center=\"{pos}\">{name}</3d_box>")
    detect = '\n'.join(instances_list)
    detect = f"Find {len(instances_list)} relevant objects:\n" + detect
    think = cot[i]['cot']
    ans = instance[i]['conversations'][-1]['value']
    combine_cot = f"<detect>{detect}</detect>\n<think>{think}</think>\n<answer>{ans}</answer>"
    instance[i]['cot'] = combine_cot
    res.append(instance[i])

with open('spar-scannet-singleimg-select-ground-coldstart.json', 'w') as f:
    json.dump(res, f, indent=4)

