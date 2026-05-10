# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json

instance = json.load(open('spar-scannet-multiimg-select-cold-3d-instance.json'))
cot = json.load(open('spar-scannet-multiimg-select-cold-3d-cot.json'))
instance_dict = {}
for item in instance:
    instance_dict[item['id']] = item

print(len(instance), len(cot))
res = []
for i in range(len(cot)):
    assert cot[i]['id'] in instance_dict
    main_frame_idx = cot[i]['main_frame_idx']
    instances_list = []
    cot_3d_pos = cot[i]['3d_pos']
    instance_3d_pos = instance_dict[cot[i]['id']]['3d_pos']
    for k,v in cot_3d_pos.items():
        vp = k.replace('_', ' ')
        if k in instance_3d_pos and instance_3d_pos[k]['instance_name'] != 'unknown':
            name = instance_3d_pos[k]['instance_name']
        else:
            name = v['current_frame_3d']['label']
        # bbox = v['main_frame_3d']['bbox_2d']
        # bbox = [round(x/1000, 2) for x in bbox]
        # center = [round(x, 2) for x in v['all_frame_3d'][str(main_frame_idx)]['center_cam']]
        # print(cot_3d_pos.keys(), k, cot[i]['conversations'])
        center = [round(x, 2) for x in v['main_frame_3d']['center_cam']]
        name = name + f' ({vp})'
        # bbox = ','.join([str(c) for c in bbox])
        pos = ','.join([str(c) for c in center])
        instances_list.append(f"<3d_box center=\"{pos}\">{name}</3d_box>")
    detect = '\n'.join(instances_list)
    detect = f"After aligning to the reference frame {main_frame_idx}, find {len(instances_list)} relevant objects:\n" + detect
    think = cot[i]['cot']
    ans = cot[i]['conversations'][-1]['value']
    combine_cot = f"<detect>{detect}</detect>\n<think>{think}</think>\n<answer>{ans}</answer>"
    cot[i]['cot'] = combine_cot
    res.append(cot[i])

with open('spar-scannet-multiimg-select-dtr-coldstart.json', 'w') as f:
    json.dump(res, f, indent=4)

