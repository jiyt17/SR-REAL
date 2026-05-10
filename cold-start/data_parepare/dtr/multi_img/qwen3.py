# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from transformers import AutoModelForCausalLM, AutoTokenizer
from sys_prompt import anno_instance_prompt
import json
import re
from tqdm import tqdm

model_name = "path/to/pretrained/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
spar = json.load(open('path/to/data/SPAR-3D/spar-scannet-multiimg-select-cold-3d.json'))
print('len:', len(spar))

res = []
spar = spar[len(res):]
chunk_size = 20
spar = [spar[i:i + chunk_size] for i in range(0, len(spar), chunk_size)]

for group in tqdm(spar):
    texts = []
    for data in group:
        ques = data["conversations"][0]['value'].strip()
        vp = list(data['3d_pos'].keys())
        vp = [v.replace('_', ' ') for v in vp]
        prompt = f"""Ques: {ques}\n\nAnno: {vp}"""
        # print('prompt:', prompt)

        messages = [
            {"role": "system", "content": anno_instance_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    # print(model_inputs.input_ids.shape)
    # print(f"Tokenizer padding side: {tokenizer.padding_side}")
    # print(f"padding: {tokenizer.pad_token_id}")

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )
    for i,data in enumerate(group):
        # print(generated_ids[i].tolist())
        output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        # print("content:", content)
        try:
            # json_str = re.search(r'\{[\s\S]*\}', content).group(0)
            json_str = content.replace("```json", "").replace("```", "").strip()
            # print('json_str:', json_str)
            instance_map = json.loads(json_str)
        except:
            instance_map = {}
            print('Error in JSON parsing!')
        print('instance_map:', instance_map)
        if len(instance_map) != len(data['3d_pos']):
            print('Length mismatch!')
            continue
        for k,v in data['3d_pos'].items():
            color_key = k.replace('_', ' ')
            if color_key in instance_map:
                v['instance_name'] = instance_map[color_key]
            else:
                v['instance_name'] = 'unknown'

    res = res + group
    with open('./spar-scannet-multiimg-select-cold-3d-instance.json', 'w') as f:
        json.dump(res, f, indent=4)