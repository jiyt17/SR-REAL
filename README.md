# SR-REAL

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)]()
[![deploy](https://img.shields.io/badge/Hugging%20Face-SR_REAL-FFEB3B)](https://huggingface.co/jiyatai/SR-REAL-RL)


## Introduction

This repository contains the official implementation of "Reinforcing Dual-Path Reasoning in Spatial Vision Language Models".

We present a unified framework that equips a spatial VLM with two complementary reasoning paths: Language-Only Reasoning (LOR), which performs step-by-step linguistic deduction, and Detect-Then-Reason (DTR), which detects 3D geometric cues (e.g., centers or bounding boxes) via region tokens before explicit geometric inference.

SR-REAL starts with a cold-start supervised fine-tuning stage, where we construct LOR and DTR chain-of-thought supervision and introduce a region-to-3D interface. This is followed by reinforcement learning, which optimizes the policy model with accuracy, format, and detection rewards.

## Cold Start

1. Create the environment

```bash
cd cold-start 
./environment_setup.sh sr-3d
```

2. Data preparation

We use [SPAR](https://github.com/LogosRoboticsGroup/SPAR) as the data source for constructing LOR and DTR chain-of-thought data.

```bash
cd SR-REAL/cold-start/data_parepare
```

For LOR CoT construction, first run `expert.py` to generate chain-of-thought rationales, and then run `process.py` to convert them into instruction-tuning data.

For DTR CoT construction, follow these steps:

- Run `deal.py` to match annotated objects from SPAR with EmbodiedScan and obtain their 3D coordinates.
- Run `expert.py` to generate reasoning chains conditioned on the image, question, and object coordinates.
- Run `qwen3.py` to extract object names from SPAR.
- Run `process.py` to combine the CoT rationales, object names, and coordinate information into DTR question-answer instruction-tuning data.
- Run `process_region.py` to produce the final instruction-tuning data with region prompts.

You can also directly download our generated cold-start CoT data from [Hugging Face](https://huggingface.co/datasets/jiyatai/spar-cot).

During cold-start training, we additionally mix CoT data with 2D/3D grounding data, spatial region QA data, and general multimodal instruction-tuning data. Please configure the corresponding data paths in `SR-REAL/cold-start/llava/data/registry/datasets/cs-oci-ord.yaml` before training.

The spatial region data is derived from [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT). The 3D grounding data is derived from [Omni3D](https://github.com/facebookresearch/omni3d), [CA1M](https://github.com/apple/ml-cubifyanything), [OmniNOCS](https://github.com/google-deepmind/omninocs), and other sources. Our processed region-to-3D data is available on [Hugging Face](https://huggingface.co/datasets/jiyatai/2D-to-3D-grounding).

3. SFT Training

Download the pretrained [SR-3D weights](https://huggingface.co/a8cheng/sr3d-nvila-8b-singleview-pretrain).

Start training:

```bash
bash scripts/sft.sh
```

## GRPO

1. Create the RL environment

```bash
cd grpo
conda create -n verl python=3.10 -y
conda activate verl
bash install.sh
```

2. Convert the cold-start model to Transformers format (add remote code)

```bash
python scripts/transform.py
```

3. RL Training

Our RL training data mainly comes from SPAR and SpatialRGPT, covering spatial multiple-choice and fill-in-the-blank questions.

The RL data follows the same format as instruction-tuning data, but the answer field contains only the final correct answer. For DTR-style RL data, we additionally provide 3D coordinate annotations and region prompts. The data processing pipeline is similar to the CoT construction process described above. The processed data is available on [Hugging Face](https://huggingface.co/datasets/jiyatai/SR-REAL-rldata).

```bash
bash run_r1_region.sh
```

4. Test

Merge model weights:

```bash
python scripts/model_merger.py --local_dir rl_model_actor_path
```

For benchmark preparation, please refer to [EVAL.md](cold-start/eval/EVAL.md). Then run:

```bash
cd cold-start/eval
bash eval.sh
```

## Citation

## Acknowledgement

This project builds on code from several repositories, especially [SR-3D](https://github.com/AnjieCheng/SR-3D), [Long-RL](https://github.com/NVlabs/Long-RL), and [veRL](https://github.com/verl-project/verl). We sincerely thank the authors for their excellent work.
