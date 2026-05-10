#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=path/to/models/qwen2-7b-longvila-256f-reasoning-128-withporker-sft-rl-version-2  # replace it with your local file path
cp verl/utils/vila_remote_code/* ${MODEL_PATH}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Perflow-Shuai/ours@train \
    data.val_files=Perflow-Shuai/ours_eval_debug@test \
    data.video_dir=path/to/datasets/longvila_r1_data \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.vila_model=true \
    data.val_batch_size=1 \
    data.rollout_batch_size=8 \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.num_video_frames=2048 \
    worker.rollout.n=2 \
    worker.rollout.tokens_per_frame=257 \
    worker.rollout.max_num_batched_tokens=540000 \
    worker.reward.reward_function=./examples/reward_function/r1v_vila.py:compute_score \
    worker.vila_model=true \
    worker.actor.model.trust_remote_code=true \
    worker.actor.ulysses_size=8 \
    worker.rollout.trust_remote_code=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=nvila_lite_2b_clevr \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 
