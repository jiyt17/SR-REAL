#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Perflow-Shuai/ours@train \
    data.val_files=Perflow-Shuai/ours_eval_debug@test \
    data.video_dir=path/to/datasets/longvila_r1_data \
    data.val_batch_size=32 \
    data.rollout_batch_size=32 \
    worker.actor.global_batch_size=32 \
    worker.actor.padding_free=true \
    worker.actor.ulysses_size=1 \
    worker.rollout.num_video_frames=8 \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_clevr_run2 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.val_freq=-1

