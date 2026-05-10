#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=BUAADreamer/clevr_count_70k@test \
    data.val_files=BUAADreamer/clevr_count_70k@test \
    data.val_batch_size=32 \
    data.rollout_batch_size=32 \
    worker.actor.global_batch_size=32 \
    worker.actor.padding_free=true \
    worker.actor.ulysses_size=1 \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_clevr_run2 \
    trainer.n_gpus_per_node=8
