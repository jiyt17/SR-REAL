#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=path/to/models/NVILA-Lite-2B-hf-preview-model  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=BUAADreamer/clevr_count_70k@test \
    data.val_files=BUAADreamer/clevr_count_70k@test \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.vila_model=true \
    data.val_batch_size=32 \
    data.rollout_batch_size=32 \
    worker.actor.global_batch_size=32 \
    worker.reward.reward_function=./examples/reward_function/r1v_vila.py:compute_score \
    worker.vila_model=true \
    worker.actor.model.trust_remote_code=true \
    worker.actor.ulysses_size=2 \
    worker.rollout.trust_remote_code=true \
    worker.rollout.padding_free=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=nvila_lite_2b_clevr_sp2 \
    trainer.n_gpus_per_node=8
