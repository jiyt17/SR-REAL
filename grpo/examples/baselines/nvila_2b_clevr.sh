#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=path/to/pretrained/NVILA-Lite-2B-hf-preview-model-verlnew  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=BUAADreamer/clevr_count_70k@train \
    data.val_files=BUAADreamer/clevr_count_70k@test \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.vila_model=true \
    worker.reward.reward_function=./examples/reward_function/r1v_vila.py:compute_score \
    worker.vila_model=true \
    worker.actor.ulysses_size=2 \
    worker.actor.model.trust_remote_code=true \
    worker.rollout.trust_remote_code=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=nvila_lite_2b_clevr \
    trainer.n_gpus_per_node=8
