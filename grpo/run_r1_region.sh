#!/bin/bash

set -x

cd path/to/svila-verl
. path/to/anaconda3/etc/profile.d/conda.sh
conda activate verl
which python

export PYTHONUNBUFFERED=1

MODEL_PATH=path/to/cold-start-sft-hf

model_name=$(basename "$MODEL_PATH")
target_dir="/root/.cache/huggingface/modules/transformers_modules"
target_path="$target_dir/$model_name"
mkdir -p "$target_path"
cp "$MODEL_PATH"/mm_utils.py "$target_path"


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=path/to/data/SR1/r1_data.json \
    data.val_files=path/to/data/SR1/r1_data_val.json \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.vila_model=true \
    data.rollout_batch_size=512 \
    algorithm.online_filtering=true \
    algorithm.online_buffer=false \
    data.val_batch_size=128 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.reward.reward_function=./examples/reward_function/r1v_vila.py:compute_score \
    worker.vila_model=true \
    worker.actor.ulysses_size=1 \
    worker.actor.model.trust_remote_code=true \
    worker.rollout.trust_remote_code=true \
    worker.rollout.num_chunk_seq=60 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=sr-real-r1 \
    trainer.save_freq=1 \
    trainer.nnodes=8 \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=path/to/results/sr-real-r1 \
    trainer.load_checkpoint_path=path/to/results/sr-real-r1
