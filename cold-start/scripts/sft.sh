#!/bin/bash

. path/to/anaconda3/etc/profile.d/conda.sh
conda activate sr-3d
which python

cd path/to/cold-start

export VILA_DATASETS="cs-oci-ord"
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export NCCL_IB_TIMEOUT=31


DEFAULT_RUN_NAME="cold-start-sft"
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_ACCUMULATION_STEPS=16

STAGE_PATH=${1:-"path/to/pretrained/SR-3D"}
DATA_MIXTURE=${2:-"spar-singleimg-cot-ground+spar-singleimg-cot+spar-multiimg-cot-ground+spar-multiimg-cot+CA-cot-select+NS-cot-select+SL-global-ground2d+srgpt-ground2d+refcoco-train-float+ca1m-ground-base-v1+ca1m-ground-base-v1-center+omni3d-region-base-v1+omni3d-region-base-v1-center+spar-multiobj-ground-center+omninocs-region-base-v1+regiongpt-ft+spatialrgpt_ft+llava_1_5_mm_align"} 

DATA_MIXTURE_VAL=""
OUTPUT_DIR=${3:-"runs/train/$DEFAULT_RUN_NAME"}

source scripts/setups/train.sh

n_node=${SLURM_JOB_NUM_NODES:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=4

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --chat_template qwen2 \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --enable_depth True \
        --region_extractor regiongpt \
        --perception_encoder abs_mlp \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --tune_region_extractor True \
        --tune_perception_encoder True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 2 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --per_device_eval_batch_size 8 \
        --eval_steps 500 \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 30 \
        --learning_rate 5e-6 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to wandb

