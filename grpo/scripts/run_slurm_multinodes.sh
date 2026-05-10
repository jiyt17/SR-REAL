#!/bin/bash

JOB_BASE_NAME="sr-real-r1"

JOB_SCRIPT="SR-REAL/grpo/run_r1_region.sh"
LOGS_DIR="SR-REAL/grpo/results/${JOB_BASE_NAME}"
WORKDIR="SR-REAL/grpo"

echo "$LOGS_DIR"

PARTITION="xxxxx"
TIME="4"
TIME_MIN="2.5"
ACCOUNT="your_slurm_account"
NUM_GPUS=8
NNODES=8

if [ ! -d "$LOGS_DIR" ]; then
    mkdir -p "$LOGS_DIR"
fi

submit_mnjob() {
    output=$(submit_job --account $ACCOUNT --nodes $NNODES --duration $TIME --duration_min $TIME_MIN \
    --logroot $LOGS_DIR \
    --image nvcr.io/nvidia/pytorch:23.09-py3 --mounts=path/to/home:/home,path/to/lustre:/lustre \
    --partition $PARTITION \
    --gpu $NUM_GPUS \
    -c 248 \
    --name $JOB_BASE_NAME \
    --workdir $WORKDIR \
    --command "set -e && echo 'Starting job execution...' && pip install ray --no-deps && pip install click filelock jsonschema msgpack packaging pyyaml requests aiohttp && echo 'Dependencies installed, starting ray_multinodes.sh...' && bash ray_multinodes.sh $JOB_SCRIPT; echo 'Command execution finished with exit code: $?'" 2>&1) # stdout and stderr
    echo "$output" | grep "Submitted batch job" | awk '{print $4}'
}


# Set the maximum number of trials
MAX_TRIALS=50
TRIAL_COUNT=0


# Loop to continuously submit the job
while [ $TRIAL_COUNT -lt $MAX_TRIALS ]; do
    TRIAL_COUNT=$((TRIAL_COUNT + 1))
    JOB_ID=$(submit_mnjob)
    echo "Submitted job $JOB_ID (Trial $TRIAL_COUNT/$MAX_TRIALS)"

    while squeue | grep -q "$JOB_ID"; do
        echo "Job $JOB_ID is still running..."
        sleep 60
    done

    echo "Job $JOB_ID has finished executing."

    if [ $TRIAL_COUNT -lt $MAX_TRIALS ]; then
        echo "Resubmitting the job for continuation..."
    else
        echo "Reached maximum number of trials ($MAX_TRIALS). Exiting loop."
    fi

done

echo "Script ended or exited loop due to termination condition."
