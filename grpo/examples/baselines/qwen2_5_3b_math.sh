set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
MODEL_PATH=path/to/model/Qwen2.5-3B-Instruct  # replace it with your local file path


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.ulysses_size=1 \
    data.val_batch_size=32 \
    data.rollout_batch_size=32 \
    worker.actor.global_batch_size=32 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_3b_math_grpo \
    trainer.save_freq=100 \
    trainer.save_model_only=true \
    trainer.project_name=longrl_llm_test