export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export WANDB_API_KEY='be32ec2a18acdc347b5d3029742c0ef1090a9e1e'
export NUMEXPR_MAX_THREADS=160
export MKL_SERVICE_FORCE_INTEL=1

PROJECT_NAME='Qwen3-RL'
# EXPERIMENT_NAME='Qwen-7B-Math-DAPO-Data'
# EXPERIMENT_NAME='Llama-8B-Math-Eurus-Data'
# EXPERIMENT_NAME='Qwen-7B-Math-Eurus-Data'
# EXPERIMENT_NAME='DS-Qwen-7B-Eurus-Data'
EXPERIMENT_NAME='Qwen-7B-Instruct-Eurus-Data'

# DATA_PATH='/nobackup/qinghao/dataset/reasoning/Bespoke-Stratos-17k'
# DATA_PATH='/nobackup/qinghao/dataset/reasoning/DAPO-Math-17k' # dapo-math-17k-dedup.parquet
DATA_PATH='/nobackup/qinghao/dataset/reasoning/Eurus-2-RL-Data'
# SFT_MODEL_PATH=/nobackup/qinghao/huggingface/OpenR1-Qwen-7B
# SFT_MODEL_PATH=/nobackup/model/llama3.1/Llama-3.1-8B-Instruct
# SFT_MODEL_PATH=/nobackup/model/qwen2.5/Qwen2.5-7B
# SFT_MODEL_PATH=/nobackup/model/qwen2.5/Qwen2.5-Math-7B
# SFT_MODEL_PATH=/nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Qwen-7B
SFT_MODEL_PATH=/nobackup/model/qwen2.5/Qwen2.5-7B-Instruct
CKPT_PATH='/nobackup/qinghao/runs/reasoning'

# Note: In each step, the total_response_number = data.train_batch_size * actor_rollout_ref.rollout.n
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m fastrl.trainer.main_fastrl \
    config=playground/grpo_config.yaml \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/validation.parquet \
    data.max_prompt_length=2048 \
    data.max_response_length=30000 \
    data.train_batch_size=32 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.save_freq=1


# srun -J eagle_loop -N 1 --exclusive torchrun --standalone --nnodes=1 --nproc_per_node=8 -m fastrl.trainer.eagle_trainer
