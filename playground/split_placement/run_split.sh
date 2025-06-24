export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export WANDB_API_KEY='be32ec2a18acdc347b5d3029742c0ef1090a9e1e'
export NUMEXPR_MAX_THREADS=160
export MKL_SERVICE_FORCE_INTEL=1

PROJECT_NAME='Qwen3-RL'
EXPERIMENT_NAME='debug-split'

DATA_PATH=/nobackup/qinghao/dataset/reasoning/gsm8k
SFT_MODEL_PATH=/nobackup/model/qwen3/Qwen3-8B

python3 main_ppo_split.py \
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.path=$SFT_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=15 $@ 2>&1| tee logs/${PROJECT_NAME}_${EXPERIMENT_NAME}_train_$(date +%Y%m%d_%H%M%S).log 

# srun -J grpo -N 1 --exclusive bash playground/split_placement/run_split.sh