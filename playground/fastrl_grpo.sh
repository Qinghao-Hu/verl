export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export WANDB_API_KEY='be32ec2a18acdc347b5d3029742c0ef1090a9e1e'
export NUMEXPR_MAX_THREADS=160
export MKL_SERVICE_FORCE_INTEL=1

PROJECT_NAME=Qwen3-RL
EXPERIMENT_NAME=debug-split

# DATA_PATH='/nobackup/qinghao/dataset/reasoning/DAPO-Math-17k'
DATA_PATH=/nobackup/qinghao/dataset/reasoning/gsm8k
# SFT_MODEL_PATH=/nobackup/model/qwen2.5/Qwen2.5-Math-7B
SFT_MODEL_PATH=/local/model/qwen3/Qwen3-0.6B
CKPT_PATH='/nobackup/qinghao/runs/reasoning'


# data.train_files=$DATA_PATH/dapo-math-17k.parquet \
# data.val_files=$DATA_PATH/aime-2024.parquet \

rollout_mode="sync"
rollout_name="sglang"

python3 -m verl.trainer.main_fastrl \
    ray_init.num_cpus=96 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
    

# srun -J grpo -N 1 --exclusive bash playground/fastrl_grpo.sh