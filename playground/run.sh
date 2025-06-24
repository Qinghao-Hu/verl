#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=Qwen3-8B

set -e

export WANDB_API_KEY='be32ec2a18acdc347b5d3029742c0ef1090a9e1e'
export NUMEXPR_MAX_THREADS=160
export MKL_SERVICE_FORCE_INTEL=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export WANDB_MODE=offline
# export VLLM_ATTENTION_BACKEND="XFORMERS"

PROJECT_NAME='Qwen3-RL'
# EXPERIMENT_NAME='Qwen2.5-32B-Eurus-Data-mini-A100'
EXPERIMENT_NAME='debug'

# DATA_PATH='/nobackup/qinghao/dataset/reasoning/Eurus-2-RL-Data'
DATA_PATH=/nobackup/qinghao/dataset/reasoning/gsm8k
# DATA_PATH=/home/qinghao/workdir/verl/data/gsm8k
# SFT_MODEL_PATH=/nobackup/model/llama3.1/Llama-3.1-8B-Instruct
SFT_MODEL_PATH=/nobackup/model/qwen3/Qwen3-8B
# SFT_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

RAY_VERSION=$(ray --version)
echo $RAY_VERSION

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo $nodes
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi


port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &

# optional, though may be useful in certain versions of Ray.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
         --gpus-per-task=4 \
        ray start --address "$ip_head" --temp-dir=$HOME/ray \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done


# __doc_script_start__
echo "End starting"
# sleep infinity

echo $SLURM_JOB_NUM_NODES

# RAY_ADDRESS="http://dgx-01:6379"
WORKING_DIR=${WORKING_DIR:-"${PWD}"}

# source ~/.bashrc
# source /home/jerguo/miniconda3/bin/activate fastrl-verl-baseline
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
RAY_ADDRESS="http://$head_node_ip:8265"

# Note: In each step, the total_response_number = data.train_batch_size * actor_rollout_ref.rollout.n
PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node"  \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.total_epochs=1 $@ 2>&1| tee logs/${PROJECT_NAME}_${EXPERIMENT_NAME}_train_$(date +%Y%m%d_%H%M%S).log 


