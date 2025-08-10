# python -u playground/test_rollout_sync_with_dummy_train.py
import json
import logging
import os
import time

import ray
from omegaconf import DictConfig, OmegaConf

from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils import hf_tokenizer
from verl.workers.fsdp_workers import ActorRolloutRefWorker

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DEBUG_POST_MORTEM"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# os.environ["NCCL_DEBUG"] = "WARN"
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))


def init_actor_rollout_wg(config: DictConfig):
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout")
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    return actor_rollout_wg


def init_config() -> DictConfig:
    config = OmegaConf.load("/home/qinghao/workdir/verl-dev/verl/trainer/config/_generated_ppo_trainer.yaml")
    assert isinstance(config, DictConfig), "Expected DictConfig from OmegaConf.load"
    model_path = "/local/model/qwen2.5/Qwen2.5-1.5B-Instruct"  # -Instruct
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "sglang"
    config.actor_rollout_ref.rollout.mode = "sync"
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 16096
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.6
    config.actor_rollout_ref.rollout.temperature = 0.0
    config.actor_rollout_ref.actor.strategy = "fsdp2"
    config.actor_rollout_ref.actor.use_dynamic_bsz=True
    config.actor_rollout_ref.rollout.enable_drafter_train = True  # Enable early memory release
    config.actor_rollout_ref.rollout.min_workers_for_training = 1
    # config.trainer.rollout_data_dir = "/home/qinghao/workdir/verl-dev/playground/rollout_data"

    # test sleep/wake_up with fsdp offload
    # config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    # config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True


    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    return config, tokenizer

def dump_generations(inputs, outputs, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"debug.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
        }

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

# =========================== 1. Init rollout manager ===========================
# runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "RAY_DEBUG_POST_MORTEM": "0"}},

ray.init(
    num_cpus=96
)
# data_path = "/home/qinghao/workdir/verl-dev/playground/gen_batch_example.pkl"
data_path = "/home/qinghao/workdir/verl-dev/playground/debug_batches/gen_batch_example_infinite_qwen1_5B.pkl"
output_path = "/home/qinghao/workdir/verl-dev/playground/rollout_data"
gen_batch = DataProto.load_from_disk(data_path)

print("First prompt:", gen_batch.non_tensor_batch["raw_prompt"][0])

config, tokenizer = init_config()

actor_rollout_wg = init_actor_rollout_wg(config)

# test sleep and wake_up
# print("Testing sleep and wake_up...")
# actor_rollout_wg.sleep()
# actor_rollout_wg.wake_up()

start_time = time.time()
# Run the normal generation which will now use early memory release
print("\nStarting generation with early memory release...")
result = actor_rollout_wg.generate_sequences(prompts=gen_batch)
end_time = time.time()

print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")

inputs = tokenizer.batch_decode(result.batch["prompts"], skip_special_tokens=True)
outputs = tokenizer.batch_decode(result.batch["responses"], skip_special_tokens=True)

dump_generations(inputs, outputs, output_path)

ray.shutdown()