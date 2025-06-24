# python -u test_chat_scheduler_clean.py
# python -u tests/ray_gpu/test_high_level_scheduling_api.py
import os
import threading
import time

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from sglang.srt.server_args import ServerArgs

from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import (ActorRolloutRefWorker,
                                       AsyncActorRolloutRefWorker)
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.workers.rollout.elastic_scheduler import (ElasticConfig,
                                                    ElasticScheduler)
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

def init_actor_rollout_wg(config: DictConfig) -> AsyncLLMServerManager:
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Drafter: ray.remote(actor_rollout_cls),
        Role.Actor: ray.remote(actor_rollout_cls),  # Full 8 GPUs
    }
    global_pool_id = "global_pool"
    actor_pool_id = "actor_pool"
    drafter_pool_id = "drafter_pool"
    resource_pool_spec = {
        # global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        actor_pool_id: [6] * config.trainer.nnodes,
        drafter_pool_id: [2] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: actor_pool_id,
        Role.Drafter: drafter_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool(merge_into_global_pool=True, global_pool_id=global_pool_id)
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    print(
        f"resource_pool_manager.resource_pool_dict: {resource_pool_manager.resource_pool_dict}"
    )  # {'actor_pool': <verl.single_controller.ray.base.RayResourcePool object at 0x1551a2393f80>, 'drafter_pool': <verl.single_controller.ray.base.RayResourcePool object at 0x1551a2393ef0>, 'global_pool': <verl.single_controller.ray.base.RayResourcePool object at 0x155194997980>}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout")
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    print(
        f"resource_pool_to_cls: {resource_pool_to_cls}"
    )  # {<verl.single_controller.ray.base.RayResourcePool object at 0x1551a2393f80>: {'actor_rollout': <verl.single_controller.ray.base.RayClassWithInitArgs object at 0x1551a316fc20>}, <verl.single_controller.ray.base.RayResourcePool object at 0x1551a2393ef0>: {}, <verl.single_controller.ray.base.RayResourcePool object at 0x155194997980>: {}}

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
        break
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    return actor_rollout_wg


def init_config() -> DictConfig:
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/nobackup/model/qwen3/Qwen3-0.6B"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "sglang"
    config.actor_rollout_ref.rollout.mode = "sync"
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 256
    # config.actor_rollout_ref.rollout.multi_turn.format = "hermes"

    # # test sleep/wake_up with fsdp offload
    # config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    # config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
    return config



# =========================== 1. Init rollout manager ===========================
ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}, num_cpus=96)
data_path = "/home/qinghao/workdir/verl-dev/playground/gen_batch_example.pkl"
gen_batch = DataProto.load_from_disk(data_path)

config = init_config()

actor_rollout_wg = init_actor_rollout_wg(config)

if config.actor_rollout_ref.rollout.mode == "async":
    async_rollout_manager = AsyncLLMServerManager(
    config=config,
    worker_group=actor_rollout_wg,
    )

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()
    print("after sleep and wake_up")

    result = async_rollout_manager.generate_sequences(gen_batch)
else:
    async_rollout_manager = None

    result = actor_rollout_wg.generate_sequences(gen_batch)

print(f"after generate_sequences, result: {result}")
print("Test passed!")

ray.shutdown()
