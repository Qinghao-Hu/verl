# 文件名: test_dynamic_resource_allocation.py
import os

# 确保在所有其他库导入之前设置MKL环境变量
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import asyncio
import logging
import time
from typing import Any, Dict, List

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from ray.util import remove_placement_group

# 导入真实的verl组件
from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls_fused
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.async_server import AsyncLLMServerManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 其他环境设置
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")

def init_config() -> DictConfig:
    logger.info("Initializing configuration...")
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/nobackup/model/qwen3/Qwen3-0.6B"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "sglang"
    config.actor_rollout_ref.rollout.mode = "async"
    logger.info(f"Configuration loaded. Model path: {config.actor_rollout_ref.model.path}")
    return config

async def wait_for_gpu_resources(expected_gpus: int, timeout: int = 60):
    logger.info(f"Waiting for {expected_gpus} GPUs to become available...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        available_gpus = ray.available_resources().get("GPU", 0)
        if available_gpus >= expected_gpus:
            logger.info(f"Success! {available_gpus} GPUs are now available.")
            return
        logger.info(f"  ... still waiting, available GPUs: {available_gpus}/{expected_gpus}")
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for {expected_gpus} GPUs to become available.")

class ElasticActorDrafterScheduler:
    def __init__(self, config: DictConfig, total_gpus: int = 8):
        logger.info("--- ElasticActorDrafterScheduler Initializing ---")
        self.config = config
        self.total_gpus = total_gpus
        self.group_blueprints: Dict[str, Dict[str, Any]] = self._prepare_blueprints()
        self.active_managers: Dict[str, AsyncLLMServerManager] = {}
        self.current_mode = None
        
    async def initialize(self):
        await self.scale_up()

    def _prepare_blueprints(self) -> Dict[str, Dict[str, Any]]:
        blueprints = {}
        pool_specs = {
            "actor_large": [self.total_gpus] * self.config.trainer.nnodes,
            "actor_small": [self.total_gpus - 2] * self.config.trainer.nnodes,
            "drafter": [2] * self.config.trainer.nnodes,
        }
        
        actor_cls_def = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=self.config.actor_rollout_ref, role="actor_rollout")
        drafter_config = self.config.get("drafter", self.config.actor_rollout_ref)
        drafter_cls_def = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=drafter_config, role="actor")
        
        group_definitions = { "actor_large": ("actor_large", actor_cls_def), "actor_small": ("actor_small", actor_cls_def), "drafter": ("drafter", drafter_cls_def) }
        for name, (pool_name, cls_def) in group_definitions.items():
            class_dict = {"worker": cls_def}
            worker_dict_cls = create_colocated_worker_cls_fused(class_dict=class_dict)
            blueprints[name] = { "process_on_nodes": pool_specs[pool_name], "worker_dict_cls": worker_dict_cls }
        
        logger.info("Scheduler blueprints are ready.")
        return blueprints
    
    async def _wakeup_group(self, name: str):
        if name in self.active_managers: return

        logger.info(f"--- Waking up group '{name}'... ---")
        blueprint = self.group_blueprints[name]
        resource_pool = RayResourcePool(process_on_nodes=blueprint["process_on_nodes"], use_gpu=True, name_prefix=f"{name}_pool")
        
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=blueprint["worker_dict_cls"])
        
        logger.info(f"Blocking until all workers in group '{name}' are alive...")
        wg._block_until_all_workers_alive()
        logger.info(f"All workers in group '{name}' are alive.")
        
        wg.resource_pool = resource_pool
        
        manager = AsyncLLMServerManager(config=self.config, worker_group=wg)
        
        logger.info(f"Waking up underlying inference engine for group '{name}' via manager...")
        await manager.wake_up()
        
        self.active_managers[name] = manager
        logger.info(f"--- Group '{name}' is now active with {wg.world_size} workers. ---")

    async def _sleep_group(self, name: str) -> bool:
        if name not in self.active_managers: return False
        logger.info(f"--- Putting group '{name}' to sleep... ---")
        manager = self.active_managers.pop(name)
        wg = manager.worker_group
        
        logger.info(f"Asynchronously sleeping inference engine for group '{name}' via manager...")
        try:
            await asyncio.wait_for(manager.sleep(), timeout=15.0)
        except Exception as e:
            logger.warning(f"Error or timeout during manager.sleep() for group '{name}'. Killing actors anyway. Error: {e}")
            
        if hasattr(wg, "_workers"):
            for worker_handle in wg._workers:
                logger.info(f"Killing worker: {worker_handle}")
                ray.kill(worker_handle, no_restart=True)
            wg._workers.clear()
        
        if hasattr(wg, "resource_pool") and hasattr(wg.resource_pool, "get_placement_groups"):
            for pg in wg.resource_pool.get_placement_groups():
                if pg:
                    logger.info(f"Removing placement group: {pg.id}")
                    remove_placement_group(pg)
        logger.info(f"--- Group '{name}' is now asleep. Resources released. ---")
        return True

    async def scale_down(self):
        if self.current_mode == "scaled_down": return
        logger.info("#################### SCALING DOWN ####################")
        if await self._sleep_group("actor_large"):
            await wait_for_gpu_resources(self.total_gpus)
        await self._wakeup_group("actor_small")
        await self._wakeup_group("drafter")
        self.current_mode = "scaled_down"
        logger.info("#################### SCALED DOWN #####################")

    async def scale_up(self):
        if self.current_mode == "scaled_up": return
        logger.info("#################### SCALING UP ######################")
        did_sleep1 = await self._sleep_group("actor_small")
        did_sleep2 = await self._sleep_group("drafter")
        if did_sleep1 or did_sleep2:
            await wait_for_gpu_resources(self.total_gpus)
        await self._wakeup_group("actor_large")
        self.current_mode = "scaled_up"
        logger.info("#################### SCALED UP #######################")

    async def _dispatch_chat_requests(self, manager: AsyncLLMServerManager, prompts: List[List[Dict]]):
        tasks = []
        for prompt_messages in prompts:
            json_request = { "model": "worker", "messages": prompt_messages, "temperature": 0.0 }
            task = manager.chat_completion(json_request)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def generate(self, data: DataProto):
        active_actor_group_name = "actor_large" if self.current_mode == "scaled_up" else "actor_small"
        active_manager = self.active_managers.get(active_actor_group_name)
        if not active_manager: raise RuntimeError(f"Actor manager '{active_actor_group_name}' not active!")
        logger.info(f"Routing generation request to '{active_actor_group_name}' ({self.current_mode} mode)...")
        return await self._dispatch_chat_requests(active_manager, data.non_tensor_batch["raw_prompt"])

    async def train_drafter(self, data: DataProto):
        drafter_manager = self.active_managers.get("drafter")
        if not drafter_manager: raise RuntimeError("Drafter manager is not active. Cannot train drafter.")
        logger.info("Routing training request to 'drafter' group...")
        return await self._dispatch_chat_requests(drafter_manager, data.non_tensor_batch["raw_prompt"])

async def main():
    if ray.is_initialized(): ray.shutdown()
    ray.init(num_gpus=8)
    
    logger.info("==================================================")
    logger.info("  Dynamic Actor-Drafter Scheduling Demonstration")
    logger.info("==================================================")

    config = init_config()
    test_prompts = [[{"role": "user", "content": "What is the capital of France?"}],[{"role": "user", "content": "Write a short poem about Ray AI."}],]
    test_batch = DataProto(non_tensor_batch={"raw_prompt": np.array(test_prompts, dtype=object)})

    scheduler = None
    try:
        scheduler = ElasticActorDrafterScheduler(config=config, total_gpus=8)
        await scheduler.initialize()
        await asyncio.sleep(1)

        logger.info("\n--- STEP 1: Simulating HIGH load scenario (Initial State) ---")
        high_load_results = await scheduler.generate(test_batch)
        logger.info(f"Result from high load (x{len(high_load_results)}): {high_load_results[0] if high_load_results else 'N/A'}\n")
        
        try:
            await scheduler.train_drafter(test_batch)
        except Exception as e:
            logger.info(f"As expected, failed to train drafter in high-load mode: {e}\n")
        await asyncio.sleep(1)

        logger.info("\n--- STEP 2: Load is low (num_requests < 16), scaling down... ---")
        await scheduler.scale_down()
        await asyncio.sleep(1)

        logger.info("\n--- STEP 3: Simulating LOW load scenario ---")
        low_load_results = await scheduler.generate(test_batch)
        logger.info(f"Result from low load (x{len(low_load_results)}): {low_load_results[0] if low_load_results else 'N/A'}\n")
        
        drafter_result = await scheduler.train_drafter(test_batch)
        logger.info(f"Drafter training result (x{len(drafter_result)}): {drafter_result[0] if drafter_result else 'N/A'}\n")
        await asyncio.sleep(1)

        logger.info("\n--- STEP 4: Load is high again, scaling up... ---")
        await scheduler.scale_up()
        await asyncio.sleep(1)

        logger.info("\n--- STEP 5: Simulating HIGH load scenario again ---")
        high_load_results_2 = await scheduler.generate(test_batch)
        logger.info(f"Result from second high load (x{len(high_load_results_2)}): {high_load_results_2[0] if high_load_results_2 else 'N/A'}\n")

        logger.info("==================================================")
        logger.info("      Demonstration Finished Successfully!")
        logger.info("==================================================")
    except Exception as e:
        logger.error("An error occurred during the demonstration.", exc_info=True)
    finally:
        if scheduler:
            logger.info("Cleaning up all active groups...")
            await scheduler._sleep_group("actor_large")
            await scheduler._sleep_group("actor_small")
            await scheduler._sleep_group("drafter")
        
        logger.info("Shutting down Ray...")
        ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())