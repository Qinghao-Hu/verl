# 文件名: test_elastic_actor_drafter.py
import os
import time
from typing import Any, Dict, List

import ray
import torch
from omegaconf import DictConfig, OmegaConf
# <--- 变化点 1: 导入 remove_placement_group
from ray.util import remove_placement_group

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls_fused

# =====================================================================================
# 1. 导入和模拟 (Mock) verl 框架中的核心类
# =====================================================================================


print("Successfully imported real 'verl' components.")


class MockDataProto:
    def __init__(self, text):
        self.text = text
    def __repr__(self):
        return f"MockDataProto(text='{self.text}')"

@ray.remote(num_cpus=1, num_gpus=1)
class MockGenericWorker(Worker):
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        self.role = role
        self.model = None
        self.actor_name = ray.get_runtime_context().get_actor_name()
        print(f"[{self.actor_name}] Worker of role '{self.role}' has been created.")

    def init_model(self):
        print(f"[{self.actor_name}] Role '{self.role}': Initializing model...")
        time.sleep(0.5)
        self.model = f"MockModel_{self.role}_Loaded"
        print(f"[{self.actor_name}] Role '{self.role}': Model initialized.")
        return True

    def generate_sequences(self, data: MockDataProto):
        print(f"[{self.actor_name}] Role '{self.role}': Received data: {data.text}. Generating sequence...")
        return f"Response from {self.actor_name} ({self.role}) for '{data.text}'"

    def train_drafter_step(self, data: MockDataProto):
        print(f"[{self.actor_name}] Role '{self.role}': Received data: {data.text}. Training drafter...")
        time.sleep(0.2)
        return f"Drafter trained by {self.actor_name} with '{data.text}'"


def init_mock_config() -> DictConfig:
    return OmegaConf.create({
        'trainer': {'nnodes': 1},
        'actor_rollout_ref': {},
        'drafter': {}
    })

def wait_for_gpu_resources(expected_gpus: int, timeout: int = 30):
    print(f"\nWaiting for {expected_gpus} GPUs to become available...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        available_gpus = ray.available_resources().get("GPU", 0)
        if available_gpus >= expected_gpus:
            print(f"Success! {available_gpus} GPUs are now available.")
            return
        print(f"  ... still waiting, available GPUs: {available_gpus}/{expected_gpus}")
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {expected_gpus} GPUs to become available.")

# =====================================================================================
# 2. 带有 Sleep / Wakeup 功能的弹性调度器 (最终修正版)
# =====================================================================================

class ElasticActorDrafterScheduler:
    def __init__(self, config: DictConfig, total_gpus: int = 8):
        print("--- ElasticActorDrafterScheduler Initializing ---")
        self.config = config
        self.total_gpus = total_gpus
        self.group_blueprints: Dict[str, Dict[str, Any]] = self._prepare_blueprints()
        self.active_groups: Dict[str, RayWorkerGroup] = {}
        self.current_mode = None
        self.scale_up()

    def _prepare_blueprints(self) -> Dict[str, Dict[str, Any]]:
        blueprints = {}
        pool_specs = {
            "actor_large": [self.total_gpus] * self.config.trainer.nnodes,
            "actor_small": [self.total_gpus - 2] * self.config.trainer.nnodes,
            "drafter": [2] * self.config.trainer.nnodes,
        }
        
        actor_cls_def = RayClassWithInitArgs(cls=MockGenericWorker, config=self.config.actor_rollout_ref, role="ActorRollout")
        drafter_cls_def = RayClassWithInitArgs(cls=MockGenericWorker, config=self.config.drafter, role="Drafter")
        
        group_definitions = {
            "actor_large": ("actor_large", actor_cls_def),
            "actor_small": ("actor_small", actor_cls_def),
            "drafter": ("drafter", drafter_cls_def),
        }

        for name, (pool_name, cls_def) in group_definitions.items():
            class_dict = {"worker": cls_def}
            worker_dict_cls = create_colocated_worker_cls_fused(class_dict=class_dict)
            blueprints[name] = {
                "process_on_nodes": pool_specs[pool_name],
                "worker_dict_cls": worker_dict_cls,
            }
        
        print("Scheduler blueprints are ready.")
        return blueprints
    
    def _wakeup_group(self, name: str):
        if name in self.active_groups:
            return

        print(f"\n--- Waking up group '{name}'... ---")
        blueprint = self.group_blueprints[name]
        resource_pool = RayResourcePool(
            process_on_nodes=blueprint["process_on_nodes"], use_gpu=True, name_prefix=f"{name}_pool"
        )
        wg = RayWorkerGroup(
            resource_pool=resource_pool, ray_cls_with_init=blueprint["worker_dict_cls"],
        )
        
        wg.sub_cls_name = "worker"
        # <--- 变化点 2: 将 resource_pool 附加到 wg 实例上，以便之后销毁它
        wg.resource_pool = resource_pool
        
        wg.execute_all_sync("init_model")
        
        self.active_groups[name] = wg
        print(f"--- Group '{name}' is now active with {wg.world_size} workers. ---")

    def _sleep_group(self, name: str):
        if name not in self.active_groups:
            return False

        print(f"\n--- Putting group '{name}' to sleep... ---")
        wg = self.active_groups.pop(name)
        
        if hasattr(wg, "_workers"):
            for worker_handle in wg._workers:
                print(f"Killing worker: {worker_handle}")
                ray.kill(worker_handle)
            wg._workers.clear()
        
        # <--- 变化点 3: 销毁预留资源的 Placement Group
        if hasattr(wg, "resource_pool") and hasattr(wg.resource_pool, "get_placement_groups"):
            pgs = wg.resource_pool.get_placement_groups()
            for pg in pgs:
                if pg:
                    print(f"Removing placement group: {pg.id}")
                    remove_placement_group(pg)

        print(f"--- Group '{name}' is now asleep. Resources should be released. ---")
        return True

    def scale_down(self):
        if self.current_mode == "scaled_down": return
        print("\n" + "#"*20 + " SCALING DOWN " + "#"*20)
        
        if self._sleep_group("actor_large"):
            wait_for_gpu_resources(self.total_gpus)

        self._wakeup_group("actor_small")
        self._wakeup_group("drafter")
        self.current_mode = "scaled_down"
        print("#"*20 + " SCALED DOWN " + "#"*21)

    def scale_up(self):
        if self.current_mode == "scaled_up": return
        print("\n" + "#"*20 + " SCALING UP " + "#"*22)

        # 两个组都需要休眠，我们只需要等待一次
        did_sleep1 = self._sleep_group("actor_small")
        did_sleep2 = self._sleep_group("drafter")
        if did_sleep1 or did_sleep2:
            wait_for_gpu_resources(self.total_gpus)

        self._wakeup_group("actor_large")
        self.current_mode = "scaled_up"
        print("#"*20 + " SCALED UP " + "#"*23)

    def generate(self, data: MockDataProto):
        active_actor_group_name = "actor_large" if self.current_mode == "scaled_up" else "actor_small"
        active_actor_group = self.active_groups.get(active_actor_group_name)
        if not active_actor_group:
            raise RuntimeError(f"Actor group '{active_actor_group_name}' not active!")
        
        print(f"\n>>> Routing generation request to '{active_actor_group_name}' ({self.current_mode} mode)...")
        results = active_actor_group.execute_all_sync("generate_sequences", data)
        return results

    def train_drafter(self, data: MockDataProto):
        drafter_group = self.active_groups.get("drafter")
        if not drafter_group:
            raise RuntimeError("Drafter group is not active. Cannot train drafter.")
        
        print("\n>>> Routing training request to 'drafter' group...")
        results = drafter_group.execute_all_sync("train_drafter_step", data)
        return results

# =====================================================================================
# 3. 主执行逻辑
# =====================================================================================

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if ray.is_initialized(): ray.shutdown()
    ray.init(num_gpus=8)
    
    print("\n" + "="*50)
    print("  Dynamic Actor-Drafter Scheduling Demonstration")
    print("="*50 + "\n")

    mock_config = init_mock_config()
    mock_batch = MockDataProto(text="Analyze this market trend.")

    scheduler = ElasticActorDrafterScheduler(config=mock_config, total_gpus=8)
    time.sleep(1)

    print("\n--- STEP 1: Simulating HIGH load scenario (Initial State) ---")
    high_load_results = scheduler.generate(mock_batch)
    print(f"\n[Main] Result from high load: {high_load_results}\n")
    
    try:
        scheduler.train_drafter(mock_batch)
    except Exception as e:
        print(f"\n[Main] As expected, failed to train drafter in high-load mode: {e}\n")
    time.sleep(1)

    print("\n--- STEP 2: Load is low, scaling down... ---")
    scheduler.scale_down()
    time.sleep(1)

    print("\n--- STEP 3: Simulating LOW load scenario ---")
    low_load_results = scheduler.generate(mock_batch)
    print(f"\n[Main] Result from low load: {low_load_results}\n")
    
    drafter_result = scheduler.train_drafter(mock_batch)
    print(f"\n[Main] Drafter training result: {drafter_result}\n")
    time.sleep(1)

    print("\n--- STEP 4: Load is high again, scaling up... ---")
    scheduler.scale_up()
    time.sleep(1)

    print("\n--- STEP 5: Simulating HIGH load scenario again ---")
    high_load_results_2 = scheduler.generate(mock_batch)
    print(f"\n[Main] Result from second high load: {high_load_results_2}\n")

    print("\n" + "="*50)
    print("      Demonstration Finished Successfully!")
    print("="*50 + "\n")
    
    ray.shutdown()