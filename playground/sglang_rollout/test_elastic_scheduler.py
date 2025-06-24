"""
Test script for elastic scheduler with SGLang.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=8 playground/sglang_rollout/test_elastic_scheduler.py
"""

import asyncio
import time
from typing import Any, Dict

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion

from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.rollout.async_server import AsyncLLMServerManager


async def simulate_load_pattern(manager: AsyncLLMServerManager, num_initial_requests: int = 1024):
    """Simulate a load pattern that triggers elastic scaling."""
    
    # Callback to handle responses
    completed_requests = []
    async def handle_response(completion: ChatCompletion, info: Dict[str, Any], exception: Exception):
        if exception:
            print(f"Request {info['request_id']} failed: {exception}")
        else:
            completed_requests.append(info['request_id'])
    
    # Phase 1: High load - submit many requests
    print(f"Phase 1: Submitting {num_initial_requests} requests...")
    tasks = []
    for i in range(num_initial_requests):
        request = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": f"Hello, this is request {i}"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        
        # Submit request asynchronously
        task = asyncio.create_task(
            manager.chat_scheduler.submit_chat_completions(
                callback=handle_response,
                callback_additional_info={"request_id": i},
                **request
            )
        )
        tasks.append(task)
        
        # Add small delay to avoid overwhelming the system
        if i % 100 == 0:
            await asyncio.sleep(0.1)
    
    print("Waiting for initial burst to process...")
    await asyncio.sleep(30)  # Let the system handle the initial load
    
    # Phase 2: Low load - requests gradually complete
    print("Phase 2: Load decreasing, elastic scaling should kick in...")
    
    # Monitor the elastic scaling
    for i in range(10):
        active_servers = len(manager.chat_scheduler.active_servers)
        sleeping_servers = len(manager.chat_scheduler.sleeping_servers)
        print(f"Time {i*5}s: Active servers: {active_servers}, Sleeping servers: {sleeping_servers}")
        await asyncio.sleep(5)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Completed {len(completed_requests)} requests")


def main():
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create configuration
    config = OmegaConf.create({
        "model": {
            "path": "/nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B",
        },
        "rollout": {
            "name": "sglang",
            "tensor_model_parallel_size": 2,
            "chat_scheduler": "verl.workers.rollout.elastic_scheduler.ElasticChatCompletionScheduler",
        }
    })
    
    # Create a mock worker group for testing
    # In real usage, this would be created by the RL training framework
    class MockWorkerGroup:
        def __init__(self, world_size):
            self.world_size = world_size
            self.name_prefix = "test_elastic"
    
    worker_group = MockWorkerGroup(world_size=8)  # 8 GPUs total
    
    # Additional scheduler kwargs for elastic scaling
    scheduler_kwargs = {
        "low_threshold": 8,       # Scale down when all workers have < 8 requests
        "high_threshold": 16,     # Scale up when any worker has > 16 requests
        "monitor_interval": 1.0,  # Check every second
        "scale_cooldown": 10.0,   # Wait 10 seconds between scaling operations
    }
    
    # Create the AsyncLLMServerManager with elastic scheduler
    print("Creating AsyncLLMServerManager with elastic scheduler...")
    manager = AsyncLLMServerManager(
        config=config,
        worker_group=worker_group,
        scheduler_kwargs=scheduler_kwargs
    )
    
    print(f"Initial state: {manager.rollout_dp_size} dp workers (tp_size={manager.rollout_tp_size})")
    
    # Start the elastic monitoring
    asyncio.run(manager.chat_scheduler.start_monitoring())
    
    # Run the load simulation
    asyncio.run(simulate_load_pattern(manager))
    
    # Stop monitoring
    asyncio.run(manager.chat_scheduler.stop_monitoring())
    
    print("Test completed!")


if __name__ == "__main__":
    main() 