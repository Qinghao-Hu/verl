# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import socket
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple, Type

import aiohttp
import fastapi
import ray
import uvicorn
from omegaconf import DictConfig
from starlette.requests import Request

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler

logger = logging.getLogger(__file__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class AsyncServerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print(f"FastAPI listen on {self.address}:{self.port}")
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])
        app.router.add_api_route("/metrics", self.get_metrics, methods=["GET"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> Tuple[str, int]:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    @abstractmethod
    async def chat_completion(self, raw_request: Request):
        """OpenAI chat completion API.

        API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        """Init async LLM engine."""
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        """Wake up engine to load model weights and build kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep engine to offload model weights and discard kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self):
        """Get server metrics including number of running requests."""
        raise NotImplementedError


class AsyncLLMServerManager:
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize AsyncLLMServerManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
        """
        self.full_config = config
        self.config = config.actor_rollout_ref
        self.worker_group = worker_group

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        server_class = async_server_class(
            rollout_backend=self.config.rollout.name,
        )

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        # Init user provided chat scheduler in sperate thread.
        self.chat_scheduler: ChatCompletionScheduler = None
        self.chat_scheduler_exception: Exception = None
        self.chat_scheduler_loop = None
        self.chat_scheduler_ready = threading.Event()
        self.chat_scheduler_thread = threading.Thread(target=self._init_chat_scheduler, daemon=True)
        self.chat_scheduler_thread.start()
        self.chat_scheduler_ready.wait()
        
        print(f"worker_group: {self.worker_group}") # <verl.single_controller.ray.base.RayWorkerGroup object at 0x152560e39850>
        print(f"register_center: {register_center}") # Actor(WorkerGroupRegisterCenter, cab177916ede06d6fde8476f01000000)
        print(f"workers_info: {workers_info}") #{0: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 3: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 5: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 4: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 1: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 7: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 2: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee', 6: '93270264da6dfe71e571d779614d22c69d074d3436fa0d0bb725faee'}
        print(f"server_addresses: {self.server_addresses}") # ['10.1.200.5:57957', '10.1.200.5:38321', '10.1.200.5:48591', '10.1.200.5:57225']
        print(f"async_llm_servers: {self.async_llm_servers}") #[Actor(AsyncSglangServer, a851267c37553807be135d1d01000000), Actor(AsyncSglangServer, 002f250afe4bccc492492f0601000000), Actor(AsyncSglangServer, 268d0368a022e6cdad1e87ec01000000), Actor(AsyncSglangServer, fdb0aa69952f5f0c64aac31601000000)]

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)

        try:
            self.chat_scheduler = ChatCompletionScheduler(
                config=self.full_config,
                server_addresses=self.server_addresses,
            )
        except Exception as e:
            logger.exception(f"chat_scheduler init error: {e}")
            self.chat_scheduler_exception = e
        finally:
            self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()

    def wake_up(self):
        """Wake up all vllm instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def submit_chat_completions(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Dict[str, Any],
    ):
        """Submit a chat completion request to chat scheduler and wait until it is done.
        To submit multiple requests in parallel, please use `generate_sequences` instead.

        Args: same as ChatCompletionScheduler.submit_chat_completions.
        """
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler._submit_chat_completions_semaphore(
                messages=messages,
                request_id=None,
                sampling_params=sampling_params,
            ),
            self.chat_scheduler_loop,
        )
        future.result()

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."

        future = asyncio.run_coroutine_threadsafe(self.chat_scheduler.generate_sequences(prompts, **sampling_params), self.chat_scheduler_loop)
        return future.result()

    def get_running_requests_count(self, server_url: str = None) -> float:
        """Synchronously get the number of running requests from a specific server or all servers.
        
        Args:
            server_url: Optional. If provided, get requests from specific server. 
                       If None, get total requests from all servers.
                       
        Returns:
            float: Number of running requests.
        """
        if server_url is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.get_num_running_requests(server_url), 
                self.chat_scheduler_loop
            )
            return future.result()
        else:
            future = asyncio.run_coroutine_threadsafe(
                self.get_total_running_requests(), 
                self.chat_scheduler_loop
            )
            return future.result()

    def get_all_server_requests_count(self) -> Dict[str, float]:
        """Get the number of running requests from each server.
        
        Returns:
            Dict[str, float]: Mapping of server address to number of running requests.
        """
        async def _get_all_counts():
            results = {}
            for server_address in self.server_addresses:
                try:
                    count = await self.get_num_running_requests(server_address)
                    results[server_address] = count
                except Exception as e:
                    logger.warning(f"Failed to get metrics from {server_address}: {e}")
                    results[server_address] = 0.0
            return results
        
        future = asyncio.run_coroutine_threadsafe(_get_all_counts(), self.chat_scheduler_loop)
        return future.result()

    async def get_num_running_requests(self, server_url: str):
        """Get the number of running requests from a specific server."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=60,
            ),
        ) as session:
            async with session.get(f"http://{server_url}/metrics") as response:
                response.raise_for_status()
                text = await response.text()
                for line in text.split("\n"):
                    if line.startswith("sglang:num_running_reqs"):
                        return float(line.split(" ")[1])
        raise RuntimeError(
            f"Failed to get num running requests metrics from {server_url}"
        )

    async def get_total_running_requests(self):
        """Get the total number of running requests across all servers."""
        total_requests = 0
        for server_address in self.server_addresses:
            try:
                num_requests = await self.get_num_running_requests(server_address)
                total_requests += num_requests
            except Exception as e:
                logger.warning(f"Failed to get metrics from {server_address}: {e}")
        return total_requests


def async_server_class(rollout_backend: str) -> Type[AsyncServerBase]:
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend, should be "vllm" or "sglang".

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    if rollout_backend == "vllm":
        from verl.workers.rollout.vllm_rollout.vllm_async_server import \
            AsyncvLLMServer

        return AsyncvLLMServer
    elif rollout_backend == "sglang":
        from verl.workers.rollout.sglang_rollout.async_sglang_server import \
            AsyncSglangServer

        return AsyncSglangServer
    else:
        raise NotImplementedError
