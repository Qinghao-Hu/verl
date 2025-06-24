# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class AsyncSglangServer(AsyncServerBase):
    def __init__(self, config: DictConfig, dp_size: int, dp_rank: int, wg_prefix: str):
        super().__init__()
        self.config = config.actor_rollout_ref
        self._tp_size = self.config.rollout.get("tensor_model_parallel_size", 1)
        self._dp_size = dp_size
        self._dp_rank = dp_rank
        self.wg_prefix = wg_prefix
        self.workers = []
        self.master_worker = None

        print(f"IN __init__ {config=}")
        print(f"{self.wg_prefix=}")  # self.wg_prefix='bb67tI'
        print(f"{self._dp_size=}")  # self._dp_size=4
        print(f"{self._dp_rank=}")  # self._dp_rank=0      /    self._dp_rank=1
        print(f"{self._tp_size=}")  # self._tp_size=2
        print(f"{self.workers=}")  # self.workers=[]

    async def init_engine(self):
        if self.workers:
            # avoid init twice
            return
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        matched_actors = [actor for actor in all_actors if actor.get("name", None).startswith(self.wg_prefix + "WorkerDict_")]

        print(
            f"{matched_actors=}"
        )  # matched_actors=[{'name': 'bb67tIWorkerDict_0:7', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:2', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:0', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:1', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:3', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:5', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:6', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}, {'name': 'bb67tIWorkerDict_0:4', 'namespace': 'f3b40f51-d725-4fc2-8378-c6b55d811c57'}]

        for matched_actor in matched_actors:
            fields = matched_actor["name"].split(":")
            assert len(fields) == 2, f"invalid actor name: {matched_actor['name']}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])

            if (self._dp_size * pg_index + local_rank) // self._tp_size == self._dp_rank:
                worker = ray.get_actor(**matched_actor)
                self.workers.append(worker)
                if (self._dp_size * pg_index + local_rank) / self._tp_size == self._dp_rank:
                    self.master_worker = worker

        print(f"IN init_engine {self.workers=}")

    async def chat_completion(self, raw_request: Request):
        request = await raw_request.json()

        # only send request to master worker in tp rank 0
        output_future = self.master_worker.chat_completion.remote(request)
        [outputs] = await asyncio.gather(output_future)
        return JSONResponse(outputs)

    async def get_metrics(self):
        """Get server metrics including number of running requests."""
        try:
            # Get metrics from the master worker
            if self.master_worker is None:
                metrics_text = "sglang:num_running_reqs 0\n"
            else:
                # Get the number of running requests from the master worker
                num_running_reqs_future = self.master_worker.get_num_running_requests.remote()
                num_running_reqs = await asyncio.gather(num_running_reqs_future)

                # Format as prometheus-style metrics
                metrics_text = f"sglang:num_running_reqs {num_running_reqs[0]}\n"

            # Return proper HTTP response
            return Response(content=metrics_text, media_type="text/plain")
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            # Return 0 if we can't get metrics
            return Response(content="sglang:num_running_reqs 0\n", media_type="text/plain")

    async def wake_up(self):
        print(f"IN wake_up {self.workers=}")
        tasks = [worker.wake_up.remote() for worker in self.workers]
        if tasks:
            await asyncio.gather(*tasks)

    async def sleep(self):
        print(f"IN sleep {self.workers=}")
        tasks = [worker.sleep.remote() for worker in self.workers]
        if tasks:
            await asyncio.gather(*tasks)