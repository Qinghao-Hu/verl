import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

import torch.distributed as dist
import zmq
import zmq.asyncio
import zmq.error
from torch.distributed.device_mesh import DeviceMesh

from verl.workers.drafter.background_trainer import BackgroundTrainer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def get_free_port() -> int:
    """Get a free port for ZMQ communication."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def get_host_ip():
    """Get host IP from environment variables."""
    host_ipv4 = os.environ.get("MY_HOST_IP", None)
    host_ipv6 = os.environ.get("MY_HOST_IPV6", None)
    return host_ipv4 or host_ipv6 or "127.0.0.1"


class WorkerState(Enum):
    BUSY = "busy"
    RELEASED = "released"
    TRAINING = "training"


@dataclass
class WorkerInfo:
    worker_id: int
    gpu_id: int
    dp_rank: int
    tp_rank: int
    state: WorkerState
    last_release_time: Optional[float] = None
    hostname: Optional[str] = None


class CentralCoordinator:
    """Central coordinator for managing global state using ZMQ REQ-REP pattern."""

    def __init__(self, port: int, tp_size: int = 1):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        # Global state maintained by coordinator
        self.worker_states: Dict[int, WorkerInfo] = {}
        self.released_workers: Set[int] = set()
        self.training_workers: Set[int] = set()
        self.completed_workers: Set[int] = set()
        self.training_started = False
        self.tp_size = tp_size

        logger.info(f"Central coordinator started on port {port}, tp_size={tp_size}")

    async def run(self):
        """Main coordinator loop."""
        while True:
            try:
                # Receive request with timeout
                poller = zmq.asyncio.Poller()
                poller.register(self.socket, zmq.POLLIN)

                # Poll with 10ms timeout for better responsiveness
                socks = dict(await poller.poll(10))

                if self.socket in socks:
                    # Receive request
                    message = await self.socket.recv_json()

                    # Process request and prepare response
                    response = await self._process_request(message)

                    # Send response immediately
                    await self.socket.send_json(response)
                else:
                    # No request, yield to other tasks with minimal sleep
                    await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Coordinator error: {e}")
                # Try to send error response if socket is in correct state
                try:
                    await self.socket.send_json({"status": "error", "message": str(e)})
                except:
                    # Socket might be in wrong state, skip
                    pass

    async def _process_request(self, message: dict) -> dict:
        """Process incoming request and return response."""
        request_type = message.get("type")
        worker_id = message.get("worker_id")

        if request_type == "register":
            # Register new worker
            worker_info = WorkerInfo(
                worker_id=worker_id,
                gpu_id=message["gpu_id"],
                dp_rank=message["dp_rank"],
                tp_rank=message["tp_rank"],
                state=WorkerState.BUSY,
                hostname=message.get("hostname"),
            )
            self.worker_states[worker_id] = worker_info
            return {"status": "ok", "message": "Worker registered"}

        elif request_type == "release":
            # Mark worker as released
            if worker_id in self.worker_states:
                worker_info = self.worker_states[worker_id]
                self.worker_states[worker_id].state = WorkerState.RELEASED
                self.worker_states[worker_id].last_release_time = time.time()

                # When a worker is released, we need to add ALL TP ranks for that DP worker
                # because they all share the same memory release
                dp_rank = worker_info.dp_rank
                for wid, winfo in self.worker_states.items():
                    if winfo.dp_rank == dp_rank:
                        self.released_workers.add(wid)
                        self.worker_states[wid].state = WorkerState.RELEASED
                        self.worker_states[wid].last_release_time = time.time()

                # Check if this worker should start training
                # IMPORTANT: Set training_started BEFORE returning to prevent race conditions
                should_start = self._should_start_training(worker_id)
                if should_start:
                    # Immediately mark training as started to prevent other workers from initiating
                    self.training_started = True
                    logger.info(
                        f"Coordinator: Worker {worker_id} selected to coordinate training, "
                        f"marking training_started=True. Released workers: {sorted(self.released_workers)}"
                    )

                # Return global state
                return {
                    "status": "ok",
                    "global_state": self._get_global_state(),
                    "should_start_training": should_start,
                }
            return {"status": "error", "message": "Worker not found"}

        elif request_type == "mark_training":
            # Mark worker as training
            if worker_id in self.released_workers:
                self.worker_states[worker_id].state = WorkerState.TRAINING
                self.training_workers.add(worker_id)
                # Note: training_started should already be set by the coordinator when selecting the training initiator
                # Don't set it here to avoid race conditions
                logger.debug(f"Worker {worker_id} marked as training. Total training: {len(self.training_workers)}")
                return {
                    "status": "ok",
                    "training_started": self.training_started,
                    "training_workers": list(self.training_workers),
                }
            else:
                # Check if this is a TP partner that should be included
                if worker_id in self.worker_states:
                    worker_info = self.worker_states[worker_id]
                    dp_rank = worker_info.dp_rank

                    # Check if any worker with same DP rank is already in training
                    dp_in_training = any(
                        self.worker_states[wid].dp_rank == dp_rank
                        for wid in self.training_workers
                        if wid in self.worker_states
                    )

                    if dp_in_training or self.training_started:
                        # This is a TP partner that should be in training
                        # Add to released and training sets
                        self.released_workers.add(worker_id)
                        self.worker_states[worker_id].state = WorkerState.TRAINING
                        self.training_workers.add(worker_id)
                        logger.info(
                            f"Worker {worker_id} (TP partner) added to training. Total training: {len(self.training_workers)}"
                        )
                        return {
                            "status": "ok",
                            "training_started": self.training_started,
                            "training_workers": list(self.training_workers),
                        }

                logger.warning(f"Worker {worker_id} not eligible for training")
                return {
                    "status": "error",
                    "message": f"Worker {worker_id} not in released workers",
                    "training_started": self.training_started,
                    "training_workers": list(self.training_workers),
                }

        elif request_type == "mark_completed":
            # Mark worker as completed
            self.completed_workers.add(worker_id)
            return {"status": "ok", "all_completed": len(self.completed_workers) >= len(self.worker_states)}

        elif request_type == "get_state":
            # Return current global state
            return {"status": "ok", "global_state": self._get_global_state()}

        elif request_type == "reset":
            # Reset for new batch
            for w in self.worker_states.values():
                w.state = WorkerState.BUSY
                w.last_release_time = None
            self.released_workers.clear()
            self.completed_workers.clear()
            # Don't clear training_workers if training is active
            if not self.training_started:
                self.training_workers.clear()
            return {"status": "ok"}

        return {"status": "error", "message": "Unknown request type"}

    def _get_global_state(self) -> dict:
        """Get current global state."""
        return {
            "released_workers": list(self.released_workers),
            "training_workers": list(self.training_workers),
            "completed_workers": list(self.completed_workers),
            "training_started": self.training_started,
            "worker_states": {
                wid: {
                    "state": info.state.value,
                    "gpu_id": info.gpu_id,
                    "dp_rank": info.dp_rank,
                    "tp_rank": info.tp_rank,
                    "hostname": info.hostname,
                }
                for wid, info in self.worker_states.items()
            },
        }

    def _should_start_training(self, worker_id: int) -> bool:
        """Check if this worker should start training.
        Note: worker_id is a process rank, not DP rank."""
        # If training has already started, no worker should initiate it again
        if self.training_started:
            return False

        # Count DP workers that have been released
        # Since we now add all TP ranks when a DP worker releases,
        # we can count unique DP ranks from released workers
        released_dp_workers = set()
        for w_id in self.released_workers:
            if w_id in self.worker_states:
                released_dp_workers.add(self.worker_states[w_id].dp_rank)

        # Use the configured min_workers or default to half
        min_dp_workers = 1  # TODO: make this configurable, for now use 1 for testing

        # Check conditions
        released_dp_count = len(released_dp_workers)
        not_training = worker_id not in self.training_workers
        threshold_met = released_dp_count >= min_dp_workers

        # Only the first released worker that meets threshold should initiate training
        if threshold_met:
            # Get all process ranks for the released DP workers that should train
            # Important: Include ALL TP ranks for each DP worker
            training_process_ranks = []
            for dp_rank in sorted(released_dp_workers)[:min_dp_workers]:
                # Add all process ranks (all TP ranks) for this DP worker
                for wid, winfo in self.worker_states.items():
                    if winfo.dp_rank == dp_rank:
                        training_process_ranks.append(wid)

            training_process_ranks = sorted(training_process_ranks)

            # Only the first worker in the released set should initiate
            # But make sure it's actually released
            eligible_initiators = [w for w in training_process_ranks if w in self.released_workers]
            if eligible_initiators:
                should_start = worker_id == min(eligible_initiators) and not_training
            else:
                should_start = False

            if should_start:
                # Log which workers will be used for training
                logger.info(f"Worker {worker_id} will coordinate training with process ranks: {training_process_ranks}")

            return should_start

        return False

    async def cleanup(self):
        """Clean up resources."""
        self.socket.close()
        self.context.term()


class WorkerClient:
    """Client for workers to communicate with central coordinator."""

    def __init__(self, coordinator_address: str):
        self.coordinator_address = coordinator_address  # Store for reconnection
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(coordinator_address)
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 10000)  # 10 second send timeout
        self._lock = asyncio.Lock()  # Add lock for thread-safe socket access

    async def register_worker(self, worker_info: dict) -> dict:
        """Register worker with coordinator."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self._lock:
                    request = {"type": "register", **worker_info}
                    await self.socket.send_json(request)
                    return await self.socket.recv_json()
            except zmq.error.Again:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Worker {worker_info.get('worker_id', 'unknown')} registration timeout, "
                        f"retrying... (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    # Reset socket connection
                    self.socket.close()
                    self.socket = self.context.socket(zmq.REQ)
                    self.socket.connect(self.coordinator_address)
                    self.socket.setsockopt(zmq.RCVTIMEO, 10000)
                    self.socket.setsockopt(zmq.SNDTIMEO, 10000)
                else:
                    logger.error(
                        f"Worker {worker_info.get('worker_id', 'unknown')} "
                        f"failed to register after {max_retries} attempts"
                    )
                    raise

    async def release_worker(self, worker_id: int) -> dict:
        """Notify coordinator that worker has released memory."""
        async with self._lock:
            request = {"type": "release", "worker_id": worker_id}
            await self.socket.send_json(request)
            return await self.socket.recv_json()

    async def mark_training(self, worker_id: int) -> dict:
        """Mark worker as training with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self._lock:
                    request = {"type": "mark_training", "worker_id": worker_id}
                    await self.socket.send_json(request)
                    response = await self.socket.recv_json()
                    return response
            except (zmq.error.Again, zmq.error.ZMQError) as e:
                # Always reset socket on error to clear invalid state
                logger.warning(
                    f"Worker {worker_id} error marking as training: {e}, "
                    f"resetting socket... (attempt {attempt + 1}/{max_retries})"
                )

                # Reset socket connection immediately to clear state
                try:
                    self.socket.close()
                except:
                    pass  # Socket might already be closed

                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(self.coordinator_address)
                self.socket.setsockopt(zmq.RCVTIMEO, 15000)  # Increase timeout
                self.socket.setsockopt(zmq.SNDTIMEO, 15000)

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Worker {worker_id} failed to mark as training after {max_retries} attempts: {e}")
                    return {"status": "error", "training_started": False}

    async def mark_completed(self, worker_id: int) -> dict:
        """Mark worker as completed generation."""
        async with self._lock:
            request = {"type": "mark_completed", "worker_id": worker_id}
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.socket.send_json(request)
                    return await self.socket.recv_json()
                except (zmq.error.Again, zmq.error.ZMQError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Worker {worker_id} error marking completed: {e}, retrying... (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        # Always reset socket on any error
                        try:
                            self.socket.close()
                        except:
                            pass
                        self.socket = self.context.socket(zmq.REQ)
                        self.socket.connect(self.coordinator_address)
                        self.socket.setsockopt(zmq.RCVTIMEO, 10000)
                        self.socket.setsockopt(zmq.SNDTIMEO, 10000)
                    else:
                        logger.error(f"Worker {worker_id} failed to mark completed after {max_retries} attempts: {e}")
                        # Return a default response to avoid crashing
                        return {"status": "error", "all_completed": False}

    async def get_global_state(self) -> dict:
        """Get current global state."""
        max_retries = 1  # Reduce retries for faster failure detection
        for attempt in range(max_retries):
            try:
                async with self._lock:
                    request = {"type": "get_state"}
                    # Use shorter timeout for faster failure detection
                    self.socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
                    await self.socket.send_json(request)
                    response = await self.socket.recv_json()
                    # Reset timeout back
                    self.socket.setsockopt(zmq.RCVTIMEO, 10000)
                    return response
            except (zmq.error.Again, zmq.error.ZMQError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error getting global state: {e} (attempt {attempt + 1}/{max_retries})")
                    # Reset socket to clear any invalid state
                    try:
                        self.socket.close()
                    except:
                        pass

                    self.socket = self.context.socket(zmq.REQ)
                    self.socket.connect(self.coordinator_address)
                    self.socket.setsockopt(zmq.RCVTIMEO, 2000)
                    self.socket.setsockopt(zmq.SNDTIMEO, 10000)
                    await asyncio.sleep(0.1)  # Very small delay before retry
                else:
                    # Don't log as error - coordinator shutdown is expected
                    logger.debug(f"Coordinator unavailable after {max_retries} attempts")
                    return {"status": "error", "coordinator_down": True}

    async def reset_state(self) -> dict:
        """Reset global state for new batch."""
        async with self._lock:
            request = {"type": "reset"}
            await self.socket.send_json(request)
            return await self.socket.recv_json()

    async def cleanup(self):
        """Clean up resources."""
        self.socket.close()
        self.context.term()


class RolloutDrafterManager:
    """Manages early memory release and coordinates background training on freed GPUs."""

    def __init__(self, device_mesh: DeviceMesh, rollout_config):
        self.device_mesh = device_mesh
        self.rollout_config = rollout_config

        # Check if distributed is initialized
        assert dist.is_initialized()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tp_size = device_mesh["tp"].size()
        self.dp_size = device_mesh["dp"].size()
        self.dp_rank = device_mesh["dp"].get_local_rank()
        self.tp_rank = device_mesh["tp"].get_local_rank()

        # Default to half of DP workers if not specified
        self.min_workers_for_training = rollout_config.min_workers_for_training

        # Local state cache
        self.global_state_cache = None
        self.last_state_update = 0

        # Coordinator setup
        self.coordinator = None
        self.worker_client = None
        self.coordinator_task = None

        # Only rank 0 runs coordinator
        self.is_coordinator = self.rank == 0

        # Background training
        self._training_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self.background_trainer = BackgroundTrainer(self.rank)
        # Lifecycle flags
        self._training_coordination_done = False  # Set True once coordinator has executed training setup
        # Local fast-path completion flags (avoid repeated coordinator polling after global completion)
        self._all_completed_local = False
        self._all_completed_mark_time = None  # type: Optional[float]
        # If we can't reach coordinator after marking all complete, allow local stale timeout to stop training
        self._all_completed_stale_timeout = 15.0  # seconds

        # Get local worker info
        self.gpu_id = self.rank
        self.hostname = os.environ.get("HOSTNAME", "unknown")

        self.global_device_mesh_list = [
            DeviceMesh("cuda", list(range(i * self.tp_size, (i + 1) * self.tp_size))) for i in range(self.dp_size)
        ]
        self.drafter_device_mesh = self.global_device_mesh_list[self.dp_rank]

        logger.info(
            f"RolloutDrafterManager initialized: rank={self.rank}, "
            f"gpu_id={self.gpu_id}, tp_size={self.tp_size}, dp_size={self.dp_size}, "
            f"min_workers_for_training={self.min_workers_for_training}, "
            f"is_coordinator={self.is_coordinator}"
        )

    async def initialize(self):
        """Initialize communication system."""
        if self.is_coordinator:
            # Start coordinator on rank 0
            port = get_free_port()
            self.coordinator = CentralCoordinator(port, tp_size=self.tp_size)
            self.coordinator_task = asyncio.create_task(self.coordinator.run())

            # Give coordinator time to start listening
            await asyncio.sleep(0.5)

            # Broadcast coordinator address to all workers
            coordinator_address = f"tcp://{get_host_ip()}:{port}"
            logger.info(f"Coordinator started at {coordinator_address}")
        else:
            coordinator_address = None

        # All workers get coordinator address
        address_list = [coordinator_address]
        dist.broadcast_object_list(address_list, src=0)
        coordinator_address = address_list[0]

        # Create worker client
        self.worker_client = WorkerClient(coordinator_address)

        # Small delay for non-coordinator workers to ensure coordinator is ready
        if not self.is_coordinator:
            await asyncio.sleep(0.2)

        # Register this worker
        await self.worker_client.register_worker(
            {
                "worker_id": self.rank,
                "gpu_id": self.gpu_id,
                "dp_rank": self.dp_rank,
                "tp_rank": self.tp_rank,
                "hostname": self.hostname,
            }
        )

        logger.info(f"Worker {self.rank} registered with coordinator")
        # Reset coordination flag at the start of a new generation round
        self._training_coordination_done = False

    async def release_worker_memory(self, worker_id: int) -> bool:
        """Release memory for a worker and check if training should start.
        Note: worker_id here is the process rank (not DP rank)."""
        if worker_id != self.rank:
            logger.warning(f"Worker {self.rank} cannot release memory for worker {worker_id}")
            return False

        # Notify coordinator with process rank
        response = await self.worker_client.release_worker(worker_id)

        if response["status"] != "ok":
            logger.error(f"Failed to release worker {worker_id}: {response}")
            return False

        # Update local cache
        self.global_state_cache = response["global_state"]
        self.last_state_update = time.time()

        released_count = len(self.global_state_cache["released_workers"])
        released_workers = sorted(self.global_state_cache["released_workers"])
        released_gpus = self._get_released_gpu_ids()
        logger.info(
            f"Worker {worker_id} memory released. Total released: {released_count}/{self.dp_size}. "
            f"Released Workers: {released_workers}, Released GPUs: {released_gpus} "
            f"(TP={self.tp_size})"
        )

        # Check if we should start training (as coordinator)
        if response.get("should_start_training", False) and not self.background_trainer.is_training_active:
            logger.info(f"Worker {self.rank} selected as training coordinator by central coordinator")
            # Small delay to ensure state propagation
            await asyncio.sleep(0.2)
            if not self._training_coordination_done:
                asyncio.create_task(self._start_background_training())
            else:
                logger.debug(
                    f"Worker {self.rank} already coordinated training once; skipping duplicate coordination attempt"
                )
        # Also check if we should join training (as participant)
        elif not self.background_trainer.is_training_active:
            # Check if training has already started globally
            if self.global_state_cache.get("training_started", False):
                # Training already started, check immediately if we should join
                await self._check_and_join_training()
            else:
                # Give coordinator time to set things up, then check if we should join
                await asyncio.sleep(1.5)
                await self._check_and_join_training()

        return True

    def _get_released_gpu_ids(self) -> list[int]:
        """Get list of released GPU IDs considering TP degree."""
        if not self.global_state_cache:
            return []

        gpu_ids = []
        for worker_id in self.global_state_cache["released_workers"]:
            worker_info = self.global_state_cache["worker_states"].get(str(worker_id), {})
            if "gpu_id" in worker_info:
                # Since we now release all TP ranks together,
                # each worker_id has its own GPU ID
                gpu_ids.append(worker_info["gpu_id"])

        return sorted(set(gpu_ids))  # Remove duplicates and sort

    async def _check_and_join_training(self):
        """Check if this worker should join training that was initiated by another worker."""
        if self.background_trainer.is_training_active:
            # Already in training
            return

        # Get current state to check if we're in training group
        response = await self.worker_client.get_global_state()
        if response["status"] != "ok":
            return

        self.global_state_cache = response["global_state"]
        training_workers = self.global_state_cache.get("training_workers", [])

        # Check if we're in the training group OR if our TP partner is
        # This is important because we might have been marked but not yet in the list
        in_training_group = self.rank in training_workers

        # Also check if our DP partner is in training (same DP rank)
        if not in_training_group and training_workers:
            my_dp_rank = self.dp_rank
            for worker_id in training_workers:
                worker_info = self.global_state_cache["worker_states"].get(str(worker_id), {})
                if worker_info.get("dp_rank") == my_dp_rank:
                    # Our DP partner is training, we should join too
                    in_training_group = True
                    logger.info(f"Worker {self.rank} detected DP partner {worker_id} in training, joining...")
                    break

        if in_training_group:
            logger.info(f"Worker {self.rank} detected it should join training")

            # Get all workers with the same DP rank that should train together
            # This includes ALL TP partners, not just those already marked
            my_dp_rank = self.dp_rank
            training_process_ranks = []

            # Include ALL workers with the same DP rank (all TP partners)
            for w_id, w_info in self.global_state_cache["worker_states"].items():
                if w_info.get("dp_rank") == my_dp_rank:
                    training_process_ranks.append(int(w_id))

            training_process_ranks = sorted(training_process_ranks)
            training_gpu_ids = []
            for worker_id in training_process_ranks:
                worker_info = self.global_state_cache["worker_states"].get(str(worker_id), {})
                if "gpu_id" in worker_info:
                    training_gpu_ids.append(worker_info["gpu_id"])

            logger.info(f"Worker {self.rank} joining training with ranks {training_process_ranks}")

            # Mark as training
            self.background_trainer.set_training_active(True)

            # Opportunistic: mark this worker as completed (generation) if not already to speed up preemption
            if str(self.rank) not in self.global_state_cache.get("worker_states", {}):
                # State inconsistency; skip
                pass
            else:
                if self.rank not in self.global_state_cache.get("completed_workers", []):
                    try:
                        # Fire-and-forget (don't block join) but await small timeout
                        await asyncio.wait_for(self.worker_client.mark_completed(self.rank), timeout=5.0)
                        # Refresh state cache after marking
                        state_resp = await self.worker_client.get_global_state()
                        if state_resp.get("status") == "ok":
                            self.global_state_cache = state_resp["global_state"]
                    except Exception as e:  # Non-fatal
                        logger.debug(f"Worker {self.rank} failed to auto-mark completed on join: {e}")

            # Initialize training model
            success = await asyncio.wait_for(
                self.background_trainer.initialize_training_model(self.drafter_device_mesh, training_process_ranks),
                timeout=30.0,
            )

            if success:
                logger.info(
                    f"Worker {self.rank} successfully joined training. "
                    f"Process ranks: {training_process_ranks}, GPU IDs: {training_gpu_ids}"
                )

                # Run training
                self._training_task = asyncio.create_task(self._run_background_training())
                self._monitor_task = asyncio.create_task(self._monitor_completion())

                # Give the tasks a moment to start
                await asyncio.sleep(0.1)
            else:
                logger.error(f"Worker {self.rank} failed to join training")
                self.background_trainer.set_training_active(False)
        else:
            # Not in training group yet, but might be soon
            logger.debug(f"Worker {self.rank} not in training group yet. Training workers: {training_workers}")

    async def _start_background_training(self):
        """Start background training on released workers."""
        # This worker is the coordinator - it needs to coordinate all released workers
        if self._training_coordination_done:
            logger.info(f"Worker {self.rank} attempted to re-enter _start_background_training; ignoring (already done)")
            return
        logger.info(f"Worker {self.rank} coordinating background training setup")

        released_workers = sorted(self.global_state_cache["released_workers"])

        # Get all TP ranks for the first DP worker(s)
        # We need to ensure all TP ranks for a DP worker participate
        released_dp_workers = set()
        for w_id in released_workers:
            if str(w_id) in self.global_state_cache["worker_states"]:
                released_dp_workers.add(self.global_state_cache["worker_states"][str(w_id)]["dp_rank"])

        # Get all process ranks for the first min_dp_workers DP workers
        training_workers = []
        for dp_rank in sorted(released_dp_workers)[: self.min_workers_for_training]:
            for w_id, w_info in self.global_state_cache["worker_states"].items():
                if w_info["dp_rank"] == dp_rank:
                    training_workers.append(int(w_id))

        training_workers = sorted(training_workers)

        logger.info(
            f"Worker {self.rank} coordinating training with DP workers:  "
            f"(process ranks: {training_workers}, from {len(released_workers)} released ranks)"
        )

        # IMPORTANT: Mark all workers as training BEFORE any can check their status
        # This prevents workers from seeing incomplete training groups
        successful_marks = []
        for worker_id in training_workers:
            # Avoid duplicate mark_training if this worker already in training_workers
            already_training = self.global_state_cache and worker_id in self.global_state_cache.get(
                "training_workers", []
            )
            if already_training:
                logger.debug(
                    f"Coordinator sees worker {worker_id} already in training_workers; skipping mark_training call"
                )
                successful_marks.append(worker_id)
                continue
            logger.info(f"Coordinator marking worker {worker_id} for training...")
            result = await self.worker_client.mark_training(worker_id)
            if result.get("status") == "error":
                logger.warning(
                    f"Failed to mark worker {worker_id} as training: {result.get('message', 'Unknown error')}"
                )
            else:
                successful_marks.append(worker_id)
                logger.info(f"Successfully marked worker {worker_id} for training")

        # Check if we successfully marked at least the minimum required workers
        if len(successful_marks) < len(training_workers):
            logger.warning(
                f"Only marked {len(successful_marks)}/{len(training_workers)} workers for training. "
                f"Marked: {successful_marks}, Expected: {training_workers}"
            )

        # Now all workers should check if they're in the training group
        response = await self.worker_client.get_global_state()
        if response["status"] == "ok":
            self.global_state_cache = response["global_state"]
            final_training_workers = sorted(self.global_state_cache["training_workers"])

            # final_training_workers contains process ranks that should participate in training
            training_process_ranks = sorted(final_training_workers)

            # Get GPU IDs for logging
            training_gpu_ids = []
            for worker_id in training_process_ranks:
                worker_info = self.global_state_cache["worker_states"].get(str(worker_id), {})
                if "gpu_id" in worker_info:
                    training_gpu_ids.append(worker_info["gpu_id"])

            total_training_gpus = len(training_process_ranks)

            logger.info(f"Training setup: process_ranks={training_process_ranks}, gpu_ids={training_gpu_ids}")

            # Check if this process is in the training group
            is_training_rank = self.rank in training_process_ranks

            logger.info(
                f"Worker {self.rank}: is_training_rank={is_training_rank}, "
                f"my_rank={self.rank}, training_ranks={training_process_ranks}"
            )

            if is_training_rank:
                logger.info(f"Worker {self.rank} is in training group, creating process group...")

                # Proceed with training initialization
                self.background_trainer.set_training_active(True)

                success = await asyncio.wait_for(
                    self.background_trainer.initialize_training_model(self.drafter_device_mesh, training_process_ranks),
                    timeout=30.0,
                )

                logger.info(
                    f"Worker {self.rank} participating in training. "
                    f"Process ranks: {sorted(training_process_ranks)}, "
                    f"GPU IDs: {sorted(training_gpu_ids)}, "
                    f"Total GPUs for training: {total_training_gpus} "
                    f"(Initialization success: {success})"
                )

                # Run training
                # Guard: do not spawn duplicate training loop for same worker
                if self._training_task and not self._training_task.done():
                    logger.warning(f"Worker {self.rank} already has an active training task; skipping new creation")
                else:
                    self._training_task = asyncio.create_task(self._run_background_training())
                logger.info(f"Worker {self.rank} created training task")

                # Auto-mark all training workers as completed (they have finished generation if released)
                # Use mark_worker_completed to trigger local fast-path flag setting if this results in all-complete.
                if self.rank == min(training_process_ranks):
                    for tw in training_process_ranks:
                        if tw not in self.global_state_cache.get("completed_workers", []):
                            try:
                                # Await sequentially to ensure ordering; last one may set all_completed flag
                                await self.mark_worker_completed(tw)
                            except Exception as e:  # Non-fatal
                                logger.debug("Auto mark_completed failed for worker %s: %s", tw, e)

                # Also create a monitoring task to check for all workers completed
                self._monitor_task = asyncio.create_task(self._monitor_completion())

                # Give the task a moment to start
                await asyncio.sleep(0.1)
            else:
                logger.info(f"Worker {self.rank} not selected for training, continuing with generation")

        # Mark coordination as done to prevent re-entry
        self._training_coordination_done = True

    async def _run_background_training(self):
        """Run background training loop."""
        logger.info(f"Worker {self.rank} starting background training loop")
        try:
            step = 0
            max_steps = 1000
            # Prevent duplicate shutdown attempts
            if not hasattr(self, "_shutdown_initiated"):
                self._shutdown_initiated = False
            # Frequency (in steps) to poll coordinator for completion status
            completion_poll_interval = 2  # More responsive preemption
            consecutive_poll_failures = 0

            # Log initial state
            logger.info(
                f"Worker {self.rank} training loop started: shutdown={self._shutdown_event.is_set()}, "
                f"active={self.background_trainer.is_training_active}"
            )

            while not self._shutdown_event.is_set() and step < max_steps and self.background_trainer.is_training_active:
                # Local fast-path: if we've already observed global completion earlier, exit promptly
                if self._all_completed_local:
                    # Double-check stale timeout or just proceed to shutdown
                    if self._all_completed_mark_time is None or (time.time() - self._all_completed_mark_time) >= 0:
                        logger.info(
                            "Worker %s stopping training due to local all-complete flag at step %d", self.rank, step
                        )
                        self.background_trainer.set_training_active(False)
                        self._shutdown_event.set()
                        break
                # Cooperative preemption: stop promptly if inference finished everywhere
                if step % completion_poll_interval == 0:
                    response = await self.worker_client.get_global_state()

                    # If coordinator is down, assume all workers completed and stop immediately
                    if response.get("coordinator_down", False):
                        logger.info(
                            f"Worker {self.rank} detected coordinator shutdown; "
                            f"stopping training immediately at step {step}"
                        )
                        self._all_completed_local = True
                        self._all_completed_mark_time = time.time()
                        self.background_trainer.set_training_active(False)
                        self._shutdown_event.set()
                        break

                    if response.get("status") == "ok":
                        self.global_state_cache = response["global_state"]
                        completed = len(self.global_state_cache["completed_workers"])
                        total = len(self.global_state_cache["worker_states"])
                        if completed >= total and total > 0:
                            logger.info(
                                f"Worker {self.rank} detected all inference workers completed "
                                f"({completed}/{total}); preempting background training at step {step}"
                            )
                            # Set local flag so if coordinator goes away we still terminate decisively
                            if not self._all_completed_local:
                                self._all_completed_local = True
                                self._all_completed_mark_time = time.time()
                            self.background_trainer.set_training_active(False)
                            self._shutdown_event.set()
                            break
                        consecutive_poll_failures = 0  # reset on success
                    else:
                        consecutive_poll_failures += 1
                        if consecutive_poll_failures >= 2:  # Reduce to 2 for faster shutdown
                            logger.info(
                                f"Worker {self.rank} stopping training after {consecutive_poll_failures} poll failures "
                                f"(coordinator likely down)"
                            )
                            self._all_completed_local = True
                            self._all_completed_mark_time = time.time()
                            self.background_trainer.set_training_active(False)
                            self._shutdown_event.set()
                            break

                # Training step with timeout
                success = await asyncio.wait_for(
                    self.background_trainer.training_step(
                        step, len(self.global_state_cache.get("training_workers", []))
                    ),
                    timeout=30.0,  # 30 second timeout per step
                )

                if success:
                    step += 1
                else:
                    # Log the failure but continue training
                    logger.warning(f"Worker {self.rank} training step {step} failed, continuing...")
                    # Try to reinitialize if too many failures
                    if step == 0:
                        logger.error(f"Worker {self.rank} failed on first step, stopping training")
                        break
                    step += 1  # Continue to next step even on failure

                if step % 50 == 0:
                    logger.info(f"Worker {self.rank} completed {step} training steps")

                await asyncio.sleep(0.01)

            logger.info(f"Worker {self.rank} training completed: {step}/{max_steps} steps")

        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            # Always cleanup training resources
            logger.info(f"Worker {self.rank} cleaning up training resources...")
            await self.background_trainer.cleanup_training()

    async def _monitor_completion(self):
        """Monitor for all workers completing to stop training."""
        logger.info(f"Worker {self.rank} starting completion monitor")

        while self.background_trainer.is_training_active:
            await asyncio.sleep(1.0)  # Check more frequently

            response = await self.worker_client.get_global_state()

            # If coordinator is down, stop immediately
            if response.get("coordinator_down", False):
                logger.info(f"Worker {self.rank} monitor detected coordinator shutdown, stopping training")
                self._all_completed_local = True
                self._all_completed_mark_time = time.time()
                self.background_trainer.set_training_active(False)
                self._shutdown_event.set()
                break

            if response["status"] == "ok":
                self.global_state_cache = response["global_state"]
                completed = len(self.global_state_cache["completed_workers"])
                total = len(self.global_state_cache["worker_states"])

                logger.debug(f"Worker {self.rank} monitor: {completed}/{total} workers completed")

                if completed >= total and total > 0:
                    logger.info(
                        f"Worker {self.rank} monitor detected all workers completed "
                        f"({completed}/{total}). Requesting training stop..."
                    )
                    if not self._all_completed_local:
                        self._all_completed_local = True
                        self._all_completed_mark_time = time.time()
                    self.background_trainer.set_training_active(False)
                    self._shutdown_event.set()
                    break

    async def mark_worker_completed(self, worker_id: int):
        """Mark worker as completed generation."""
        # Before marking as completed, check if we should join training
        if not self.background_trainer.is_training_active:
            # Get latest state to check if training has started
            state_response = await self.worker_client.get_global_state()

            # If coordinator is down, just return - we're done
            if state_response.get("coordinator_down", False):
                logger.info(f"Worker {worker_id} skipping mark_completed - coordinator is down")
                return

            if state_response["status"] == "ok":
                self.global_state_cache = state_response["global_state"]
                if self.global_state_cache.get("training_started", False):
                    # Training has started, check if we should join
                    await self._check_and_join_training()

        response = await self.worker_client.mark_completed(worker_id)

        # If we get an error response, assume coordinator is down and just continue
        if response["status"] == "error":
            logger.info(f"Worker {worker_id} couldn't mark completed (coordinator likely down)")
            return

        if response["status"] == "ok":
            all_completed = response.get("all_completed", False)
            logger.info(f"Worker {worker_id} marked as completed. All completed: {all_completed}")
            # If all inference workers are done, ensure background training is stopped
            if all_completed and self.background_trainer.is_training_active and not self._shutdown_event.is_set():
                logger.info(f"Worker {self.rank} initiating background training shutdown after all inference completed")
                # Record local completion flags for fast-path termination
                if not self._all_completed_local:
                    self._all_completed_local = True
                    self._all_completed_mark_time = time.time()
                self.background_trainer.set_training_active(False)
                self._shutdown_event.set()

    def get_global_release_status(self) -> dict:
        """Get current global release status."""
        if not self.global_state_cache:
            return {
                "total_workers": self.dp_size,
                "total_released": 0,
                "released_percentage": 0,
                "released_worker_ids": [],
                "released_gpu_ids": [],
                "training_active": False,
                "training_workers": [],
                "current_worker": self.rank,
                "error": "No state available",
            }

        state = self.global_state_cache
        released_workers = state["released_workers"]
        training_workers = state["training_workers"]
        completed_workers = state["completed_workers"]

        # Get GPU IDs
        released_gpu_ids = self._get_released_gpu_ids()

        # Get training worker details
        training_info = []
        for wid in training_workers:
            worker_info = state["worker_states"].get(str(wid), {})
            training_info.append(
                {
                    "worker_id": wid,
                    "gpu_id": worker_info.get("gpu_id", -1),
                    "hostname": worker_info.get("hostname", "unknown"),
                }
            )

        return {
            "total_workers": self.dp_size,
            "total_released": len(released_workers),
            "released_percentage": len(released_workers) / self.dp_size * 100,
            "released_worker_ids": sorted(released_workers),
            "released_gpu_ids": released_gpu_ids,
            "training_active": self.background_trainer.is_training_active,
            "training_initialized": self.background_trainer.is_training_initialized,
            "training_workers": sorted(training_workers),
            "training_worker_info": training_info,
            "total_training_gpus": len(training_workers) * self.tp_size,
            "completed_workers": len(completed_workers),
            "all_generation_complete": len(completed_workers) >= self.dp_size,
            "current_worker": self.rank,
            "min_workers_for_training": self.min_workers_for_training,
            "last_update": self.last_state_update,
        }

    async def reset_all_workers(self):
        """Reset all worker states for new batch."""
        await self.worker_client.reset_state()
        logger.info("Worker states reset for new batch")

    async def shutdown_background_training(self):
        """Gracefully shutdown background training."""
        logger.info("Shutting down background training")

        self._shutdown_event.set()
        self.background_trainer.set_training_active(False)

        # Cancel training task
        if self._training_task and not self._training_task.done():
            self._training_task.cancel()
            try:
                await asyncio.wait_for(self._training_task, timeout=2.0)  # Reduced timeout
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=0.5)  # Reduced timeout
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Shutdown coordinator quickly if we're the coordinator
        if self.is_coordinator and self.coordinator:
            if self.coordinator_task:
                self.coordinator_task.cancel()
                await asyncio.wait_for(self.coordinator_task, timeout=0.5)
            await self.coordinator.cleanup()

        # Clean up worker client communication - but don't wait too long
        if self.worker_client:
            await asyncio.wait_for(self.worker_client.cleanup(), timeout=1.0)

        logger.info("Background training shutdown complete")
