import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import ray

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status for elastic scaling"""
    ACTIVE = "active"
    DRAINING = "draining"  # Worker is being drained before shutdown
    INACTIVE = "inactive"  # Worker is shut down
    STARTING = "starting"  # Worker is being activated


@dataclass
class ElasticConfig:
    """Configuration for elastic scaling"""
    # Scaling thresholds
    scale_down_threshold: float = 0.5  # Scale down when load < threshold * capacity
    scale_up_threshold: float = 0.8    # Scale up when load > threshold * capacity
    min_workers: int = 1               # Minimum number of active workers
    max_workers: int = 8               # Maximum number of workers
    
    # Request load calculation
    requests_per_worker_capacity: int = 8  # Expected requests per worker
    
    # Timing parameters
    monitoring_interval: float = 5.0    # How often to check scaling (seconds)
    drain_timeout: float = 30.0         # Max time to wait for worker drain (seconds)
    scale_down_cooldown: float = 60.0   # Cooldown after scaling down (seconds)
    scale_up_cooldown: float = 30.0     # Cooldown after scaling up (seconds)
    
    # Safety parameters
    enable_elastic: bool = True         # Master switch for elastic scaling
    min_stable_period: float = 10.0     # Min time with stable load before scaling


@dataclass 
class WorkerInfo:
    """Information about a worker"""
    worker_id: int
    status: WorkerStatus
    active_requests: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    total_processed: int = 0
    
    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if worker is healthy based on heartbeat"""
        return time.time() - self.last_heartbeat < timeout


@dataclass
class RequestInfo:
    """Information about a request"""
    request_id: str
    worker_id: Optional[int] = None
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    
    @property
    def is_pending(self) -> bool:
        return self.worker_id is None
    
    @property 
    def is_running(self) -> bool:
        return self.worker_id is not None and self.completed_time is None
        
    @property
    def is_completed(self) -> bool:
        return self.completed_time is not None


class ElasticScheduler:
    """
    Elastic scheduler for SGLangRollout that monitors request load and 
    dynamically scales workers up/down to optimize resource usage.
    """
    
    def __init__(self, config: ElasticConfig, worker_group: Any):
        """
        Initialize elastic scheduler
        
        Args:
            config: Elastic scaling configuration
            worker_group: Ray worker group to manage
        """
        self.config = config
        self.worker_group = worker_group
        
        # Worker management
        self.workers: Dict[int, WorkerInfo] = {}
        self.active_workers: Set[int] = set()
        self.draining_workers: Set[int] = set()
        
        # Request tracking
        self.pending_requests: deque = deque()
        self.active_requests: Dict[str, RequestInfo] = {}
        self.request_history: deque = deque(maxlen=1000)  # Keep recent history
        
        # Scaling state
        self.last_scale_down_time: float = 0
        self.last_scale_up_time: float = 0
        self.load_history: deque = deque(maxlen=20)  # Track load over time
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        
        # Initialize workers
        self._initialize_workers()
        
    def _initialize_workers(self):
        """Initialize worker tracking based on current worker group"""
        world_size = self.worker_group.world_size
        for i in range(world_size):
            self.workers[i] = WorkerInfo(worker_id=i, status=WorkerStatus.ACTIVE)
            self.active_workers.add(i)
            
        logger.info(f"Initialized elastic scheduler with {world_size} workers")
        
    async def start(self):
        """Start the elastic scheduler monitoring loop"""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started elastic scheduler")
        
    async def stop(self):
        """Stop the elastic scheduler"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped elastic scheduler")
        
    async def _monitoring_loop(self):
        """Main monitoring loop that handles scaling decisions"""
        while self.is_running:
            try:
                await self._update_worker_status()
                await self._make_scaling_decision()
                await self._redistribute_requests()
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry
                
    async def _update_worker_status(self):
        """Update worker status by querying worker group"""
        try:
            # Get worker status from each worker
            if hasattr(self.worker_group, 'get_worker_status'):
                statuses = await self.worker_group.get_worker_status()
                
                for worker_id, status_info in statuses.items():
                    if worker_id in self.workers:
                        worker = self.workers[worker_id]
                        worker.active_requests = status_info.get('active_requests', 0)
                        worker.last_heartbeat = time.time()
                        worker.total_processed = status_info.get('total_processed', 0)
                        
        except Exception as e:
            logger.warning(f"Failed to update worker status: {e}")
            
    async def _make_scaling_decision(self):
        """Make decision about scaling workers up or down"""
        if not self.config.enable_elastic:
            return
            
        current_time = time.time()
        total_active_requests = self._get_total_active_requests()
        total_capacity = len(self.active_workers) * self.config.requests_per_worker_capacity
        
        # Calculate current load ratio
        load_ratio = total_active_requests / max(total_capacity, 1)
        self.load_history.append((current_time, load_ratio, total_active_requests))
        
        # Check if we have stable load for scaling decision
        if not self._has_stable_load():
            return
            
        # Check scaling conditions
        should_scale_down = (
            load_ratio < self.config.scale_down_threshold and
            len(self.active_workers) > self.config.min_workers and
            current_time - self.last_scale_down_time > self.config.scale_down_cooldown
        )
        
        should_scale_up = (
            load_ratio > self.config.scale_up_threshold and
            len(self.active_workers) < self.config.max_workers and
            current_time - self.last_scale_up_time > self.config.scale_up_cooldown
        )
        
        if should_scale_down:
            await self._scale_down()
        elif should_scale_up:
            await self._scale_up()
            
    def _has_stable_load(self) -> bool:
        """Check if load has been stable for minimum required period"""
        if len(self.load_history) < 3:
            return False
            
        recent_time = time.time() - self.config.min_stable_period
        stable_points = [point for point in self.load_history if point[0] >= recent_time]
        
        if len(stable_points) < 2:
            return False
            
        # Check if load variance is low
        load_values = [point[1] for point in stable_points]
        mean_load = sum(load_values) / len(load_values)
        variance = sum((x - mean_load) ** 2 for x in load_values) / len(load_values)
        
        return variance < 0.1  # Low variance threshold
        
    async def _scale_down(self):
        """Scale down by removing one worker"""
        if len(self.active_workers) <= self.config.min_workers:
            return
            
        # Choose worker with least load to drain
        worker_loads = []
        for worker_id in self.active_workers:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker_loads.append((worker_id, worker.active_requests))
                
        if not worker_loads:
            return
            
        # Sort by load and choose the least loaded worker
        worker_loads.sort(key=lambda x: x[1])
        worker_to_drain = worker_loads[0][0]
        
        logger.info(f"Scaling down: draining worker {worker_to_drain}")
        
        # Mark worker as draining
        self.workers[worker_to_drain].status = WorkerStatus.DRAINING
        self.active_workers.remove(worker_to_drain)
        self.draining_workers.add(worker_to_drain)
        
        # Start draining process
        asyncio.create_task(self._drain_worker(worker_to_drain))
        
        self.last_scale_down_time = time.time()
        
    async def _scale_up(self):
        """Scale up by activating an inactive worker"""
        # Find an inactive worker to activate
        inactive_workers = []
        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.INACTIVE:
                inactive_workers.append(worker_id)
                
        if not inactive_workers:
            logger.warning("No inactive workers available for scaling up")
            return
            
        worker_to_activate = inactive_workers[0]
        logger.info(f"Scaling up: activating worker {worker_to_activate}")
        
        # Mark worker as starting
        self.workers[worker_to_activate].status = WorkerStatus.STARTING
        
        try:
            # Resume worker (implementation depends on SGLang integration)
            await self._resume_worker(worker_to_activate)
            
            # Mark as active
            self.workers[worker_to_activate].status = WorkerStatus.ACTIVE
            self.active_workers.add(worker_to_activate)
            
            self.last_scale_up_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to scale up worker {worker_to_activate}: {e}")
            self.workers[worker_to_activate].status = WorkerStatus.INACTIVE
            
    async def _drain_worker(self, worker_id: int):
        """Drain a worker by waiting for its requests to complete, then flush cache"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.drain_timeout:
            if worker_id in self.workers:
                active_requests = self.workers[worker_id].active_requests
                if active_requests == 0:
                    break
                    
            await asyncio.sleep(1.0)
            
        # Force flush cache and mark inactive
        try:
            await self._flush_worker_cache(worker_id)
            logger.info(f"Successfully drained and flushed worker {worker_id}")
        except Exception as e:
            logger.error(f"Failed to flush cache for worker {worker_id}: {e}")
            
        # Update worker status
        if worker_id in self.draining_workers:
            self.draining_workers.remove(worker_id)
        self.workers[worker_id].status = WorkerStatus.INACTIVE
        
    async def _flush_worker_cache(self, worker_id: int):
        """Flush cache for a specific worker"""
        if hasattr(self.worker_group, 'flush_worker_cache'):
            await self.worker_group.flush_worker_cache(worker_id)
        else:
            logger.warning(f"Worker group does not support cache flushing for worker {worker_id}")
        
    async def _resume_worker(self, worker_id: int):
        """Resume a worker that was previously inactive"""
        if hasattr(self.worker_group, 'resume_worker_memory'):
            await self.worker_group.resume_worker_memory(worker_id)
        else:
            logger.warning(f"Worker group does not support memory resumption for worker {worker_id}")
        
    async def _redistribute_requests(self):
        """Redistribute pending requests to active workers"""
        if not self.pending_requests:
            return
            
        # Simple round-robin distribution for now
        active_worker_list = list(self.active_workers)
        if not active_worker_list:
            logger.warning("No active workers available for request distribution")
            return
            
        requests_to_process = []
        while self.pending_requests and len(requests_to_process) < len(active_worker_list):
            requests_to_process.append(self.pending_requests.popleft())
            
        for i, request_info in enumerate(requests_to_process):
            worker_id = active_worker_list[i % len(active_worker_list)]
            request_info.worker_id = worker_id
            request_info.started_time = time.time()
            
        logger.debug(f"Redistributed {len(requests_to_process)} requests")
        
    def _get_total_active_requests(self) -> int:
        """Get total number of active requests across all workers"""
        total = 0
        for worker_id in self.active_workers:
            if worker_id in self.workers:
                total += self.workers[worker_id].active_requests
        return total
        
    def add_request(self, request_id: str) -> None:
        """Add a new request to the scheduler"""
        request_info = RequestInfo(request_id=request_id)
        self.active_requests[request_id] = request_info
        self.pending_requests.append(request_info)
        
    def complete_request(self, request_id: str) -> None:
        """Mark a request as completed"""
        if request_id in self.active_requests:
            request_info = self.active_requests[request_id]
            request_info.completed_time = time.time()
            self.request_history.append(request_info)
            del self.active_requests[request_id]
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring"""
        total_active = self._get_total_active_requests()
        total_capacity = len(self.active_workers) * self.config.requests_per_worker_capacity
        
        return {
            "active_workers": len(self.active_workers),
            "draining_workers": len(self.draining_workers),
            "total_workers": len(self.workers),
            "active_requests": total_active,
            "pending_requests": len(self.pending_requests),
            "load_ratio": total_active / max(total_capacity, 1),
            "total_capacity": total_capacity,
        } 