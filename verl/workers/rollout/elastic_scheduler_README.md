# Elastic Scheduler for SGLangRollout

## Overview

The Elastic Scheduler provides dynamic worker scaling for SGLangRollout engines, automatically adjusting the number of active workers based on request load to optimize resource utilization and memory usage.

## Key Features

- **Dynamic Scaling**: Automatically scale workers up/down based on request load
- **Memory Management**: Flush caches when scaling down to release GPU memory
- **Request Redistribution**: Seamlessly redistribute requests when workers change
- **Load Monitoring**: Continuous monitoring of request patterns and worker health
- **Safety Controls**: Configurable thresholds and cooldowns to prevent thrashing

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Worker 0      │    │   Worker 1      │    │   Worker 2      │
│   (Active)      │    │   (Draining)    │    │   (Inactive)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ Elastic Scheduler   │
                    │ - Load monitoring   │
                    │ - Scaling decisions │
                    │ - Request routing   │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Request Queue     │
                    │ ┌─────┐ ┌─────┐     │
                    │ │Req1 │ │Req2 │ ... │
                    │ └─────┘ └─────┘     │
                    └─────────────────────┘
```

## Configuration

```python
from verl.workers.rollout.elastic_scheduler import ElasticConfig, ElasticScheduler

config = ElasticConfig(
    # Scaling thresholds (load ratio)
    scale_down_threshold=0.5,  # Scale down when load < 50% capacity
    scale_up_threshold=0.8,    # Scale up when load > 80% capacity
    
    # Worker limits
    min_workers=1,             # Always keep at least 1 worker
    max_workers=8,             # Maximum workers to use
    
    # Capacity planning
    requests_per_worker_capacity=8,  # Expected requests per worker
    
    # Timing controls
    monitoring_interval=5.0,    # Check every 5 seconds
    drain_timeout=30.0,         # Max time to drain a worker
    scale_down_cooldown=60.0,   # Wait 60s between scale downs
    scale_up_cooldown=30.0,     # Wait 30s between scale ups
    min_stable_period=10.0,     # Require 10s stable load before scaling
    
    # Safety
    enable_elastic=True         # Master switch
)
```

## Worker States

### ACTIVE
- Worker is handling requests normally
- Participates in load balancing
- Counted in capacity calculations

### DRAINING  
- Worker is being prepared for shutdown
- No new requests assigned
- Existing requests continue processing
- Cache will be flushed when empty

### INACTIVE
- Worker is shut down
- Cache has been flushed
- Memory released for other tasks
- Can be reactivated if needed

### STARTING
- Worker is being reactivated
- Not yet ready for requests
- Transitioning to ACTIVE state

## Integration with SGLangRollout

### Step 1: Modify SGLangRollout Class

Add elastic scheduler support to the SGLangRollout initialization:

```python
# In sglang_rollout.py
from verl.workers.rollout.elastic_scheduler import ElasticScheduler, ElasticConfig

class SGLangRollout(BaseRollout):
    def __init__(self, ..., elastic_config: ElasticConfig = None):
        super().__init__()
        # ... existing initialization ...
        
        # Initialize elastic scheduler if enabled
        self.elastic_scheduler = None
        if elastic_config and elastic_config.enable_elastic:
            self.elastic_scheduler = ElasticScheduler(
                config=elastic_config,
                worker_group=self  # Pass self as worker group
            )
    
    async def start_elastic_scheduling(self):
        """Start elastic scheduling if enabled"""
        if self.elastic_scheduler:
            await self.elastic_scheduler.start()
            
    async def stop_elastic_scheduling(self):
        """Stop elastic scheduling"""
        if self.elastic_scheduler:
            await self.elastic_scheduler.stop()
```

### Step 2: Add Worker Status Tracking

Add methods to track worker status and request counts:

```python
# In SGLangRollout class
async def get_worker_status(self) -> Dict[int, Dict[str, Any]]:
    """Get status of all workers for elastic scheduler"""
    statuses = {}
    for worker_id in range(self.world_size):
        if worker_id == self._tp_rank:
            # Get local worker status
            active_requests = self._get_active_request_count()
            statuses[worker_id] = {
                'active_requests': active_requests,
                'total_processed': self._total_processed_requests,
                'status': 'active'  # or appropriate status
            }
    
    # Broadcast and collect from all workers
    all_statuses = self._collect_worker_statuses(statuses)
    return all_statuses

def _get_active_request_count(self) -> int:
    """Get number of active requests on this worker"""
    # Implementation depends on SGLang's internal request tracking
    # This is a placeholder
    return 0
```

### Step 3: Implement Cache Management

Add methods for cache flushing during scaling:

```python
# In SGLangRollout class  
async def flush_worker_cache(self, worker_id: int):
    """Flush cache for a specific worker"""
    if worker_id == self._tp_rank and self._engine is not None:
        await self._engine.flush_cache()
        logger.info(f"Flushed cache for worker {worker_id}")

async def resume_worker_memory(self, worker_id: int):
    """Resume worker memory occupation"""
    if worker_id == self._tp_rank and self._engine is not None:
        await self._engine.resume_memory_occupation()
        logger.info(f"Resumed memory for worker {worker_id}")
```

### Step 4: Integrate with Ray Trainer

Modify the RayPPOTrainer to use elastic scheduling:

```python
# In ray_trainer.py
def init_workers(self):
    # ... existing worker initialization ...
    
    # Add elastic scheduling for rollout workers
    elastic_config = self.config.get('elastic_config', None)
    if elastic_config:
        self.actor_rollout_wg.start_elastic_scheduling()

async def _validate(self):
    # Wake up elastic scheduler before validation
    if hasattr(self.actor_rollout_wg, 'elastic_scheduler'):
        await self.actor_rollout_wg.wake_up_all_workers()
    
    # ... existing validation logic ...
    
    # Let scheduler resume normal operation
    if hasattr(self.actor_rollout_wg, 'elastic_scheduler'):
        await self.actor_rollout_wg.resume_elastic_scheduling()
```

## Example Usage

### Basic Configuration

```yaml
# In your training config
elastic_config:
  enable_elastic: true
  scale_down_threshold: 0.4  # Scale down when load < 40%
  scale_up_threshold: 0.8    # Scale up when load > 80%
  min_workers: 2             # Always keep 2 workers minimum
  max_workers: 8             # Use up to 8 workers maximum
  requests_per_worker_capacity: 10
  monitoring_interval: 3.0
  scale_down_cooldown: 120.0  # Wait 2 minutes between scale downs
```

### Integration Example

```python
# In your training script
from verl.workers.rollout.elastic_scheduler import ElasticConfig

# Create elastic config
elastic_config = ElasticConfig(
    scale_down_threshold=0.5,
    scale_up_threshold=0.8,
    min_workers=1,
    max_workers=4,
    enable_elastic=True
)

# Pass to trainer
trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    elastic_config=elastic_config  # Add this parameter
)

# Training will automatically use elastic scaling
trainer.fit()
```

## Monitoring and Metrics

The elastic scheduler provides metrics for monitoring:

```python
# Get current metrics
metrics = scheduler.get_metrics()
print(f"Active workers: {metrics['active_workers']}")
print(f"Load ratio: {metrics['load_ratio']:.2f}")
print(f"Total requests: {metrics['active_requests']}")
```

Available metrics:
- `active_workers`: Number of currently active workers
- `draining_workers`: Number of workers being drained
- `total_workers`: Total number of workers in pool
- `active_requests`: Current number of active requests
- `pending_requests`: Number of queued requests
- `load_ratio`: Current load as ratio of capacity (0.0-1.0+)
- `total_capacity`: Maximum request capacity

## Best Practices

### Threshold Selection
- Set `scale_down_threshold` low enough to release resources when truly idle
- Set `scale_up_threshold` high enough to avoid unnecessary scaling
- Leave gap between thresholds to prevent oscillation

### Timing Parameters
- Use longer `scale_down_cooldown` than `scale_up_cooldown`
- Set `min_stable_period` to avoid reacting to temporary spikes
- Adjust `monitoring_interval` based on your workload patterns

### Capacity Planning
- Set `requests_per_worker_capacity` based on your hardware and model
- Monitor actual throughput to tune this parameter
- Consider memory constraints when setting `max_workers`

### Safety Considerations
- Always set `min_workers >= 1` to avoid complete shutdown
- Test thoroughly before production use
- Monitor for request timeouts during scaling events
- Have manual override capabilities

## Troubleshooting

### Common Issues

1. **Workers not scaling down**
   - Check if requests are actually completing
   - Verify `scale_down_threshold` is appropriate
   - Check cooldown periods

2. **Memory not being released**
   - Ensure cache flush is working properly
   - Check for memory leaks in worker processes
   - Verify GPU memory monitoring

3. **Request redistribution failures**  
   - Check worker health monitoring
   - Verify request routing logic
   - Monitor for dropped requests

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('verl.workers.rollout.elastic_scheduler').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Predictive Scaling**: Use request patterns to predict scaling needs
- **Multi-tier Scaling**: Different scaling policies for different request types
- **Cross-node Scaling**: Scale across multiple nodes, not just workers
- **Integration with Kubernetes**: Auto-scaling at pod level
- **Advanced Load Balancing**: Consider request complexity, not just count 