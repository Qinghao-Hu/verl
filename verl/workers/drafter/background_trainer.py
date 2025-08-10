import logging
import os

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class FSDPDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2048, 4096)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(4096, 8192)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.1)
        self.fc3 = torch.nn.Linear(8192, 4096)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(0.1)
        self.fc4 = torch.nn.Linear(4096, 2048)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(0.1)
        self.fc5 = torch.nn.Linear(2048, 1024)
        self.relu5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(1024, 512)
        self.relu6 = torch.nn.ReLU()
        self.fc7 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.dropout4(self.relu4(self.fc4(x)))
        x = self.relu5(self.fc5(x))
        x = self.relu6(self.fc6(x))
        x = self.fc7(x)
        return x


class BackgroundTrainer:
    """Handles FSDP2 distributed training for background training during memory release."""

    def __init__(self, rank: int):
        self.rank = rank

        # Training model attributes
        self.model = None
        self.optimizer = None
        self.training_group = None
        self.training_device_mesh = None

        # Training state
        self._training_initialized = False
        self._training_active = False

    async def initialize_training_model(self, device_mesh, training_ranks: list[int]) -> bool:
        """Initialize training model with FSDP2 if multiple workers, else independent training."""
        # Avoid re-initialization
        if self._training_initialized:
            logger.info(f"Worker {self.rank} model already initialized, skipping re-initialization")
            return True

        logger.info(f"Worker {self.rank} starting model initialization with process ranks: {training_ranks}")

        # Create model on the correct device
        model = FSDPDummyModel().cuda()

        # Ensure all parameters require gradients BEFORE FSDP
        for param in model.parameters():
            param.requires_grad_(True)

        # For single worker (no TP partners), don't use FSDP
        if len(training_ranks) == 1:
            logger.info(f"Worker {self.rank} using single GPU training without FSDP")
            # No FSDP wrapping for single GPU
            self.using_fsdp = False
        else:
            logger.info(f"Worker {self.rank} using FSDP2 with {len(training_ranks)} workers on mesh {device_mesh}")
            # Apply FSDP2 with the device mesh for multi-GPU training
            fully_shard(model, mesh=device_mesh)
            self.using_fsdp = True

            # Log the process group for verification
            pg = device_mesh.get_group()
            pg_ranks = dist.get_process_group_ranks(pg) if hasattr(dist, "get_process_group_ranks") else training_ranks
            logger.info(f"Worker {self.rank} FSDP2 process group ranks: {pg_ranks}")

        # Verify parameters still require gradients
        params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        logger.info(f"Worker {self.rank}: {params_requiring_grad}/{total_params} parameters require grad after setup")

        if params_requiring_grad == 0:
            logger.error(f"Worker {self.rank}: No parameters require grad! Forcing grad requirement...")
            for param in model.parameters():
                param.requires_grad_(True)
            params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
            logger.info(
                f"Worker {self.rank}: After forcing: {params_requiring_grad}/{total_params} parameters require grad"
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95))

        # Store as instance variables
        self.model = model
        self.optimizer = optimizer
        self.training_device_mesh = device_mesh
        self._training_initialized = True

        logger.info(
            f"Worker {self.rank} training model initialized successfully "
            f"(training with {len(training_ranks)} total process ranks)"
        )
        return True

    async def training_step(self, step: int, num_released_workers: int) -> bool:
        """Perform one training step with optimized batch size for multi-GPU training."""
        # IMPORTANT: Re-enable gradients since we're called from a no_grad context
        with torch.enable_grad():
            return await self._training_step_impl(step, num_released_workers)

    async def _training_step_impl(self, step: int, num_released_workers: int) -> bool:
        """Implementation of training step with gradients enabled."""
        if step == 0:
            logger.info(f"Worker {self.rank} starting first training step")

        # Check if model is initialized
        if self.model is None:
            logger.error(f"Worker {self.rank} step {step}: Model not initialized!")
            return False

        # Ensure model is in training mode
        self.model.train()

        # Check and log parameter gradient status at first step
        if step == 0:
            params_with_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
            total_params = sum(1 for p in self.model.parameters())
            logger.info(f"Worker {self.rank} step {step}: {params_with_grad}/{total_params} parameters require grad")
            if params_with_grad == 0:
                logger.error(f"Worker {self.rank}: No parameters require grad! Model might be frozen.")
                # Try to unfreeze
                for param in self.model.parameters():
                    param.requires_grad_(True)
                params_with_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
                logger.info(f"Worker {self.rank}: After unfreezing: {params_with_grad}/{total_params} require grad")

        # Synchronize workers in FSDP group
        if getattr(self, "using_fsdp", False) and self.training_device_mesh is not None:
            pg = self.training_device_mesh.get_group()
            dist.barrier(group=pg)
            if step == 0:
                logger.info(f"Worker {self.rank} synchronized with FSDP2 group for training")

        # Use larger batch size for better GPU utilization
        base_batch_size = 128
        batch_size = base_batch_size * max(1, num_released_workers // 2)  # Scale with workers

        device = next(self.model.parameters()).device
        # Use larger input dimension to match the updated model
        if step == 0:
            logger.info(f"Worker {self.rank} training on device: {device}")
        x = torch.randn(batch_size, 2048, device=device, requires_grad=False)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass with gradient checking
        self.optimizer.zero_grad()

        # Enable gradient computation explicitly
        with torch.set_grad_enabled(True):
            outputs = self.model(x)
            loss = torch.nn.functional.cross_entropy(outputs, y)

        if step == 0:
            logger.info(
                f"Worker {self.rank} step {step}: Loss computed: {loss.item()}, requires_grad: {loss.requires_grad}"
            )
            logger.info(f"Worker {self.rank} step {step}: Output requires_grad: {outputs.requires_grad}")
            # Check a few model parameters
            param_count = 0
            for name, param in self.model.named_parameters():
                if param_count < 3:  # Log first 3 parameters
                    logger.info(
                        f"Worker {self.rank}: Param {name} shape={param.shape}, requires_grad={param.requires_grad}"
                    )
                param_count += 1

        # Check if loss requires grad
        if not loss.requires_grad:
            logger.error(f"Worker {self.rank} step {step}: Loss does not require grad!")

            # Debug: Check if we're in no_grad mode
            logger.error(f"Worker {self.rank}: torch.is_grad_enabled() = {torch.is_grad_enabled()}")

            # Try a simple test to see if gradients work at all
            test_tensor = torch.randn(2, 2, device=device, requires_grad=True)
            test_loss = test_tensor.sum()
            logger.info(f"Worker {self.rank}: Test tensor loss requires_grad = {test_loss.requires_grad}")

            return False

        # Backward pass
        loss.backward()

        # Check if gradients were computed
        has_grads = any(param.grad is not None for param in self.model.parameters())

        if not has_grads:
            logger.error(f"Worker {self.rank} step {step}: No gradients computed for any parameter!")
            return False

        # Gradient clipping for stability in large-scale training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimization step
        self.optimizer.step()

        if step % 10 == 0:  # Log every 10 steps for better visibility
            # Check if we're using FSDP2 (multi-GPU) or single GPU
            group_info = "FSDP2" if getattr(self, "using_fsdp", False) else "single-GPU"

            # Get gradient norm for debugging
            total_norm = 0.0
            param_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            total_norm = total_norm**0.5

            logger.info(
                f"Worker {self.rank} training step {step} SUCCESS, "
                f"loss: {loss.item():.4f}, grad_norm: {total_norm:.4f}, "
                f"batch_size: {batch_size}, "
                f"gradients: {param_count}, "
                f"released_workers: {num_released_workers}, "
                f"training_mode: {group_info}"
            )

        return True

    async def cleanup_training(self):
        """Clean up training resources."""
        logger.info(f"Worker {self.rank} starting training cleanup")
        
        # Clean up model
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "optimizer") and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if hasattr(self, "training_group"):
            self.training_group = None
        if hasattr(self, "training_device_mesh"):
            self.training_device_mesh = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._training_initialized = False
        self._training_active = False
        
        logger.info(f"Worker {self.rank} training cleanup complete")
    
    @property
    def is_training_initialized(self) -> bool:
        """Check if training is initialized."""
        return self._training_initialized
    
    @property
    def is_training_active(self) -> bool:
        """Check if training is active."""
        return self._training_active
    
    def set_training_active(self, active: bool):
        """Set training active state."""
        self._training_active = active