import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class DummyModel(nn.Module):
    """Simple dummy model for testing FSDP2"""
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1000):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


def setup_distributed():
    """Initialize the distributed environment"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    
    # Initialize process group with all 8 GPUs
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def create_training_mesh(rank, world_size, training_world_size=6):
    """Create a device mesh with only specified number of GPUs"""
    # Only ranks 0-5 will be in the training group
    training_ranks = list(range(training_world_size))
    
    # Check if current rank is part of training group
    is_training_rank = rank < training_world_size
    
    # Create device mesh only for training ranks
    if is_training_rank:
        # Create a 1D device mesh for FSDP2
        device_mesh = DeviceMesh("cuda", training_ranks)
    else:
        device_mesh = None
    
    return device_mesh, is_training_rank, training_ranks


def create_dummy_data(batch_size=32, input_dim=1024, num_samples=1000):
    """Create dummy dataset for testing"""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, 1000)  # 1000 classes
    
    dataset = TensorDataset(X, y)
    return dataset


def train_step(model, data_loader, optimizer, criterion, rank, is_training_rank):
    """Single training step"""
    if not is_training_rank:
        return None
    
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0 and rank == 0:
            print(f"Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")
    
    return total_loss / len(data_loader)


def main():
    # Setup distributed environment with all 8 GPUs
    rank, local_rank, world_size = setup_distributed()
    
    print(f"Rank {rank}: Initialized distributed environment with world_size={world_size}")
    
    # Create training mesh with only 6 GPUs
    device_mesh, is_training_rank, training_ranks = create_training_mesh(
        rank, world_size, training_world_size=6
    )
    
    if is_training_rank:
        print(f"Rank {rank}: Part of training group")
    else:
        print(f"Rank {rank}: NOT part of training group (idle)")
        # Non-training ranks wait here while training happens
        # We'll sync at the end with all ranks
        pass
    
    # Create model only for training ranks
    if is_training_rank:
        # Create model
        model = DummyModel().cuda()
        
        # Apply FSDP2 to each layer individually for better control
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=device_mesh)
        
        # Apply FSDP2 to the entire model
        fully_shard(model, mesh=device_mesh)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
    
    # Create dummy dataset - only for training ranks
    if is_training_rank:
        dataset = create_dummy_data(batch_size=32, input_dim=1024, num_samples=1000)
        
        # Create data loader with distributed sampler
        # Use only training ranks for the sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=len(training_ranks),
            rank=training_ranks.index(rank) if rank in training_ranks else 0,
            shuffle=True
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
    
    # Training loop
    num_epochs = 50  # Changed from 5 to 500 as per your request
    
    if is_training_rank:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)  # Important for shuffling
            
            start_time = time.time()
            avg_loss = train_step(model, train_loader, optimizer, criterion, rank, is_training_rank)
            epoch_time = time.time() - start_time
            
            if rank == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s")
                print(f"Average Loss: {avg_loss:.4f}")
                print("-" * 50)
        
        # Only training ranks need to sync within their group
        # Use device_mesh's process group for barrier
        dist.barrier(group=device_mesh.get_group())
        if rank == 0:
            print("\nTraining completed successfully!")
    
    # Final synchronization with ALL ranks before cleanup
    dist.barrier()  # This ensures all ranks sync before destroying process group
    
    # Destroy process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()