# torchrun --nproc_per_node=8 test_dist.py
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SmallDummyModel(nn.Module):
    """Smaller dummy model for the second training job"""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=500):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def setup_distributed():
    """Initialize the distributed environment"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 8))

    # Initialize process group with all 8 GPUs
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def create_training_meshes(rank, world_size):
    """Create device meshes for two separate training groups"""
    # First training group: ranks 0-5
    training_group1_ranks = list(range(6))
    # Second training group: ranks 6-7
    training_group2_ranks = list(range(6, 8))

    # Determine which group this rank belongs to
    is_group1 = rank < 6
    is_group2 = rank >= 6

    mesh1 = DeviceMesh("cuda", training_group1_ranks)  # ① 创建组 A
    mesh2 = DeviceMesh("cuda", training_group2_ranks)  # ② 创建组 B
    device_mesh = mesh1 if dist.get_rank() in training_group1_ranks else mesh2
    if is_group1:
        # device_mesh = DeviceMesh("cuda", training_group1_ranks)
        group_id = 1
        group_rank = rank
    elif is_group2:
        # device_mesh = DeviceMesh("cuda", training_group2_ranks)
        group_id = 2
        group_rank = rank - 6  # Normalize rank within group
    else:
        device_mesh = None
        group_id = None
        group_rank = None

    return device_mesh, group_id, group_rank, training_group1_ranks, training_group2_ranks


def create_dummy_data(batch_size=32, input_dim=1024, num_samples=1000):
    """Create dummy dataset for testing"""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, 1000)  # 1000 classes

    dataset = TensorDataset(X, y)
    return dataset


def create_small_dummy_data(batch_size=16, input_dim=512, num_samples=500):
    """Create smaller dummy dataset for second training job"""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, 500)  # 500 classes

    dataset = TensorDataset(X, y)
    return dataset


def train_step(model, data_loader, optimizer, criterion, rank, group_id, epoch):
    """Single training step"""
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

        if batch_idx % 10 == 0 and rank in [0, 6]:  # Print from first rank of each group
            print(
                f"[Group {group_id}] Rank {rank}: Epoch {epoch + 1}, Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}"
            )

    return total_loss / len(data_loader)


def main():
    # Setup distributed environment with all 8 GPUs
    rank, local_rank, world_size = setup_distributed()

    print(f"Rank {rank}: Initialized distributed environment with world_size={world_size}")

    # Create training meshes for two groups
    device_mesh, group_id, group_rank, group1_ranks, group2_ranks = create_training_meshes(rank, world_size)

    print(f"Rank {rank}: Part of training group {group_id}")

    local_device = torch.cuda.current_device()
    print(f"{local_device=}, {dist.get_rank()}")

    # Create models based on group membership
    if group_id == 1:
        # First group uses larger model
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

        # Create dataset
        dataset = create_dummy_data(batch_size=32, input_dim=1024, num_samples=1000)

        # Create data loader with distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=len(group1_ranks), rank=group_rank, shuffle=True
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

    elif group_id == 2:
        # Second group uses smaller model
        model = SmallDummyModel().cuda()

        # Apply FSDP2 to each layer individually for better control
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=device_mesh)

        # Apply FSDP2 to the entire model
        fully_shard(model, mesh=device_mesh)

        # Create optimizer with different learning rate
        optimizer = Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()

        # Create smaller dataset
        dataset = create_small_dummy_data(batch_size=16, input_dim=512, num_samples=500)

        # Create data loader with distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=len(group2_ranks), rank=group_rank, shuffle=True
        )

        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=2, pin_memory=True)
    
    # Training loop
    num_epochs = 500

    # Different number of epochs for each group if desired
    if group_id == 2:
        num_epochs = 30  # Smaller model trains for fewer epochs

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling

        start_time = time.time()
        avg_loss = train_step(model, train_loader, optimizer, criterion, rank, group_id, epoch)
        epoch_time = time.time() - start_time

        # Print summary from first rank of each group
        if (group_id == 1 and rank == 0) or (group_id == 2 and rank == 6):
            print(f"\n[Group {group_id}] Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f}s")
            print(f"[Group {group_id}] Average Loss: {avg_loss:.4f}")
            print("-" * 60)

    # Synchronize within each training group
    dist.barrier(group=device_mesh.get_group())

    if group_id == 1 and rank == 0:
        print(f"\nTraining Group 1 (GPUs 0-5) completed successfully!")
    elif group_id == 2 and rank == 6:
        print(f"\nTraining Group 2 (GPUs 6-7) completed successfully!")
    
    # Final synchronization with ALL ranks before cleanup
    dist.barrier()
    
    # Destroy process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()