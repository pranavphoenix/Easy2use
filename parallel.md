Parallel processing

```python


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

assert torch.cuda.is_available(), "GPU not available"
assert <global batch size> % dist.get_world_size() == 0, "batch size not divisible"
rank = dist.get_rank()
device = rank % torch.cuda.device_count()
torch.cuda.set_device(device)

model = SimpleModel().to(device)
ddp_model = DDP(model, device_ids=[rank])

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_data_loaders(batch_size, rank, world_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    return loader


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)


def train(rank, world_size, epochs=5, batch_size=256):
    setup(rank, world_size)

    print(f"Rank {rank}: Starting training...")

    

    optimizer = optim.AdamW(ddp_model.parameters(), fused=True)  # Fused AdamW if available
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    train_loader = get_data_loaders(batch_size, rank, world_size)

    for epoch in range(epochs):
        ddp_model.train()
        train_loader.sampler.set_epoch(epoch)  # Important for shuffling

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)

            with autocast(device_type='cuda'):
                outputs = ddp_model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    WORLD_SIZE = torch.cuda.device_count()  # Automatically detects number of GPUs
    EPOCHS = 5
    BATCH_SIZE = 512  # Total batch size across all GPUs

    print(f"Detected {WORLD_SIZE} GPUs. Starting DDP training...")

    mp.spawn(
        train,
        args=(WORLD_SIZE, EPOCHS, BATCH_SIZE),
        nprocs=WORLD_SIZE,
        join=True
    )

```
