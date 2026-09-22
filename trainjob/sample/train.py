"""Minimal training sample for TrainJob.

The same file covers one node with one GPU, one node with several GPUs, and several nodes:
torchrun sets RANK / WORLD_SIZE / LOCAL_RANK, and DDP is used only when WORLD_SIZE > 1.
No dataset download: the model learns y = sum(x) on synthetic data.
"""
import os
import socket
import time

import torch
import torch.distributed as dist
from torch import nn


def env_int(name, default):
    return int(os.environ.get(name, default))


def main():
    world = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    epochs = env_int("EPOCHS", 3)
    steps = env_int("STEPS_PER_EPOCH", 50)
    batch = env_int("BATCH_SIZE", 64)

    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")

    torch.manual_seed(rank)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if cuda else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    print(f"rank {rank}/{world} host {socket.gethostname()} device {device}", flush=True)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        total = 0.0
        for _ in range(steps):
            x = torch.randn(batch, 32, device=device)
            y = x.sum(dim=1, keepdim=True)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = torch.tensor(total / steps, device=device)
        if world > 1:
            dist.all_reduce(avg)
            avg /= world
        history.append(avg.item())
        if rank == 0:
            print(f"epoch {epoch}/{epochs} loss {avg.item():.4f} ({time.time() - started:.1f}s)", flush=True)

    if rank == 0 and os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow

            with mlflow.start_run(run_name=os.environ.get("KUBE_TRAINJOB_NAME", "train-sample")):
                mlflow.log_params({"world_size": world, "epochs": epochs, "steps_per_epoch": steps, "batch_size": batch})
                for i, value in enumerate(history, start=1):
                    mlflow.log_metric("loss", value, step=i)
        except ImportError:
            print("mlflow not installed; skipping tracking", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print("done", flush=True)


if __name__ == "__main__":
    main()
