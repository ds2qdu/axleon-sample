"""TrainJob sample that keeps the GPU busy for MIN_SECONDS on a synthetic regression.

torchrun sets RANK / WORLD_SIZE / LOCAL_RANK; DDP is used only when WORLD_SIZE > 1.
No dataset download: the model learns y = sum(x) / sqrt(dim) on random inputs.
"""
import math
import os
import socket
import time

import torch
import torch.distributed as dist
from torch import nn


def env_int(name, default):
    return int(os.environ.get(name, default))


def build_model(dim, hidden, layers):
    blocks = [nn.Linear(dim, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
    blocks.append(nn.Linear(hidden, 1))
    return nn.Sequential(*blocks)


def main():
    world = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    epochs = env_int("EPOCHS", 5)
    steps = env_int("STEPS_PER_EPOCH", 100)
    batch = env_int("BATCH_SIZE", 4096)
    dim = env_int("INPUT_DIM", 1024)
    hidden = env_int("HIDDEN", 4096)
    layers = env_int("LAYERS", 4)
    min_seconds = env_int("MIN_SECONDS", 120)

    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")

    torch.manual_seed(rank)
    model = build_model(dim, hidden, layers).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if cuda else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scale = math.sqrt(dim)

    gpu = torch.cuda.get_device_name(device) if cuda else "cpu"
    print(f"rank {rank}/{world} host {socket.gethostname()} device {device} ({gpu})", flush=True)
    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"model {layers}x{hidden} params {params / 1e6:.1f}M batch {batch} steps/epoch {steps} "
              f"min epochs {epochs} min seconds {min_seconds}", flush=True)

    started = time.time()
    history = []
    epoch = 0
    while True:
        epoch += 1
        epoch_started = time.time()
        total = 0.0
        for _ in range(steps):
            x = torch.randn(batch, dim, device=device)
            y = x.sum(dim=1, keepdim=True) / scale
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
        elapsed = time.time() - started
        if rank == 0:
            mem = f" gpu_mem {torch.cuda.max_memory_allocated(device) / 2**20:.0f}MiB" if cuda else ""
            print(f"epoch {epoch} loss {avg.item():.4f} ({time.time() - epoch_started:.1f}s, total {elapsed:.0f}s){mem}",
                  flush=True)
        more = torch.tensor(int(epoch < epochs or elapsed < min_seconds), device=device)
        if world > 1:
            dist.all_reduce(more, op=dist.ReduceOp.MAX)
        if not more.item():
            break

    if rank == 0 and os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow

            with mlflow.start_run(run_name=os.environ.get("KUBE_TRAINJOB_NAME", "train-sample")):
                mlflow.log_params({"world_size": world, "epochs": len(history), "steps_per_epoch": steps,
                                   "batch_size": batch, "hidden": hidden, "layers": layers})
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
