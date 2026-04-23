from __future__ import annotations

import os

import torch
import torch.distributed as dist


def distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not distributed_available():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not distributed_available():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode() -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return True, local_rank, world_size


def cleanup_distributed() -> None:
    if distributed_available():
        dist.barrier()
        dist.destroy_process_group()


def reduce_scalar(value: float, device: torch.device) -> float:
    if not distributed_available():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return float(tensor.item())
