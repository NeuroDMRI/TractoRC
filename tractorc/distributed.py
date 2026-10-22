"""Small single-node distributed-training helpers."""

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_master(self):
        return self.rank == 0


class DistributedEvalSampler(Sampler):
    """Shard evaluation indices across ranks, padding each shard to full batches.

    The shard is padded by wrapping around its own indices so that every batch
    has exactly ``batch_size`` examples: the compiled backbone then never sees
    a new input shape during validation (a partial last batch forces a
    recompilation, which has produced faulty kernels). The few duplicated
    pairs only affect the logged validation average.
    """

    def __init__(self, dataset, rank, world_size, batch_size=1):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.batch_size = max(1, int(batch_size))

    def _indices(self):
        indices = list(range(self.rank, len(self.dataset), self.world_size))
        remainder = len(indices) % self.batch_size
        if indices and remainder:
            pad = self.batch_size - remainder
            indices += [indices[i % len(indices)] for i in range(pad)]
        return indices

    def __iter__(self):
        return iter(self._indices())

    def __len__(self):
        return len(self._indices())


def initialize(configured_gpu, requested_local_rank=None):
    """Create the launch context, using NCCL whenever torchrun has >1 ranks."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_local_rank is not None and requested_local_rank != env_local_rank:
        raise ValueError(
            "--local-rank does not match LOCAL_RANK supplied by torchrun: "
            f"{requested_local_rank} != {env_local_rank}."
        )

    if world_size == 1:
        device = torch.device(
            f"cuda:{configured_gpu}" if torch.cuda.is_available() else "cpu"
        )
        return DistributedContext(0, 0, 1, device)

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA and an NCCL-capable PyTorch build.")
    if env_local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={env_local_rank} exceeds the {torch.cuda.device_count()} "
            "CUDA devices visible to this process. Set CUDA_VISIBLE_DEVICES first."
        )

    device = torch.device(f"cuda:{env_local_rank}")
    torch.cuda.set_device(device)
    # The K-means centroid initialization runs on rank 0 while the other ranks
    # wait in a barrier; on the full training set it can take 5+ hours, far
    # beyond NCCL's 10-minute default collective timeout.
    dist.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(hours=24)
    )
    return DistributedContext(rank, env_local_rank, world_size, device)


def destroy(context):
    if context is not None and context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context):
    if context.enabled:
        if context.device.type == "cuda":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def broadcast_model_state(model, context):
    """Copy rank-0 parameters and buffers to every worker."""
    if not context.enabled:
        return
    for value in unwrap_model(model).state_dict().values():
        if torch.is_tensor(value):
            dist.broadcast(value, src=0)


def all_gather_cat(tensor, context):
    """Autograd-aware all-gather followed by concatenation along batch axis."""
    if not context.enabled:
        return tensor
    return torch.cat(dist_functional.all_gather(tensor), dim=0)


def sum_across_ranks(values, device, context):
    """Sum scalar statistics and return them as Python floats."""
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if context.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.tolist()
