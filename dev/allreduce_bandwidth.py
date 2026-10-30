"""
NCCL all-reduce bandwidth across the local GPUs, to estimate the DDP gradient synchronization cost.

Usage: torchrun --nproc-per-node 8 dev/allreduce_bandwidth.py
"""

import os
import time

import torch
import torch.distributed as distributed

MESSAGE_MEGABYTES = (25, 100, 400, 1600)
REPEATS = 10


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl")
    world_size = distributed.get_world_size()
    for dtype in (torch.bfloat16, torch.float32):
        for megabytes in MESSAGE_MEGABYTES:
            element_count = megabytes * 1024**2 // torch.empty((), dtype=dtype).element_size()
            buffer = torch.ones(element_count, dtype=dtype, device="cuda")
            for _ in range(3):
                distributed.all_reduce(buffer)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(REPEATS):
                distributed.all_reduce(buffer)
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - start_time) / REPEATS
            bus_bandwidth = 2 * (world_size - 1) / world_size * megabytes / 1024 / seconds
            if local_rank == 0:
                print(
                    f"{str(dtype):15s} {megabytes:5d} MB  {1000 * seconds:8.1f} ms  "
                    f"bus bandwidth {bus_bandwidth:6.1f} GB/s",
                    flush=True,
                )
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
