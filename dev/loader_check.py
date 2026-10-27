import time

from torch.utils.data import DataLoader

from stdev.upscalers import run_0000

data_loader = DataLoader(
    run_0000.LatentPairDataset(run_0000.DATASET_DIRECTORY),
    batch_size=run_0000.BATCH_SIZE,
    shuffle=True,
    num_workers=run_0000.NUM_WORKERS,
    drop_last=True,
)
iterator = iter(data_loader)
next(iterator)
start_time = time.perf_counter()
for _ in range(50):
    next(iterator)
print(f"loader only: {50 / (time.perf_counter() - start_time):.1f} batches/s")
