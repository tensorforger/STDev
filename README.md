# STDev

### Environment

```bash
uv venv --python=3.12
source .venv/bin/activate
# adjust to your CUDA version
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu134
uv pip install -e .
```

### Models

```bash
cd models

hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B
hf download stabilityai/stable-diffusion-xl-base-1.0 --include "text_encoder/*" --include "tokenizer/*" --local-dir stable-diffusion-xl-base-1.0
hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir MOSS-Image-SFT
```


### Datasets
```bash
cd datasets
git clone https://huggingface.co/datasets/refine-axis/axis-v1_1k
```

## Usage

### Generate FLUX datasets

`scripts/generate_flux_dataset.py` shards samples across GPUs with `torchrun` (rank `r` handles every `WORLD_SIZE`-th sample). Already existing files are skipped, so interrupted runs can be resumed by re-running the same command.

```bash
# benchmark dataset (1000 samples) -> benchmark_datasets/flux_1024
torchrun --standalone --nproc_per_node=8 scripts/generate_flux_dataset.py --benchmark

# training dataset (20k blocks) -> train_datasets/flux_1024_20k
torchrun --standalone --nproc_per_node=8 scripts/generate_flux_dataset.py
```

Without `torchrun` the script runs on a single GPU (`python scripts/generate_flux_dataset.py`). Use `CUDA_VISIBLE_DEVICES` with a smaller `--nproc_per_node` to use a subset of GPUs.


## License

See [LICENSE](LICENSE). Datasets and model weights are subject to their own licenses.
