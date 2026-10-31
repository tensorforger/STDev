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


### Upload checkpoints to S3 (Cloudflare R2)

Put credentials in `.env` at the repo root (gitignored):

```bash
S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
S3_ACCESS_KEY_ID=<access key id>
S3_SECRET_ACCESS_KEY=<sha256 of the R2 API token value>
```

```bash
# newest checkpoint under pretrain_runs/ -> s3://checkpoints/<run_name>/<checkpoint>.pt
python scripts/upload_checkpoint.py
# newest checkpoint of a specific run, or an exact file / bucket / key
python scripts/upload_checkpoint.py pretrain_runs/stage1_v1
python scripts/upload_checkpoint.py pretrain_runs/stage1_v1/checkpoints/checkpoint_000056599.pt --bucket checkpoints --key stage1_v1/final.pt
```

Uploads are skipped if an object with the same key and size already exists (`--force` to overwrite).

## License

See [LICENSE](LICENSE). Datasets and model weights are subject to their own licenses.
