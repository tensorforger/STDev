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

Run `scripts/download_vaes.sh` (reads `HF_TOKEN` from `.env` for gated models), or run the commands below manually (after `hf auth login` for gated models).

```bash
cd models

hf download black-forest-labs/FLUX.2-klein-4B --include "vae/*" --local-dir FLUX.2-klein-4B
hf download madebyollin/texture-fix-vae-for-qwen-image-2.1 --include "config.json" "diffusion_pytorch_model.safetensors" --local-dir texture-fix-vae-for-qwen-image-2.1
hf download stabilityai/sdxl-vae --include "config.json" "diffusion_pytorch_model.safetensors" --local-dir sdxl-vae
hf download black-forest-labs/FLUX.1-schnell --include "vae/*" --local-dir FLUX.1-schnell
hf download stabilityai/stable-diffusion-3.5-large --include "vae/*" --local-dir stable-diffusion-3.5-large
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*" --local-dir Wan2.1-T2V-1.3B-Diffusers
hf download Lightricks/LTX-2.5-Diffusers --include "vae/*" --local-dir LTX-2.5-Diffusers
hf download MiniMaxAI/MiniMax-H3 --include "vae/*" --local-dir MiniMax-H3
```

## Usage

### Generate FLUX datasets

`scripts/generate_flux_dataset.py` shards samples across GPUs with `torchrun` (rank `r` handles every `WORLD_SIZE`-th sample). Already existing files are skipped, so interrupted runs can be resumed by re-running the same command.

```bash
# benchmark dataset (1000 samples) -> benchmark_datasets/flux_1024
torchrun --standalone --nproc_per_node=8 scripts/generate_flux_dataset.py --benchmark

# SFT training dataset (resolution buckets in stdev.sft.config.RESOLUTION_BUCKETS) -> train_datasets/flux_sft_buckets
torchrun --standalone --nproc_per_node=8 scripts/generate_flux_dataset.py

# tiny dataset for a quick check -> dev/temporary/sft_smoke_dataset
python scripts/generate_flux_dataset.py --smoke --output-directory dev/temporary/sft_smoke_dataset
```

Without `torchrun` the script runs on a single GPU (`python scripts/generate_flux_dataset.py`). Use `CUDA_VISIBLE_DEVICES` with a smaller `--nproc_per_node` to use a subset of GPUs.

Each training block (`<bucket>/block_NNNNNN.vz.safetensors`) holds one FLUX.2-klein-4B teacher sample (4 steps) at that bucket's resolution: `latents_orig` (high resolution latents), `latents_div_2` (their bilinear 2x downscale, decoded and re-encoded), `text_embedding` (mean pooled SDXL text embedding of the prompt) and `high_frequency_raw` (raw, not quantile normalized, high frequency features of the decoded image).

### SFT the latent upscaler

`scripts/run_sft.py` trains `stdev.upscalers.pretrain_large.UpscalerUNet` on a dataset generated above, single GPU, no online data generation: it samples one resolution bucket per batch, rectified flow between the high resolution latents and noise conditioned on the low resolution latents, the pooled text embedding and the quantile normalized high frequency features (quantiles are fitted once from the dataset and cached as `<dataset>/high_frequency_quantiles.json`).

```bash
python scripts/run_sft.py --run-name NAME [--init-checkpoint PATH]

# smoke test on the tiny dataset above
python scripts/run_sft.py --run-name smoke --dataset-directory dev/temporary/sft_smoke_dataset --max-steps 2
```

Checkpoints, EMA/raw exported weights and tensorboard logs are written under `sft_runs/NAME`.


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
