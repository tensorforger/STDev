#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set (add it to ${ENV_FILE}); gated models will fail to download." >&2
fi

mkdir -p "${REPO_ROOT}/models"
cd "${REPO_ROOT}/models"

hf download black-forest-labs/FLUX.2-klein-4B --include "vae/*" --local-dir FLUX.2-klein-4B
hf download madebyollin/texture-fix-vae-for-qwen-image-2.1 --include "config.json" "diffusion_pytorch_model.safetensors" --local-dir texture-fix-vae-for-qwen-image-2.1
hf download stabilityai/sdxl-vae --include "config.json" "diffusion_pytorch_model.safetensors" --local-dir sdxl-vae
hf download black-forest-labs/FLUX.1-schnell --include "vae/*" --local-dir FLUX.1-schnell
hf download stabilityai/stable-diffusion-3.5-large --include "vae/*" --local-dir stable-diffusion-3.5-large
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*" --local-dir Wan2.1-T2V-1.3B-Diffusers
hf download Lightricks/LTX-2.5-Diffusers --include "vae/*" --local-dir LTX-2.5-Diffusers
hf download MiniMaxAI/MiniMax-H3 --include "vae/*" --local-dir MiniMax-H3
