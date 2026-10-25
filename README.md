# STDev

### Environment

```bash
uv venv --python=3.12
source .venv/bin/activate
# adjust to your CUDA version
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu134
uv pip install -e .

# additional
git clone https://github.com/tensorforger/rqi
uv pip install ./rqi
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

```bash

```


## License

See [LICENSE](LICENSE). Datasets and model weights are subject to their own licenses.
