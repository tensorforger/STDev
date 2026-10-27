import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTIONS_PATH = REPOSITORY_ROOT / "dev" / "captions_train2017.json"
OUTPUT_PATH = REPOSITORY_ROOT / "prompts" / "base_prompts.json"


def main() -> None:
    annotations = json.loads(CAPTIONS_PATH.read_text())["annotations"]
    base_prompts = list(dict.fromkeys(annotation["caption"].strip() for annotation in annotations))
    OUTPUT_PATH.write_text(json.dumps(base_prompts, ensure_ascii=False))
    print(f"Saved {len(base_prompts)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
