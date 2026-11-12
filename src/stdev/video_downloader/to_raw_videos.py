from yt_dlp import YoutubeDL
import os
import json
import sys
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig


import shutil


class ToRawVideosPipeline:
    def __init__(self, cfg: DictConfig):
        dataset_name = cfg.dataset_name
        self.dataset_root = f"./datasets/{dataset_name}"
        self.source_urls_root = os.path.join(self.dataset_root, "source_urls")
        self.output_root = os.path.join(self.dataset_root, "raw_videos")
        self.download_dir = os.path.join(self.dataset_root, "temp_downloads")

        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

    def _load_json_safe(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading JSON {filepath}: {e}")
            return None

    def _save_json_safe(self, filepath, data):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving JSON {filepath}: {e}")
            return False

    def download_video(self, url_json_filename):
        full_json_path = os.path.join(self.source_urls_root, url_json_filename)

        data = self._load_json_safe(full_json_path)
        if data is None:
            return

        if data.get("status") != "PENDING":
            print(
                f"Skipping {url_json_filename}, status is already {data.get('status', 'UNKNOWN')}"
            )
            return

        url = data.get("url")
        if not url:
            print(f"No URL found in {url_json_filename}")
            return

        data["status"] = "DOWNLOADING"
        if not self._save_json_safe(full_json_path, data):
            return

        base_name = os.path.splitext(url_json_filename)[0]
        output_filename = f"{base_name}.mp4"
        temp_path = os.path.join(self.download_dir, output_filename)
        final_path = os.path.join(self.output_root, output_filename)

        ydl_opts = {
            "cookies": "cookies.txt",
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": temp_path,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if os.path.exists(temp_path):
                shutil.move(temp_path, final_path)
                data["status"] = "COMPLETED"
            else:
                raise FileNotFoundError(f"Downloaded file not found: {temp_path}")

        except Exception as e:
            data["status"] = "ERROR"
            data["error_details"] = str(e)
            print(f"Error downloading {url_json_filename}: {e}")
        finally:
            self._save_json_safe(full_json_path, data)

    def run(self):
        try:
            filenames = [
                f for f in os.listdir(self.source_urls_root) if f.endswith(".json")
            ]

            if not filenames:
                print(f"No JSON files found in {self.source_urls_root}")
                return

            for filename in tqdm(filenames, desc="Processing videos"):
                try:
                    self.download_video(filename)
                except Exception as e:
                    print(f"Error while processing {filename}: {e}")
        finally:
            if os.path.exists(self.download_dir):
                shutil.rmtree(self.download_dir, ignore_errors=True)


def main():
    cfg = OmegaConf.load(sys.argv[1])
    pipeline = ToRawVideosPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
