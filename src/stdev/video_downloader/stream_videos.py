"""
Stream processing of a video dataset: download -> cut -> save/upload, without intermediate stages.

Usage:
    python -m stdev.video_downloader.stream_videos config.yaml [--s3]

Without --s3, clips are written to datasets/<dataset_name>/videos (full videos are kept in raw_videos).
With --s3, clips are encoded in memory and uploaded to the bucket named <dataset_name>
(created if missing, credentials from .env), and full videos are deleted after cutting.
"""

import argparse
import json
import os
import queue
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yt_dlp import YoutubeDL

from stdev.utils.image_tools import crop_maximal_rectangle
from stdev.video_downloader.cut_videos import compute_spread_starts

REQUIRED_S3_VARIABLES = ("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
MISSING_BUCKET_CODES = ("404", "NoSuchBucket", "NotFound")
STARTABLE_STATUSES = ("PENDING", "DOWNLOADING")
MAX_RETRY_DELAY_SECONDS = 60


def load_env_file(path: Path) -> None:
    """Loads KEY=VALUE lines into the environment; variables already set win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip("'\""))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except (json.JSONDecodeError, FileNotFoundError) as error:
        print(f"Error loading JSON {path}: {error}")
        return None


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        os.replace(temporary_path, path)
    except OSError as error:
        print(f"Error saving JSON {path}: {error}")


def retry_delay(attempt: int) -> float:
    return min(2.0**attempt, MAX_RETRY_DELAY_SECONDS)


class SourceVideo:
    """Source url record tracking its status until all of its clips are saved."""

    def __init__(self, json_path: Path, data: dict[str, Any]) -> None:
        self.json_path = json_path
        self.data = data
        self.video_path: Path | None = None
        self.pending_clips = 0
        self.cutting_finished = False
        self.lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.json_path.stem

    def set_status(self, status: str, error_details: str | None = None) -> None:
        self.data["status"] = status
        if error_details is not None:
            self.data["error_details"] = error_details
        save_json(self.json_path, self.data)

    def clip_submitted(self) -> None:
        with self.lock:
            self.pending_clips += 1

    def clip_saved(self) -> None:
        with self.lock:
            self.pending_clips -= 1
            self._complete_if_done()

    def finish_cutting(self) -> None:
        with self.lock:
            self.cutting_finished = True
            self._complete_if_done()

    def _complete_if_done(self) -> None:
        if self.cutting_finished and self.pending_clips == 0:
            self.set_status("COMPLETED")


class ClipEncoder:
    """ffmpeg process fed with raw RGB frames; output goes to a file or stays in memory."""

    def __init__(self, width: int, height: int, fps: int, output_path: Path | None) -> None:
        self.output_path = output_path
        in_memory = output_path is None
        command = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        ]  # fmt: skip
        if in_memory:
            command += ["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4"]
            command.append("pipe:1")
        else:
            command.append(str(output_path))

        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE if in_memory else None
        )
        self.chunks: list[bytes] = []
        self.reader: threading.Thread | None = None
        if in_memory:
            self.reader = threading.Thread(target=self._drain_output, daemon=True)
            self.reader.start()

    def _drain_output(self) -> None:
        while chunk := self.process.stdout.read(1 << 20):
            self.chunks.append(chunk)

    def write(self, frame: np.ndarray) -> None:
        self.process.stdin.write(frame.tobytes())

    def finish(self) -> bytes | None:
        self.process.stdin.close()
        return_code = self.process.wait()
        if self.reader is not None:
            self.reader.join()
        if return_code != 0:
            raise RuntimeError("ffmpeg failed while encoding clip")
        return b"".join(self.chunks) if self.reader is not None else None

    def abort(self) -> None:
        self.process.kill()
        self.process.wait()
        if self.reader is not None:
            self.reader.join()
        if self.output_path is not None:
            self.output_path.unlink(missing_ok=True)


class UploadItem:
    def __init__(
        self, objects: list[tuple[str, bytes, str]], on_uploaded: Callable[[], None]
    ) -> None:
        self.objects = objects
        self.on_uploaded = on_uploaded
        self.uploaded_count = 0


class S3ClipUploader:
    """Background uploader that keeps clips in memory and retries until they are stored."""

    def __init__(self, bucket: str, worker_count: int, max_pending_items: int) -> None:
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )
        self.pending: queue.Queue[UploadItem | None] = queue.Queue(maxsize=max_pending_items)
        self.abort_event = threading.Event()
        self.bucket_ready = False
        self.bucket_lock = threading.Lock()
        self.threads = [
            threading.Thread(target=self._worker, daemon=True) for _ in range(worker_count)
        ]
        for thread in self.threads:
            thread.start()

    def ensure_bucket(self) -> None:
        with self.bucket_lock:
            if self.bucket_ready:
                return
            try:
                self.client.head_bucket(Bucket=self.bucket)
            except ClientError as error:
                if error.response["Error"]["Code"] not in MISSING_BUCKET_CODES:
                    raise
                self.client.create_bucket(Bucket=self.bucket)
                print(f"Created bucket {self.bucket}")
            self.bucket_ready = True

    def submit(self, item: UploadItem) -> None:
        self.pending.put(item)

    def _worker(self) -> None:
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                self._upload_with_retries(item)
            finally:
                self.pending.task_done()

    def _upload_with_retries(self, item: UploadItem) -> None:
        attempt = 0
        while item.uploaded_count < len(item.objects):
            key, body, content_type = item.objects[item.uploaded_count]
            try:
                self.ensure_bucket()
                self.client.put_object(
                    Bucket=self.bucket, Key=key, Body=body, ContentType=content_type
                )
                item.uploaded_count += 1
                attempt = 0
            except (BotoCoreError, ClientError) as error:
                attempt += 1
                delay = retry_delay(attempt)
                print(f"Upload of {key} failed ({error}); retrying in {delay:.0f}s")
                if self.abort_event.wait(delay):
                    return
        item.on_uploaded()

    def close(self) -> None:
        """Blocks until every pending clip is uploaded; Ctrl-C drops what is left."""
        try:
            self.pending.join()
        except KeyboardInterrupt:
            self.abort_event.set()
            print(f"Interrupted, {self.pending.qsize()} queued clips were not uploaded")
            return
        for _ in self.threads:
            self.pending.put(None)
        for thread in self.threads:
            thread.join()


class StreamVideosPipeline:
    def __init__(self, cfg: DictConfig, use_s3: bool) -> None:
        self.dataset_name: str = cfg.dataset_name
        self.height: int = cfg.general.height
        self.width: int = cfg.general.width
        self.clip_frames: int = cfg.general.clip_frames
        self.clip_fps: int = cfg.general.clip_fps
        self.filename_prefix: str = cfg.general.filename_prefix

        self.skip_start_pct: float = cfg.cut_videos.skip_start_pct
        self.skip_end_pct: float = cfg.cut_videos.skip_end_pct
        self.max_total_clips: int = cfg.cut_videos.max_total_clips
        self.max_clips_per_video: int | None = cfg.cut_videos.get("max_clips_per_video", None)
        self.spread_clips: bool = cfg.cut_videos.get("spread_clips", False)

        stream_cfg = cfg.stream_videos
        self.download_workers: int = stream_cfg.download_workers
        self.prefetch_videos: int = stream_cfg.prefetch_videos
        self.download_timeout_seconds: float = stream_cfg.download_timeout_seconds
        self.upload_workers: int = stream_cfg.upload_workers
        self.max_pending_uploads: int = stream_cfg.max_pending_uploads

        self.use_s3 = use_s3
        dataset_root = Path(f"./datasets/{self.dataset_name}")
        self.source_urls_root = dataset_root / "source_urls"
        self.clips_root = dataset_root / "videos"
        self.metadata_root = dataset_root / "metadata"
        self.download_dir = dataset_root / ("temp_downloads" if use_s3 else "raw_videos")

        self.uploader: S3ClipUploader | None = None
        self.stop_event = threading.Event()
        self.total_clips = 0
        self.clips_lock = threading.Lock()
        self.ready_videos: queue.Queue[SourceVideo | None] = queue.Queue(
            maxsize=self.prefetch_videos
        )
        self.download_tasks: queue.Queue[SourceVideo] = queue.Queue()

    def random_id(self) -> str:
        return f"{self.filename_prefix}_{secrets.token_urlsafe(32)}"

    def should_stop(self) -> bool:
        return self.total_clips >= self.max_total_clips

    def select_format_id(self, formats: list[dict[str, Any]]) -> str:
        """Smallest video format that still covers the target frame, else the largest one."""
        candidates = [
            video_format
            for video_format in formats
            if video_format.get("vcodec") not in (None, "none")
            and video_format.get("width")
            and video_format.get("height")
            and video_format.get("ext") != "mhtml"
        ]
        if not candidates:
            raise RuntimeError("no video formats available")

        def size_key(video_format: dict[str, Any]) -> tuple[int, float]:
            area = video_format["width"] * video_format["height"]
            return area, video_format.get("tbr") or 0.0

        sufficient = [
            video_format
            for video_format in candidates
            if video_format["width"] >= self.width and video_format["height"] >= self.height
        ]
        if sufficient:
            return min(sufficient, key=size_key)["format_id"]
        return max(candidates, key=size_key)["format_id"]

    def download_once(self, source: SourceVideo) -> Path:
        base_options: dict[str, Any] = {
            "cookies": "cookies.txt",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": 15,
            "retries": 0,
            "extractor_retries": 0,
            "fragment_retries": 0,
            "continuedl": True,
        }
        url: str = source.data["url"]

        with YoutubeDL(base_options) as probe:
            info = probe.extract_info(url, download=False)
        format_id = self.select_format_id(info["formats"])
        deadline = time.monotonic() + self.download_timeout_seconds

        def enforce_deadline(progress: dict[str, Any]) -> None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"download exceeded {self.download_timeout_seconds:.0f}s")

        download_options = {
            **base_options,
            "progress_hooks": [enforce_deadline],
            "format": format_id,
            "outtmpl": str(self.download_dir / f"{source.name}.%(ext)s"),
            "concurrent_fragment_downloads": 4,
        }
        with YoutubeDL(download_options) as downloader:
            result = downloader.extract_info(url, download=True)
        video_path = Path(result["requested_downloads"][0]["filepath"])
        if not video_path.exists():
            raise FileNotFoundError(f"Downloaded file not found: {video_path}")
        return video_path

    def remove_downloads(self, source: SourceVideo) -> None:
        for path in self.download_dir.glob(f"{source.name}.*"):
            path.unlink(missing_ok=True)

    def download_source(self, source: SourceVideo) -> bool:
        """Single attempt; any failure marks the source as ERROR so broken videos are skipped."""
        if self.stop_event.is_set():
            return False
        source.set_status("DOWNLOADING")
        try:
            source.video_path = self.download_once(source)
            return True
        except Exception as error:
            print(f"Skipping {source.name}, download failed: {error}")
            self.remove_downloads(source)
            source.set_status("ERROR", str(error))
            return False

    def download_worker(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    source = self.download_tasks.get_nowait()
                except queue.Empty:
                    break
                if not self.download_source(source):
                    continue
                self.ready_videos.put(source)
        finally:
            self.ready_videos.put(None)

    def finish_clip(self, source: SourceVideo, encoder: ClipEncoder, clip_id: str) -> None:
        clip_bytes = encoder.finish()
        metadata = {"total_frames": self.clip_frames}

        if self.uploader is None:
            save_json(self.metadata_root / f"{clip_id}.json", metadata)
        else:
            source.clip_submitted()
            self.uploader.submit(
                UploadItem(
                    objects=[
                        (f"videos/{clip_id}.mp4", clip_bytes, "video/mp4"),
                        (
                            f"metadata/{clip_id}.json",
                            json.dumps(metadata, indent=2).encode(),
                            "application/json",
                        ),
                    ],
                    on_uploaded=source.clip_saved,
                )
            )

        with self.clips_lock:
            self.total_clips += 1
        self.pbar.update(1)

    def cut_video(self, source: SourceVideo) -> None:
        capture = cv2.VideoCapture(str(source.video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(total_frames * self.skip_start_pct)
        end_frame = total_frames - int(total_frames * self.skip_end_pct)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        encoder: ClipEncoder | None = None
        clip_id = ""
        frames_in_clip = 0
        clips_from_video = 0
        clip_starts = (
            compute_spread_starts(
                frame_index, end_frame, self.clip_frames, self.max_clips_per_video
            )
            if self.spread_clips
            else []
        )
        try:
            while (
                frame_index < end_frame
                and not self.should_stop()
                and (
                    self.max_clips_per_video is None or clips_from_video < self.max_clips_per_video
                )
            ):
                if self.spread_clips and encoder is None:
                    if clips_from_video >= len(clip_starts):
                        break
                    frame_index = clip_starts[clips_from_video]
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                frame_index += 1

                is_read, bgr_frame = capture.read()
                if not is_read:
                    break

                bgr_frame = crop_maximal_rectangle(
                    bgr_frame, target_height=self.height, target_width=self.width
                )
                if encoder is None:
                    clip_id = self.random_id()
                    output_path = None if self.use_s3 else self.clips_root / f"{clip_id}.mp4"
                    encoder = ClipEncoder(self.width, self.height, self.clip_fps, output_path)
                encoder.write(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                frames_in_clip += 1

                if frames_in_clip == self.clip_frames:
                    self.finish_clip(source, encoder, clip_id)
                    encoder = None
                    frames_in_clip = 0
                    clips_from_video += 1
        finally:
            if encoder is not None:
                encoder.abort()
            capture.release()

    def process_source(self, source: SourceVideo) -> None:
        try:
            self.cut_video(source)
        except Exception as error:
            print(f"Error while processing {source.name}: {error}")
            source.set_status("ERROR", str(error))
        else:
            source.finish_cutting()
        finally:
            if self.use_s3:
                self.remove_downloads(source)

    def discard_source(self, source: SourceVideo) -> None:
        if self.use_s3:
            self.remove_downloads(source)
        source.set_status("PENDING")

    def collect_sources(self) -> list[SourceVideo]:
        sources = []
        for json_path in sorted(self.source_urls_root.glob("*.json")):
            data = load_json(json_path)
            if data is None or not data.get("url"):
                continue
            if data.get("status") not in STARTABLE_STATUSES:
                continue
            sources.append(SourceVideo(json_path, data))
        return sources

    def setup_s3(self) -> None:
        load_env_file(Path(".env"))
        missing = [name for name in REQUIRED_S3_VARIABLES if not os.environ.get(name)]
        if missing:
            raise SystemExit(f"Missing S3 credentials in environment/.env: {', '.join(missing)}")

        self.uploader = S3ClipUploader(
            self.dataset_name, self.upload_workers, self.max_pending_uploads
        )
        try:
            self.uploader.ensure_bucket()
        except ClientError as error:
            raise SystemExit(f"S3 bucket {self.dataset_name} is not usable: {error}")
        except BotoCoreError as error:
            print(f"S3 is unreachable now ({error}); uploads will retry in background")

    def run(self) -> None:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if self.use_s3:
            self.setup_s3()
        else:
            self.clips_root.mkdir(parents=True, exist_ok=True)
            self.metadata_root.mkdir(parents=True, exist_ok=True)

        sources = self.collect_sources()
        for source in sources:
            self.download_tasks.put(source)

        self.pbar = tqdm(total=self.max_total_clips, desc="Total Clips")
        start_time = time.time()
        workers = [
            threading.Thread(target=self.download_worker, daemon=True)
            for _ in range(self.download_workers)
        ]
        for worker in workers:
            worker.start()

        try:
            finished_workers = 0
            while finished_workers < len(workers):
                source = self.ready_videos.get()
                if source is None:
                    finished_workers += 1
                elif self.should_stop():
                    self.stop_event.set()
                    self.discard_source(source)
                else:
                    self.process_source(source)
        finally:
            self.stop_event.set()
            self.pbar.close()
            if self.uploader is not None:
                self.uploader.close()
            if self.use_s3:
                shutil.rmtree(self.download_dir, ignore_errors=True)
            elapsed = time.time() - start_time
            print(f"Produced {self.total_clips} clips in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--s3", action="store_true", help="upload clips to the S3 bucket")
    arguments = parser.parse_args()

    cfg = OmegaConf.load(arguments.config)
    StreamVideosPipeline(cfg, use_s3=arguments.s3).run()


if __name__ == "__main__":
    main()
