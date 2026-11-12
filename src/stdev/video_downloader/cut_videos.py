import json
import os
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from stdev.utils.image_tools import crop_maximal_rectangle


class CutVideosPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        self.height: int = cfg.general.height
        self.width: int = cfg.general.width
        self.clip_frames: int = cfg.general.clip_frames
        self.clip_fps: int = cfg.general.clip_fps
        self.filename_prefix: str = cfg.general.filename_prefix

        dataset_root = f"./datasets/{cfg.dataset_name}"
        if cfg.general.raw_videos_source is not None:
            self.input_root = f"./datasets/{cfg.general.raw_videos_source}/raw_videos"
        else:
            self.input_root = os.path.join(dataset_root, "raw_videos")
        self.output_root = os.path.join(dataset_root, "videos")
        self.metadata_root = os.path.join(dataset_root, "metadata")

        self.skip_start_pct: float = cfg.cut_videos.skip_start_pct
        self.skip_end_pct: float = cfg.cut_videos.skip_end_pct
        self.max_total_frames = int(cfg.cut_videos.max_total_hours * 60 * 60 * self.clip_fps)
        self.max_clips_per_video: int = cfg.cut_videos.max_clips_per_video
        self.frame_stride: int = cfg.cut_videos.frame_stride
        self.num_workers: int = cfg.cut_videos.num_workers

        self.total_frames = 0
        self._frames_lock = threading.Lock()
        self._stop_event = threading.Event()

    def random_id(self) -> str:
        return f"{self.filename_prefix}_{secrets.token_urlsafe(32)}"

    def start_encoder(self, output_path: str) -> subprocess.Popen:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.clip_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            output_path,
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    def write_clip(self, frames: list[np.ndarray]) -> None:
        clip_id = self.random_id()
        output_path = os.path.join(self.output_root, clip_id + ".mp4")

        encoder = self.start_encoder(output_path)
        for frame in frames:
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {output_path}")

        with open(os.path.join(self.metadata_root, clip_id + ".json"), "w") as metadata_file:
            json.dump({"total_frames": len(frames)}, metadata_file, indent=2)

        with self._frames_lock:
            self.total_frames += len(frames)
            self.pbar.update(len(frames))

    def should_stop(self, clips: int) -> bool:
        return (
            clips >= self.max_clips_per_video
            or self.total_frames >= self.max_total_frames
            or self._stop_event.is_set()
        )

    def process_video(self, video_path: str) -> None:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(total_frames * self.skip_start_pct)
        end_frame = total_frames - int(total_frames * self.skip_end_pct)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        clips = 0
        frame_buffer: list[np.ndarray] = []

        while frame_index < end_frame and not self.should_stop(clips):
            for _ in range(self.frame_stride - 1):
                capture.grab()
            frame_index += self.frame_stride

            is_read, bgr_frame = capture.read()
            if not is_read:
                break

            bgr_frame = crop_maximal_rectangle(
                bgr_frame, target_height=self.height, target_width=self.width
            )
            frame_buffer.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

            if len(frame_buffer) == self.clip_frames:
                self.write_clip(frame_buffer)
                frame_buffer = []
                clips += 1

        capture.release()

    def run(self) -> None:
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.metadata_root, exist_ok=True)

        filenames = sorted(os.listdir(self.input_root))
        self.pbar = tqdm(total=self.max_total_frames, desc="Total Frames")
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_video, os.path.join(self.input_root, filename)
                    ): filename
                    for filename in filenames
                }
                try:
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as error:
                            print(f"Error while processing {futures[future]}: {error}")
                except KeyboardInterrupt:
                    print("\nInterrupt received, waiting for workers to finish current clips...")
                    self._stop_event.set()
                    executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.pbar.close()
            elapsed = time.time() - start_time
            fps = self.total_frames / elapsed if elapsed > 0 else 0
            print(f"Processed {self.total_frames} frames in {elapsed:.1f}s ({fps:.1f} fps avg)")


def main() -> None:
    cfg = OmegaConf.load(sys.argv[1])
    CutVideosPipeline(cfg).run()


if __name__ == "__main__":
    main()
