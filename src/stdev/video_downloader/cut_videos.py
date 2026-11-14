import json
import os
import secrets
import subprocess
import sys
import time

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from stdev.utils.image_tools import crop_maximal_rectangle


def compute_spread_starts(
    start_frame: int, end_frame: int, clip_frames: int, max_clips: int | None
) -> list[int]:
    """Evenly spaced, non-overlapping clip start frames inside [start_frame, end_frame)."""
    usable_frames = end_frame - start_frame
    clip_count = usable_frames // clip_frames
    if max_clips is not None:
        clip_count = min(clip_count, max_clips)
    if clip_count <= 0:
        return []
    last_offset = usable_frames - clip_frames
    if clip_count == 1:
        return [start_frame + last_offset // 2]
    return [
        start_frame + round(index * last_offset / (clip_count - 1)) for index in range(clip_count)
    ]


class CutVideosPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        self.height: int = cfg.general.height
        self.width: int = cfg.general.width
        self.clip_frames: int = cfg.general.clip_frames
        self.clip_fps: int = cfg.general.clip_fps
        self.filename_prefix: str = cfg.general.filename_prefix

        dataset_root = f"./datasets/{cfg.dataset_name}"
        self.input_root = os.path.join(dataset_root, "raw_videos")
        self.output_root = os.path.join(dataset_root, "videos")
        self.metadata_root = os.path.join(dataset_root, "metadata")

        self.skip_start_pct: float = cfg.cut_videos.skip_start_pct
        self.skip_end_pct: float = cfg.cut_videos.skip_end_pct
        self.max_total_clips: int = cfg.cut_videos.max_total_clips
        self.max_clips_per_video: int | None = cfg.cut_videos.get("max_clips_per_video", None)
        self.spread_clips: bool = cfg.cut_videos.get("spread_clips", False)

        self.total_clips = 0

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

        self.total_clips += 1
        self.pbar.update(1)

    def should_stop(self) -> bool:
        return self.total_clips >= self.max_total_clips

    def process_video(self, video_path: str) -> None:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(total_frames * self.skip_start_pct)
        end_frame = total_frames - int(total_frames * self.skip_end_pct)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        frame_buffer: list[np.ndarray] = []
        clips_from_video = 0
        clip_starts = (
            compute_spread_starts(
                frame_index, end_frame, self.clip_frames, self.max_clips_per_video
            )
            if self.spread_clips
            else []
        )

        while (
            frame_index < end_frame
            and not self.should_stop()
            and (self.max_clips_per_video is None or clips_from_video < self.max_clips_per_video)
        ):
            if self.spread_clips and not frame_buffer:
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
            frame_buffer.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

            if len(frame_buffer) == self.clip_frames:
                self.write_clip(frame_buffer)
                frame_buffer = []
                clips_from_video += 1

        capture.release()

    def run(self) -> None:
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.metadata_root, exist_ok=True)

        filenames = sorted(os.listdir(self.input_root))
        self.pbar = tqdm(total=self.max_total_clips, desc="Total Clips")
        start_time = time.time()

        try:
            for filename in filenames:
                if self.should_stop():
                    break
                try:
                    self.process_video(os.path.join(self.input_root, filename))
                except Exception as error:
                    print(f"Error while processing {filename}: {error}")
        finally:
            self.pbar.close()
            elapsed = time.time() - start_time
            print(f"Wrote {self.total_clips} clips in {elapsed:.1f}s")


def main() -> None:
    cfg = OmegaConf.load(sys.argv[1])
    CutVideosPipeline(cfg).run()


if __name__ == "__main__":
    main()
