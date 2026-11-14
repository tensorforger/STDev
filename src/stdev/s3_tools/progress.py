import os
import time

from tqdm import tqdm


class ProgressCallback:
    def __init__(self, file_path, desc="Uploading"):
        self.file_size = os.path.getsize(file_path)
        self.uploaded = 0
        self.start_time = time.time()
        self.progress_bar = tqdm(
            total=self.file_size,
            unit="B",
            unit_scale=True,
            desc=desc,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def __call__(self, bytes_amount):
        self.uploaded += bytes_amount
        self.progress_bar.update(bytes_amount)

        elapsed = time.time() - self.start_time
        if elapsed > 0:
            speed = self.uploaded / elapsed
            self.progress_bar.set_postfix_str(f"Speed: {self._format_speed(speed)}")

    def _format_speed(self, speed):
        if speed < 1024:
            return f"{speed:.2f} B/s"
        elif speed < 1024 * 1024:
            return f"{speed / 1024:.2f} KB/s"
        elif speed < 1024 * 1024 * 1024:
            return f"{speed / (1024 * 1024):.2f} MB/s"
        else:
            return f"{speed / (1024 * 1024 * 1024):.2f} GB/s"

    def close(self):
        self.progress_bar.close()
