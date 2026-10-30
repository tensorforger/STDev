import glob
import time

import cv2

from high_frequency_metrics import compute_metrics


def time_metrics(image_paths: list[str], thread_count: int) -> float:
    """
    returns:
        seconds per image on thread_count OpenCV threads, PNG decode excluded
    """
    cv2.setNumThreads(thread_count)
    images = [cv2.imread(image_path) for image_path in image_paths]
    compute_metrics(images[0])
    start_time = time.perf_counter()
    for image in images:
        compute_metrics(image)
    return (time.perf_counter() - start_time) / len(images)


if __name__ == "__main__":
    paths = sorted(glob.glob("dev/axis-v1_1k_1024/*/*.png"))[:20]
    for thread_count in (1, 32):
        print(f"threads={thread_count}: {1000 * time_metrics(paths, thread_count):.1f} ms/image")
