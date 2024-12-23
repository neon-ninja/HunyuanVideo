import torch
import torchvision
import os
import shutil
import imageio
from einops import rearrange
import subprocess
import time
from enum import Enum


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class InferenceWorkerStatus(Enum):
    GOOD = "GOOD"
    ERROR = "ERROR"

    def __init__(self, status):
        self._error_message = None

    def set_error_message(self, message: str):
        if self == InferenceWorkerStatus.ERROR:
            self._error_message = message

    def get_error_message(self) -> str:
        if self == InferenceWorkerStatus.ERROR:
            return self._error_message
        return ""


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor"""
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for idx, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.permute(1, 2, 0)  # Convert to HWC
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).byte().cpu().numpy()
        outputs.append(x)

    # Create frames directory
    frames_dir = os.path.join(os.path.dirname(path), "frames_temp")
    os.makedirs(frames_dir, exist_ok=True)

    # Save frames as images
    for i, frame in enumerate(outputs):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        imageio.imwrite(frame_path, frame)

    # Build the ffmpeg command
    frame_pattern = os.path.join(frames_dir, "frame_%05d.png")
    ffmpeg_cmd = f'ffmpeg -y -framerate {fps} -i "{frame_pattern}" -c:v libx264 -pix_fmt yuv420p "{path}"'

    # Run the ffmpeg command
    os.system(ffmpeg_cmd)

    # Clean up frames directory
    shutil.rmtree(frames_dir)