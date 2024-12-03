import os
from pathlib import Path
from einops import rearrange

import torch
import torchvision
import numpy as np
import imageio
import shutil

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 24.
    """
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
    frames_dir = os.path.join(os.path.dirname(path), 'frames_temp')
    os.makedirs(frames_dir, exist_ok=True)

    # Save frames as images
    for i, frame in enumerate(outputs):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        imageio.imwrite(frame_path, frame)

    # Build the ffmpeg command
    frame_pattern = os.path.join(frames_dir, "frame_%05d.png")
    ffmpeg_cmd = f"ffmpeg -y -framerate {fps} -i \"{frame_pattern}\" -c:v libx264 -pix_fmt yuv420p \"{path}\""

    # Run the ffmpeg command
    os.system(ffmpeg_cmd)

    # Clean up frames directory
    shutil.rmtree(frames_dir)
