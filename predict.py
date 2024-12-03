# Prediction interface for Cog ⚙️
# https://cog.run/python


import subprocess
import time
from cog import BasePredictor, Input, Path
import os
from pathlib import Path

MODEL_CACHE = "ckpts"
BASE_URL = f"https://weights.replicate.delivery/default/hunyuan-video/{MODEL_CACHE}/"


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


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        os.makedirs(MODEL_CACHE, exist_ok=True)
        model_files = [
            "hunyuan-video-t2v-720p.tar",
            "text_encoder.tar",
            "text_encoder_2.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt to generate video.",
            default="A cat walks on the grass, realistic style.",
        ),
        height: int = Input(
            description="Height of the video in pixels.", default=720, ge=1
        ),
        width: int = Input(
            description="Width of the video in pixels.", default=1280, ge=1
        ),
        video_length: int = Input(
            description="Length of the video in frames.", default=129, ge=1
        ),
        infer_steps: int = Input(
            description="Number of inference steps.", default=50, ge=1
        ),
        flow_shift: float = Input(description="Flow-shift parameter.", default=7.0),
        seed: int = Input(description="Random seed for reproducibility.", default=None),
    ) -> Path:
        """Run a single prediction on the model."""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Create the save path
        save_path = "./results"
        os.makedirs(save_path, exist_ok=True)

        # Construct the command string
        cmd = (
            f"python3 sample_video.py "
            f"--video-size {height} {width} "
            f"--video-length {video_length} "
            f"--infer-steps {infer_steps} "
            f'--prompt "{prompt}" '
            f"--flow-shift {flow_shift} "
            f"--seed {seed} "
            f"--flow-reverse "
            f"--use-cpu-offload "
            f"--save-path {save_path}"
        )

        # Execute the command
        os.system(cmd)

        # Get the most recently created file in the save_path directory
        results_path = Path(save_path)
        video_files = list(results_path.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError("No video files were generated.")

        latest_file = max(video_files, key=lambda x: x.stat().st_mtime)

        return Path(str(latest_file))
