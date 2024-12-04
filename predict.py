# Prediction interface for Cog ⚙️
# https://cog.run/python


from argparse import Namespace
import shutil
import subprocess
import time
from cog import BasePredictor, Input, Path
import os
import torch
import imageio
import torchvision
from einops import rearrange

from hyvideo.inference import HunyuanVideoSampler

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

        args_dict = {
            "model": "HYVideo-T/2-cfgdistill",
            "latent_channels": 16,
            "precision": "bf16",
            "rope_theta": 256,
            "vae": "884-16c-hy",
            "vae_precision": "fp16",
            "vae_tiling": True,
            "text_encoder": "llm",
            "text_encoder_precision": "fp16",
            "text_states_dim": 4096,
            "text_len": 256,
            "tokenizer": "llm",
            "prompt_template": "dit-llm-encode",
            "prompt_template_video": "dit-llm-encode-video",
            "hidden_state_skip_layer": 2,
            "apply_final_norm": False,
            "text_encoder_2": "clipL",
            "text_encoder_precision_2": "fp16",
            "text_states_dim_2": 768,
            "tokenizer_2": "clipL",
            "text_len_2": 77,
            "denoise_type": "flow",
            "flow_solver": "euler",
            "flow_shift": 7.0,
            "flow_reverse": True,
            "use_linear_quadratic_schedule": False,
            "linear_schedule_end": 25,
            "use_cpu_offload": True,
            "batch_size": 1,
            "disable_autocast": False,
            "cfg_scale": 1.0,
            "embedded_cfg_scale": 6.0,
            "reproduce": False,
            "model_base": "ckpts",
            "dit_weight": "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            "model_resolution": "540p",
            "load_key": "module",
            "save_path": "./results",
            "save_path_suffix": "",
            "name_suffix": "",
            "num_videos": 1,
            "video_size": [480, 854],
            "seed_type": "auto",
            "video_length": 129,
            "infer_steps": 50,
            "prompt": "A cat walks on the grass, realistic style.",
            "seed": 65025,
            "neg_prompt": None,
        }
        args = Namespace(**args_dict)
        # Initialize the video sampler
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            Path(MODEL_CACHE),
            args=args,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt to generate video.",
            default="A cat walks on the grass, realistic style.",
        ),
        width: int = Input(
            description="Width of the video in pixels.", default=854, ge=1
        ),
        height: int = Input(
            description="Height of the video in pixels.", default=480, ge=1
        ),
        video_length: int = Input(
            description="Length of the video in frames.", default=129, ge=1
        ),
        infer_steps: int = Input(
            description="Number of inference steps.", default=50, ge=1
        ),
        flow_shift: float = Input(description="Flow-shift parameter.", default=7.0),
        embedded_guidance_scale: float = Input(
            description="Embedded guidance scale for generation.",
            default=6.0,
            ge=1.0,
            le=6.0,
        ),
        seed: int = Input(description="Random seed for reproducibility.", default=None),
    ) -> Path:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Update video_size in the sampler's args to match the requested dimensions
        self.hunyuan_video_sampler.args.video_size = [height, width]

        # Create save path and clear any existing files
        save_path = "./results"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

        # Generate video using HunyuanVideoSampler
        outputs = self.hunyuan_video_sampler.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=video_length,
            seed=seed,
            negative_prompt=None,
            infer_steps=infer_steps,
            guidance_scale=1.0,
            num_videos_per_prompt=1,
            flow_shift=flow_shift,
            batch_size=1,
            embedded_guidance_scale=embedded_guidance_scale,
        )

        samples = outputs["samples"]

        # Save the generated video
        sample = samples[0].unsqueeze(0)
        output_path = f"{save_path}/video.mp4"
        save_videos_grid(sample, output_path, fps=24)

        return Path(output_path)
