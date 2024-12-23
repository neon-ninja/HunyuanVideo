# Prediction interface for Cog ⚙️
# https://cog.run/python


from argparse import Namespace
import shutil
from cog import BasePredictor, Input, Path
import os

from cog_utils import download_weights, save_videos_grid, InferenceWorkerStatus
from hyvideo.inference import HunyuanVideoSampler

import torch.multiprocessing as mp
import torch

from worker import inference_worker_func
WORLD_SIZE = torch.cuda.device_count()

# when new processes are created in setup(), they will be spawned rather than forked, this is more
# resource intensive up front because a new python interpreter is launched and all modules
# in the current scope are reimported. However it is required because each subprocess needs its
# own CUDA context, and if subprocesses are forked they all end up sharing the parents which is
# broken. This check ensures that mp.set_start_method() is only called once by the main process
# and not by any of the worker processes.

if WORLD_SIZE > 1:
    current_method = mp.get_start_method(allow_none=True)
    if current_method != "spawn":
        print(f"{os.getpid()} setting start method to spawn")
        mp.set_start_method('spawn', force=True)



MODEL_CACHE = "ckpts"
BASE_URL = f"https://weights.replicate.delivery/default/hunyuan-video/{MODEL_CACHE}/"

MODEL_ARGS = {
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
    "use_cpu_offload": False,
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
    "ulysses_degree": 1,
    "ring_degree": 1
}




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

        args = Namespace(**MODEL_ARGS)
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
        negative_prompt: str = Input(
            description="Text prompt to specify what you don't want in the video.",
            default=None,
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
            negative_prompt=negative_prompt,
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





class MultiGPUPredictor(BasePredictor):
    """
    A predictor class for multi-gpu inference
    
    torch.distributed requires one process per worker, or equivalently one process per GPU that we are parallelizing over.
    In the setup function, WORLD_SIZE worker processes are spawned, each worker function executes the function defined in 
    worker.py

    Each worker processes has its own mp.Queue that it waits on. When predict() is called, the main process (the one that is
    executing the predict function) populates each processes `in_queue` with the predict() arguments (prompt, height, width, etc.)

    Once each worker process recieves predict arguments, they begin running inference, parallelizing the large matrix operations in
    certain layers as defined by the torch.distributed collectives in the internals of the model. Once the worker processes are done,
    they will each hold a copy of the model output in GPU memory.
    
    The process with rank 0 will write its output to disk, and then signal to the main process that it has done so
    """
    
    def setup(self):
        
        # communication of predict args from main process to workers
        self.in_queue = [mp.Queue() for _ in range(WORLD_SIZE)]
        self.out_queue = [mp.Queue() for _ in range(WORLD_SIZE)]
        # signal that the main process is done waiting for worker processes
        # self.done_event = mp.Event()

        # for synchronization between worker processes, to ensure they all start inference at the same time
        self.barrier = mp.Barrier(WORLD_SIZE)
        self.processes = []
        
        # Download the model files
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

        MODEL_ARGS["ulysses_degree"] = WORLD_SIZE
        model_args = Namespace(**MODEL_ARGS)

        # mp_ctx = mp.get_context("spawn")
        for rank in range(WORLD_SIZE):
            worker_args = (
                model_args,
                Path(MODEL_CACHE),
                rank,
                WORLD_SIZE,
                self.in_queue[rank],
                self.barrier,
                self.out_queue[rank]
                # self.done_event if rank == 0 else None
            )

            p = mp.Process(target=inference_worker_func, args=worker_args)
            p.start()
            self.processes.append(p)
    
    def predict(
        self,
        prompt: str = Input(
            description="Text prompt to generate video.",
            default="A cat walks on the grass, realistic style.",
        ),
        negative_prompt: str = Input(
            description="Text prompt to specify what you don't want in the video.",
            default=None,
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
        seed: int = Input(description="Random seed for reproducibility.", default=1234),
    ) -> Path:

        # set up temporary directory for output
        save_directory = MODEL_ARGS['save_path']
        save_path = os.path.join(save_directory, f"video_{seed}.mp4")
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory)
        os.makedirs(save_directory)
    
        predict_args = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'video_length': video_length,
            'infer_steps': infer_steps,
            'flow_shift': flow_shift,
            'embedded_guidance_scale': embedded_guidance_scale,
            'seed': seed,
            'save_path': save_path,
        }

        # broadcast the predict args to all worker processes
        for rank in range(WORLD_SIZE):
            self.in_queue[rank].put(predict_args)

        # wait for the main process to finish
        # this done event will be set() by the worker with rank 0 once it
        # has finished writing the video to disk
        worker_status = [self.out_queue[rank].get() for rank in range(WORLD_SIZE)]
        if any([status == InferenceWorkerStatus.ERROR for status in worker_status]):
            for status in worker_status:
                if status == InferenceWorkerStatus.ERROR:
                    raise Exception(f"Predict: Worker {rank} failed with error: {status.get_error_message()}")
            

        # worker with rank 0 should have written the video to disk by now
        assert os.path.exists(save_path), f"Video not found at {save_path}"

        return Path(save_path)
