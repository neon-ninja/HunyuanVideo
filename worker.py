

from argparse import Namespace
import os

from cog import Path
import torch.distributed as dist
import torch.multiprocessing as mp

from hyvideo.inference import HunyuanVideoSampler
from cog_utils import save_videos_grid, InferenceWorkerStatus

FIXED_ARGS = {
    'guidance_scale': 1.0,
    'num_videos_per_prompt': 1,
    'batch_size': 1
}



def inference_worker_func(
        model_args : Namespace,
        model_cache : Path,
        rank: int,
        world_size: int,
        in_queue: mp.Queue,
        barrier: mp.Barrier,
        out_queue: mp.Queue):
        # done_event: mp.Event = None):
    """
    worker function for multi gpu execution
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            Path(model_cache),
            args=model_args,
    )

    args = hunyuan_video_sampler.args

    while True:
        try:

            predict_args = in_queue.get()

            # main process saying we are done
            if predict_args is None:
                dist.destroy_process_group()
                break
            else:
                # make sure workers are all on the same predict call
                barrier.wait()

            # print(f"Worker {rank} predict height, width, video_length: {predict_args['height']}, {predict_args['width']}, {predict_args['video_length']}")
            
            outputs = hunyuan_video_sampler.predict(
                prompt=predict_args['prompt'],
                height=predict_args['height'],
                width=predict_args['width'],
                video_length=predict_args['video_length'],
                seed=predict_args['seed'],
                negative_prompt=predict_args['negative_prompt'],
                infer_steps=predict_args['infer_steps'],
                guidance_scale=FIXED_ARGS['guidance_scale'],
                num_videos_per_prompt=FIXED_ARGS['num_videos_per_prompt'],
                flow_shift=predict_args['flow_shift'],
                batch_size=FIXED_ARGS['batch_size'],
                embedded_guidance_scale=predict_args['embedded_guidance_scale'],
            )

            if rank == 0:
                samples = outputs['samples']
                sample = samples[0].unsqueeze(0)
                save_videos_grid(sample, predict_args['save_path'], fps=24)
                
            # tell main process that we are done with this predict call
            out_queue.put(InferenceWorkerStatus.GOOD)

        except Exception as e:
            print(f"Worker {rank} failed with error: {e}, worker is terminating")
            error_status = InferenceWorkerStatus.ERROR
            error_status.set_error_message(str(e))
            out_queue.put(error_status)
            
        # destroy the process group to release the port that nccl is using
        # dist.destroy_process_group()
        # raise e


