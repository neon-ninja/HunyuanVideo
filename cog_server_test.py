"""
A handy utility for verifying image generation locally.
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""

import base64
import sys
import time
from pathlib import Path
import requests
import shutil

def gen(output_fn, **kwargs):
    st = time.time()
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()
    print("Generated in: ", time.time() - st)

    try:
        # import pdb; pdb.set_trace()
        datauri = data["output"]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except Exception:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    Path(output_fn).write_bytes(data)


def test_prompts():
    """
    runs generations in fp8 and bf16 on the same node! wow!
    """
    output_directory = Path("test_outputs")
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    gen(
        output_directory / "dog.mp4",
        prompt="a cool dog walking around",
        negative_prompt=None,
        width=854,
        height=480,
        video_length=13,
        infer_steps=30,
        flow_shift=7.0,
        embedded_guidance_scale=6.0,
        seed=1234,
    )

    gen(
        output_directory / "cat.mp4",
        prompt="a cool cat walking around",
        negative_prompt=None,
        width=854,
        height=480,
        video_length=13,
        infer_steps=30,
        flow_shift=7.0,
        embedded_guidance_scale=6.0,
        seed=4567,
    )

    assert (output_directory / "dog.mp4").exists(), "dog.mp4 was not generated"
    assert (output_directory / "cat.mp4").exists(), "cat.mp4 was not generated"
