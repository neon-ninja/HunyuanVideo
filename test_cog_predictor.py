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

    if data['status'] == "succeeded":
        datauri = data["output"]
        base64_encoded_data = datauri.split(",")[1]
        content = base64.b64decode(base64_encoded_data)
        Path(output_fn).write_bytes(content)
        return data['status']

    else:
        return data['status']


def test_prompts():
    """
    runs generations in fp8 and bf16 on the same node! wow!
    """
    output_directory = Path("test_outputs")
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for _ in range(5):
        status = gen(
            output_directory / "dog.mp4",
            prompt="a cool dog walking around",
            negative_prompt=None,
            width=864,
            height=480,
            video_length=12,
            infer_steps=30,
            flow_shift=7.0,
            embedded_guidance_scale=6.0,
            seed=1234,
        )
        assert status == "failed"
    

    status = gen(
        output_directory / "dog.mp4",
        prompt="a cool dog walking around",
        negative_prompt=None,
        width=864,
        height=480,
        video_length=13,
        infer_steps=30,
        flow_shift=7.0,
        embedded_guidance_scale=6.0,
        seed=1234,
    )
    assert status == "succeeded"
    assert (output_directory / "dog.mp4").exists(), "dog.mp4 was not generated"


    status = gen(
        output_directory / "cat.mp4",
        prompt="a cool cat walking around",
        negative_prompt=None,
        width=864,
        height=480,
        video_length=13,
        infer_steps=30,
        flow_shift=7.0,
        embedded_guidance_scale=6.0,
        seed=1234,
    )
    assert status == "succeeded"
    assert (output_directory / "cat.mp4").exists(), "dog.mp4 was not generated"


    status = gen(
        output_directory / "donkey.mp4",
        prompt="a cool donkey walking around",
        negative_prompt=None,
        width=864,
        height=480,
        video_length=12,
        infer_steps=30,
        flow_shift=7.0,
        embedded_guidance_scale=6.0,
        seed=1234,
    )
    assert status == "failed"

    shutil.rmtree(output_directory)

if __name__ == "__main__":
    test_prompts()
