import os
import numpy as np
from PIL import Image
import cv2
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline
from diffusers import DiffusionPipeline
from controlnet_aux import OpenposeDetector
from accelerate import PartialState


pipe = None

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
distributed_state = PartialState()
pipe.to(distributed_state.device)


prompt_list = ["working dog", "walking dog", "working duck", "walking duck",
                "working dog", "walking dog", "working duck", "walking duck",
                  "working dog", "walking dog", "working duck", "walking duck",
                    "working dog", "walking dog", "working duck", "walking duck"]

with distributed_state.split_between_processes(prompt_list) as prompt_list:
    # images = pipe(prompt=prompt_list, num_inference_steps=15, num_images_per_prompt=4, device_map="auto", width=800, height=1024)
    images = [pipe(prompt=p, num_inference_steps=15, num_images_per_prompt=4, device_map="auto", width=800, height=1024).images for p in prompt_list]
    print(images)

# print(images.images)

# for i, im in enumerate(images.images):
#     print(i, im)
#     im.save(f"tmp2/{prompt_list[i//4]}_{i%4}.jpg")