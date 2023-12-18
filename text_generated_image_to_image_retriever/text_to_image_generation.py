import os
import numpy as np
from PIL import Image
import cv2
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline
from diffusers import DiffusionPipeline, LCMScheduler, AutoPipelineForText2Image
from controlnet_aux import OpenposeDetector

COMMON_POSITIVE_PROMPTS = "In a photo studio, a 35-year-old female  fashion model wearing, best quality "
COMMON_NEGATIVE_PROMPTS = "looking back, cartoon, anime, mannequin, illustration, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, stage equipment"

pipe = None
def generate_image(text_list, common_positive_prompts=COMMON_POSITIVE_PROMPTS, common_negative_prompts=COMMON_NEGATIVE_PROMPTS, guidance_scale=13, batch_size=1, disable_tqdm=False, width=800, height=1024, num_inference_steps=15):
    global pipe
    if pipe is None:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=disable_tqdm)
    
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
    # print(text_list)

    images = [pipe(prompt=common_positive_prompts+p, num_inference_steps=num_inference_steps, guidance_scale = guidance_scale, negative_prompt=common_negative_prompts, num_images_per_prompt=batch_size, width=width, height=height).images for p in text_list]
    
    return images

control_pipe = None
openpose_image = None
controlnet = None
generator = None

OPEN_POSE_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/547388_hani.jpg")
OPEN_POSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/pose_2023_11_17_16_04_50.png")

def generate_controlnet_image(text_list, common_positive_prompts=COMMON_POSITIVE_PROMPTS, common_negative_prompts=COMMON_NEGATIVE_PROMPTS, guidance_scale=13, pose_image=None, batch_size=1):
    global control_pipe, openpose_image, controlnet, generator
    
    if control_pipe is None:
        controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
        control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        )

        control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)

        control_pipe.enable_xformers_memory_efficient_attention()
        control_pipe.enable_model_cpu_offload()
    
    
    #generator = torch.Generator(device="cpu").manual_seed(1)

    # get open pose image from a photo
    # openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    # openpose_source_image = load_image(OPEN_POSE_SOURCE_PATH)
    #openpose_source_image = load_image("https://storage.googleapis.com/k_fashion_images/k_fashion_images/547388.jpg")
    # openpose_source_image = load_image("https://storage.googleapis.com/k_fashion_images/k_fashion_images/197303.jpg")
    # openpose_source_image = load_image("https://storage.googleapis.com/k_fashion_images/k_fashion_images/618719.jpg")
    # openpose_image = openpose(openpose_source_image)
    
    print("[*] pose image", pose_image)
    if pose_image is None:
        openpose_image = Image.open(OPEN_POSE_PATH)
    else:
        openpose_image = pose_image
        
    openpose_image = openpose_image.resize((512, int(openpose_image.height/openpose_image.width*512)))

    # print(openpose_image)
    # print(openpose_image.shape)

    images = [control_pipe(
        common_positive_prompts + p,
        image=openpose_image,
        num_inference_steps=15,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=1.00,
        #generator=generator,
        negative_prompt=common_negative_prompts,
        num_images_per_prompt=batch_size,
    ).images for p in text_list]

    return images

lcm_pipe = None
def generate_lcm_image(text_list, common_positive_prompts=COMMON_POSITIVE_PROMPTS, common_negative_prompts=COMMON_NEGATIVE_PROMPTS, guidance_scale=13, batch_size=1, disable_tqdm=False, width=800, height=1024, num_inference_steps=15):
    global lcm_pipe
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"
    if lcm_pipe is None:
        lcm_pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        lcm_pipe.scheduler = LCMScheduler.from_config(lcm_pipe.scheduler.config)
        lcm_pipe.to("cuda")

        # load and fuse lcm lora
        lcm_pipe.load_lora_weights(adapter_id)
        lcm_pipe.fuse_lora()
        lcm_pipe.set_progress_bar_config(disable=disable_tqdm)

    images = [lcm_pipe(
        common_positive_prompts + p,
        image=openpose_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=1.00,
        #generator=generator,
        negative_prompt=common_negative_prompts,
        num_images_per_prompt=batch_size,
    ).images for p in text_list]

    return images





if __name__ == "__main__":
    # image_list = generate_image(["Black crop top suit jacket"])
    # image_list[0].save("test3.jpg")

    image_list = generate_controlnet_image(["Black crop top suit jacket"])
    image_list[0][0].save("test4.jpg")
