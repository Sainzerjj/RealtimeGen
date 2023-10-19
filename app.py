import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import gradio as gr
from fastapi import FastAPI
import uvicorn
import torch
from generator_pipeline import StableDiffusionGeneratorPipeline, PipelineIntermediateState, StableDiffusionXLGeneratorPipeline
from preview_decoder import ApproximateDecoder
from PIL import Image
import PIL.Image
from accelerate import Accelerator
import re
from diffusers.utils import PIL_INTERPOLATION
import einops
import numpy as np
from diffusers import (DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler)
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from sd_train_utils import *
from torch.nn.parallel import DataParallel, DistributedDataParallel
from datetime import datetime

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int, default=1, help="number of cpu threads to use during batch generation")
# args = parser.parse_args()

class UNetDistributedDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):
        # This is because the timestep (inputs[1]) in UNet is a 0-d tensor and scatter will try to split inputs[1]. We simply convert it to a float so that scatter has no effect on it.
        
        inputs = inputs[0], inputs[1].item()
        return super().forward(*inputs, **kwargs)


model_type = "SDXL"
if model_type == "SDXL":
    model_id = "/data/zsz/models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/bf714989e22c57ddc1c453bf74dab4521acb81d8/"
    pipe = StableDiffusionXLGeneratorPipeline.from_pretrained(
        model_id,
        # pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
        # model_id, # use_auth_token=True,
        cache_dir="/data/zsz/models/",
        torch_dtype=torch.float16, variant="fp16",
    )
    refiner = DiffusionPipeline.from_pretrained(
        "/data/zsz/models/models--stabilityai--stable-diffusion-xl-refiner-1.0/snapshots/93b080bbdc8efbeb862e29e15316cff53f9bef86/",
        cache_dir="/data/zsz/models/",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
else:
    model_id = "/data/zsz/models/models--runwayml--stable-diffusion-v1-5/snapshots/ded79e214aa69e42c24d3f5ac14b76d568679cc2/"
    pipe = StableDiffusionGeneratorPipeline.from_pretrained(
        # pretrained_model_name_or_path = ""
        model_id, # use_auth_token=True,
        cache_dir="/data/zsz/models/",
        variant="fp16", torch_dtype=torch.float16,
    )
# model_id="runwayml/stable-diffusion-v1-5"
device = "cuda"
# torch.distributed.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)


accelerate = Accelerator()
#If you are running this code locally, you need to either do a 'huggingface-cli login` or paste your User Access Token from here https://huggingface.co/settings/tokens into the use_auth_token field below.


pipe.unet = UNetDistributedDataParallel(pipe.unet)
pipe.unet.config, pipe.unet.dtype, pipe.unet.add_embedding = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.add_embedding
pipe = pipe.to(device)


refiner.unet = UNetDistributedDataParallel(refiner.unet)
refiner.unet.config, refiner.unet.dtype, refiner.unet.add_embedding = refiner.unet.module.config, refiner.unet.module.dtype, refiner.unet.module.add_embedding
refiner = refiner.to(device)
# pipe = torch.nn.DataParallel(pipe, device_ids=[0, 1])
# pipe = pipe.module
# huggingface pipeline multi-gpu workaround
# pipe.unet = torch.nn.DataParallel(pipe.unet, device_ids=[0, 1])
# pipe.unet = pipe.unet.module
pipe.enable_attention_slicing()
torch.backends.cudnn.benchmark = True

# When running locally, you won`t have access to this, so you can remove this part
# word_list_dataset = load_dataset("stabilityai/word-list", data_files="list.txt", use_auth_token=True)
# word_list = word_list_dataset["train"]['text']
WORD_LIST = 'https://raw.githubusercontent.com/coffee-and-fun/google-profanity-words/main/data/list.txt'
import requests

# word_list = [word for word in requests.get(WORD_LIST, verify=False).text.split('\n') if word and not word.isspace()]


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

@torch.no_grad()
def infer(prompt, samples, steps, scale, seed, width, height, schedule):
    #When running locally you can also remove this filter
    # for filter in word_list:
    #     if re.search(rf"\b{filter}\b", prompt):
    #         raise gr.Error("Unsafe content found. Please try again with different prompts.")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    if schedule == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif schedule == "PNDM":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif schedule == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif schedule == "HEUN":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    # pipe = pipe.to(device)
    # pipe.enable_attention_slicing()

    if model_type == "SDXL":
        yield from pipe.generate(
                prompt = [prompt] * samples,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
                height = height,
                width = width,
            )

    else:
        with torch.autocast(pipe.device.type):
            yield from pipe.generate(
                prompt = [prompt] * samples,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
                height = height,
                width = width,
            )

@torch.no_grad()
def edit_infer(prompt, samples, steps, scale, seed, width, height, schedule, image, time):
    print("steps:", steps)
    print("time:", time)
    #When running locally you can also remove this filter
    # for filter in word_list:
    #     if re.search(rf"\b{filter}\b", prompt):
    #         raise gr.Error("Unsafe content found. Please try again with different prompts.")
    generator = torch.Generator(device=device).manual_seed(seed)
    if schedule == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif schedule == "PNDM":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif schedule == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif schedule == "HEUN":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    global denoised_images
    # pipe = pipe.to(device)
    # pipe.enable_attention_slicing()
    # print(len(noise_preds))
    if isinstance(image, PIL.Image.Image):
        image = preprocess_image(image)
        # d_image = preprocess_image(denoised_images[time - 1]) 
    if image is not None:
        image = image.to(device)
    # if d_image is not None:
    #     d_image = d_image.to(device)

    
    # subtracted_arr = torch.subtract(image, d_image)[0, :, :, :]
    # result = torch.zeros_like(subtracted_arr)
    # result[torch.where(subtracted_arr != 0)] = 1
    # result = result.to(device)
    # image = result * image + (1 - result) * denoised_images[-1]
    # print(image.shape)
    
    
    init_latent_dist = pipe.vae.encode(image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = 0.13025 * init_latents
    
    
    # init_latents = torch.cat([init_latents] * samples, dim=0)
    shape = init_latents.shape
    noise = torch.randn(shape, generator=generator, device=device) # if len(noise_preds) == 0 else noise_preds[time]
    pipe.scheduler.set_timesteps(steps)
    timesteps = pipe.scheduler.timesteps.to(device)
    t = timesteps[time - 1] * torch.ones(init_latents.shape[0]).to(device)
    t = t.long()
    latents = pipe.scheduler.add_noise(init_latents, noise, t)
    # print(torch.max(init_latents), torch.min(init_latents))
    # print(torch.max(latents), torch.min(latents))
    # latents = init_latents
    if model_type == "SDXL":
        yield from pipe.mid_generate(
                [prompt] * samples,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
                height = height,
                width = width,
                latents = latents.half(),
                time = time
            )
    else:
        with torch.autocast(pipe.device.type):
            yield from pipe.mid_generate(
                [prompt] * samples,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
                height = height,
                width = width,
                latents = latents,
                time = time
            )

def replace_unsafe_images(output):
    images = []
    # safe_image = Image.open(r"unsafe.png")
    # for image, is_unsafe in zip(output.images, output.nsfw_content_detected):
    #     if is_unsafe:
    #         images.append(safe_image)
    #     else:
    #         images.append(image)
    for image in output.images:
        images.append(image)
    return images

def float_tensor_to_pil(tensor: torch.Tensor) -> PIL.Image.Image:
    """aka torchvision's ToPILImage or DiffusionPipeline.numpy_to_pil

    (Reproduced here to save a torchvision dependency in this demo.)
    """
    tensor = (((tensor + 1) / 2)
              .clamp(0, 1)  # change scale from -1..1 to 0..1
              .mul(0xFF)  # to 0..255
              .byte())
    # tensor = einops.rearrange(tensor, 'c h w -> h w c')
    return PIL.Image.fromarray(tensor.cpu().numpy())

approximate_decoder = ApproximateDecoder.for_pipeline(pipe)
approximate_decoder_refiner = ApproximateDecoder.for_pipeline(refiner)

def percent_complete(timestep):
    max_timestep = pipe.scheduler.num_train_timesteps
    return 1. - timestep / max_timestep




css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        gallery {
            min-height: 62rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        # advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        # advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        # .footer {
        #     margin-bottom: 45px;
        #     margin-top: 35px;
        #     text-align: center;
        #     border-bottom: 1px solid #e5e5e5;
        # }
        # .footer>p {
        #     font-size: .8rem;
        #     display: inline-block;
        #     padding: 0 10px;
        #     transform: translateY(10px);
        #     background: white;
        # }
        # .dark .footer {
        #     border-color: #303030;
        # }
        # .dark .footer>p {
        #     background: #0b0f19;
        # }
        # .acknowledgments h4{
        #     margin: 1.25em 0 .25em 0;
        #     font-weight: bold;
        #     font-size: 115%;
        # }
"""

block = gr.Blocks(css=css)

noise_preds = []
denoised_images = []


examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
        1,
        20,
        7.5,
        1024,
        768,
        768,
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
        1,
        10,
        7,
        1024,
        768,
        768,
    ],
    [
        'A mecha robot in a favela in expressionist style',
        1,
        20,
        7,
        1024,
        1024,
        768,
    ],
    [
        'an insect robot preparing a delicious meal',
        1,
        30,
        7,
        1024,
        768,
        768,
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        1,
        20,
        7,
        1024,
        768,
        768,
    ],
]


# history_tab = gr.Tab("History", id="history")
with block as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion Free-Edit Demo
                </h1>
              </div>
            </div>
        """
    )
            #   <p style="margin-bottom: 10px; font-size: 94%">
            #     Stable Diffusion is a state of the art text-to-image model that generates
            #     images from text.<br>For faster generation and forthcoming API
            #     access you can try
            #     <a
            #       href="http://beta.dreamstudio.ai/"
            #       style="text-decoration: underline;"
            #       target="_blank"
            #       >DreamStudio Beta</a
            #     >
            #   </p>
            
    with gr.Row():
        with gr.Column(scale=55):
            with gr.Tab("Current Display"):
                with gr.Group():
                    with gr.Row():
                        gallery = gr.Gallery(
                            label="Generated images", show_label=False, elem_id="gallery"
                        ).style(grid=[1], height="auto", container=True)
                    
                    mid_gallery= gr.Gallery(
                        label="Generated mid_images", show_label=False, elem_id="mid_gallery"
                    ).style(grid=[5], width="auto", container=True, overflow="auto", display="flex", flex_wrap="nowrap")
            
            with gr.Tab("History", id="history"):
                with gr.Group():
                    with gr.Row():
                        history_gallery = gr.Gallery(
                            label="History", show_label=False, elem_id="history_gallery"
                        ).style(grid=[5], height="auto", container=True, display="flex", flex_wrap="nowrap")
                    show_gallery = gr.Gallery(
                            label="Show", show_label=False, elem_id="history_gallery"
                        ).style(grid=[5], width="auto", container=True, overflow="auto", isplay="flex", flex_wrap="nowrap")
        
        with gr.Column(scale=45):
            with gr.Tab("Text-to-Image"):
                with gr.Group():
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        text = gr.Textbox(
                            label="Enter your prompt",
                            show_label=False,
                            max_lines=3,
                            placeholder="Enter your prompt",
                            lines=3,
                        ).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,
                        )
                        btn_label = "Generate image"
                        btn = gr.Button("Generate image").style(
                            margin=False,
                            rounded=(False, True, True, False),
                        )
                    with gr.Row():
                        samples = gr.Slider(label="Images", minimum=1, maximum=6, value=1, step=1)
                        steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
                        
                    with gr.Row():
                        scale = gr.Slider(
                            label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
                        )
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=2147483647,
                            step=1,
                            randomize=True,
                        )

                    with gr.Row():
                        width = gr.Slider(label="Width", value=768, minimum=64, maximum=1920, step=64)
                        height = gr.Slider(label="Height", value=768, minimum=64, maximum=1920, step=64)
                    
                    scheduler_dd = gr.Radio(label="Scheduler", choices=["DPM", "DDIM", "PNDM", "HEUN"], value="DDIM", type="value")
                    settings = gr.Markdown()
            # with gr.Tab("FreeEdit"):
                    with gr.Group():
                        image = gr.Image(label="Edited_Image", height=256, tool="editor", type="pil")
                        time = gr.Slider(label="Inserted Step_position", minimum=1, maximum=50, step=1, value=1)
                    with gr.Row():
                        btn2_label = "Re-Generate"
                        generate2 = gr.Button(value="Re-Generate", variant="secondary").style(container=False)
                    gr.HTML("<br>")
                    with gr.Row():
                        btn3_label = "Refresh"
                        refresh = gr.Button(value="Refresh", variant="secondary", ).style(container=False)
                        
            # with gr.Row():



    @torch.no_grad()
    def progressive_infer(*args, **kwargs):
        global denoised_images
        # global noise_preds
        print(f"okay ready {args} {kwargs}")
        gsettings = (f"Prompt: {args[0]}, Scheduler: {args[7]}, Steps: {args[2]}, CFG: {args[3]}, Width x Height: {args[5]}x{args[6]}, Seed: {args[4]}, Model: SDXL")
        settings_dict = {
            "Prompt": args[0], 
            "Steps": args[2], 
            "CFG": args[3], 
            "Seed": args[4],
            "Width": args[5],
            "Height": args[6],
            "Scheduler": args[7], 
            "Model": "SDXL",
        }
        flag = 1
        for result in infer(*args, **kwargs):
            # print(f"getting result {type(result)}")
            print(str(datetime.now()))
            if isinstance(result, PipelineIntermediateState):
                # noise_preds.extend([noise_pred for noise_pred in result.noise_pred])
                previews = [approximate_decoder(latents) for latents in result.latents]  # float_tensor_to_pil
                denoised_images.extend([denoised_image for denoised_image in result.denoised_image])
                if flag == 1:
                    denoised_images = []
                    flag = 0
                yield {
                    mid_gallery: denoised_images,
                    gallery: previews,
                    btn: btn.update(value=f"{btn_label} [{percent_complete(result.timestep)*100:.1f}%]"),
                    settings: gsettings
                }
            else:
                # print(result)
                save_file(denoised_images, result.images[0], "original", settings_dict)
                # result = refiner(prompt=args[0], image=result.images[0][None, :])
                yield {
                    mid_gallery: denoised_images,
                    gallery: replace_unsafe_images(result),               # 
                    btn: btn.update(value=btn_label),
                    settings: gsettings
                }
        # return {
        #             mid_gallery: denoised_images,
        #             gallery: previews,
        #             btn: btn.update(value=f"{btn_label} [{percent_complete(0)*100:.1f}%]"),
        #             settings: gsettings
        #        }
    

    @torch.no_grad()
    def midedit_infer(*args, **kwargs):
        print("args in midedit_infer", args)
        global denoised_images
        print(f"okay ready {args} {kwargs}")
        gsettings = (f"Prompt: {args[0]}, Scheduler: {args[7]}, Steps: {args[2]}, CFG: {args[3]}, Width x Height: {args[5]}x{args[6]}, Seed: {args[4]}, Model: SDXL")
        settings_dict = {
            "Prompt": args[0], 
            "Steps": args[2], 
            "CFG": args[3], 
            "Seed": args[4],
            "Width": args[5],
            "Height": args[6],
            "Scheduler": args[7], 
            "Model": "SDXL",
            "Inserted_Step": args[9],
        }
        flag = 1
        mid_denoised_images = []
        print("len:", len(denoised_images))
        if len(denoised_images) > 0 :
                mid_denoised_images = denoised_images[:(args[9]-1)]    # stay args[9]-1
        for result in edit_infer(*args, **kwargs):
            print(f"getting result {type(result)}")
            if isinstance(result, PipelineIntermediateState):
                previews = [approximate_decoder(latents) for latents in result.latents]  # float_tensor_to_pil
                if flag == 1:
                    flag = 0
                else:
                    mid_denoised_images.extend([denoised_image for denoised_image in result.denoised_image])
                yield {
                    mid_gallery: mid_denoised_images,
                    gallery: previews,
                    generate2: generate2.update(value=f"{btn2_label} [{percent_complete(result.timestep)*100:.1f}%]"),
                    settings: gsettings
                }
            else:
                save_file(mid_denoised_images, result.images[0], "edit", settings_dict)
                # result = refiner(prompt=args[0], image=result.images[0][None, :])
                yield {
                    mid_gallery: mid_denoised_images,
                    gallery: replace_unsafe_images(result),
                    generate2: generate2.update(value=btn2_label),
                    settings: gsettings
                }
    

    def refresh_data():
        global denoised_images
        global noise_preds
        denoised_images = []
        noise_preds = []
        print("refreshed")
        pred_x_0_list = read_all_pred_x_0()
        print("show history successfully!")
        return mid_gallery.update(value=[]), gallery.update(value=[]), \
                    settings.update(value=""), history_gallery.update(value=pred_x_0_list), \
                        show_gallery.update(value=[])

        
        

    def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}, ")
        denoised_images = read_file(evt.value)
        return denoised_images

    def on_select_tab(evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}, ")
        pred_x_0_list = read_all_pred_x_0()
        return {history_gallery: history_gallery.update(value=pred_x_0_list)}

    
            # # advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            # with gr.Tab("Text2Img"):
            #     with gr.Group():  # elem_id="advanced-options"
                
        # mid_gallery= gr.Gallery(
        #                 label="Generated mid_images", show_label=False, elem_id="mid_gallery"
        #             ).style(grid=[5], width="auto", container=True, overflow="auto", display="flex", flex_wrap="nowrap")
            
        
        
    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, samples, steps, scale, seed, width, height],
                        outputs=gallery)  # , cache_examples=True)
    ex.dataset.headers = [""]

    text.submit(progressive_infer, inputs=[text, samples, steps, scale, seed, width, height, scheduler_dd], outputs=[mid_gallery, gallery, btn, settings])
    btn.click(progressive_infer, inputs=[text, samples, steps, scale, seed, width, height, scheduler_dd], outputs=[mid_gallery, gallery, btn, settings])
    generate2.click(midedit_infer, inputs=[text, samples, steps, scale, seed, width, height, scheduler_dd, image, time], outputs=[mid_gallery, gallery, generate2, settings], show_progress=True)
    refresh.click(refresh_data, inputs=[], outputs=[mid_gallery, gallery, settings, history_gallery, show_gallery])
    # refresh.click(show_history, inputs=[], outputs=[history_gallery])

    history_gallery.select(on_select, None, outputs=[show_gallery])
    # gr.Tab("History", id="history").select(on_select_tab, None, outputs=[history_gallery])
        # advanced_button.click
        #     None,
        #     [],
        #     text,
        #     _js="""
        #     () => {
        #         const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
        #         options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
        #     }""",
        # )
#         gr.HTML(
#             """
#                 <div class="footer">
#                     <p>Model by <a href="https://huggingface.co/CompVis" style="text-decoration: underline;" target="_blank">CompVis</a> and <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a> - Gradio Demo by 🤗 Hugging Face
#                     </p>
#                 </div>
#                 <div class="acknowledgments">
#                     <p><h4>LICENSE</h4>
# The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
#                     <p><h4>Biases and content acknowledgment</h4>
# Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
#                </div>
#            """
#         )
# app = FastAPI()
# @app.get("/")
# def read_main():
#     return {"message": "This is your main app"}
# app = gr.mount_gradio_app(app, demo, path="/ai_painting", gradio_api_url="")
demo.queue(max_size=25).launch(share=True, server_name="0.0.0.0", server_port=7683)
# uvicorn app:app --reload



# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 app.py
