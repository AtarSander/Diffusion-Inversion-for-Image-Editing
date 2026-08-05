import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import argparse
import json
from PIL import Image

from transformers import BlipForConditionalGeneration, BlipProcessor
from models.pix2pix_zero.ddim_inv import DDIMInversion
from models.pix2pix_zero.scheduler import DDIMInverseScheduler
from models.pix2pix_zero.edit_directions import construct_direction
from models.pix2pix_zero.edit_pipeline import EditingPipeline
from utils.utils import txt_draw

from diffusers import DDIMScheduler

NUM_DDIM_STEPS = 50
XA_GUIDANCE=0.1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
caption_processor = None
caption_model = None


def generate_caption(image):
    global caption_processor, caption_model
    if caption_processor is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
        caption_model = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID, torch_dtype=dtype).to(device)
        caption_model.eval()
    inputs = caption_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=caption_model.dtype)
    with torch.no_grad():
        token_ids = caption_model.generate(pixel_values=pixel_values, max_new_tokens=32)
    return caption_processor.decode(token_ids[0], skip_special_tokens=True)


def load_pipelines(model_key):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    inversion_pipe = DDIMInversion.from_pretrained(model_key, torch_dtype=dtype).to(device)
    inversion_pipe.scheduler = DDIMInverseScheduler.from_config(inversion_pipe.scheduler.config)
    inversion_pipe.scheduler.num_inference_steps = NUM_DDIM_STEPS

    forward_pipe = EditingPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
    forward_pipe.scheduler = DDIMScheduler.from_config(forward_pipe.scheduler.config)
    forward_pipe.scheduler.num_inference_steps = NUM_DDIM_STEPS
    return inversion_pipe, forward_pipe


def load_lora_inversion(checkpoint_path, rank=16, lora_alpha=8, lora_dropout=0.0,
                        adapter_name="inversion", scale=1.0):
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {checkpoint_path}")
    lora_config = LoraConfig(
        r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        init_lora_weights=True, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    inject_adapter_in_model(lora_config, pipe.unet, adapter_name=adapter_name)
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "lora_state_dict" in state_dict:
        state_dict = state_dict["lora_state_dict"]
    set_peft_model_state_dict(pipe.unet, state_dict, adapter_name=adapter_name)
    if scale is not None and hasattr(pipe.unet, "set_adapters"):
        pipe.unet.set_adapters([adapter_name], weights=[float(scale)])
    set_lora_enabled(False)
    print(f"[INFO] loaded inversion LoRA from {checkpoint_path}")


def set_lora_enabled(enabled):
    for module in pipe.unet.modules():
        if module is pipe.unet:
            continue
        if hasattr(module, "enable_adapters"):
            module.enable_adapters(enabled)


pipe = None
edit_pipe = None




def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## convert sentences to sentence embeddings
def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device=device):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
            text_inputs = tokenizer(
                    sent,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.concat(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)


def edit_image_ddim_pix2pix_zero(image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                image_size=[512,512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    # generate the caption
    prompt_str = generate_caption(image_gt)
    latent_list, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inversion_steps=NUM_DDIM_STEPS,
            img=image_gt
        )
    
    inversion_latent=latent_list[-1].detach()
    
    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    
    rec_pil, edit_pil = edit_pipe(prompt_str,
                num_inference_steps=NUM_DDIM_STEPS,
                x_in=inversion_latent,
                edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                guidance_amount=XA_GUIDANCE,
                guidance_scale=guidance_scale,
                negative_prompt=prompt_str # use the unedited prompt for the negative prompt
        )
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    
    out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
    
    return Image.fromarray(out_image)
    

def edit_image_directinversion_pix2pix_zero(image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                image_size=[512,512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    # generate the caption
    prompt_str = generate_caption(image_gt)
    latent_list, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inversion_steps=NUM_DDIM_STEPS,
            img=image_gt
        )
    
    inversion_latent=latent_list[-1].detach()
    
    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    
    rec_pil, edit_pil = edit_pipe(prompt_str,
                num_inference_steps=NUM_DDIM_STEPS,
                x_in=inversion_latent,
                edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                guidance_amount=XA_GUIDANCE,
                guidance_scale=guidance_scale,
                negative_prompt=prompt_str, # use the unedited prompt for the negative prompt
                latent_list=latent_list
        )
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    
    out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
    
    return Image.fromarray(out_image)


def edit_image_lora_pix2pix_zero(image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                image_size=[512,512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    prompt_str = generate_caption(image_gt)

    # LoRA is used only for inversion. Pix2Pix-Zero's reconstruction and edit
    # pass use the untouched SD1.5 editing pipeline.
    set_lora_enabled(True)
    try:
        latent_list, _, _ = pipe(
            prompt_str,
            guidance_scale=1,
            num_inversion_steps=NUM_DDIM_STEPS,
            img=image_gt,
        )
    finally:
        set_lora_enabled(False)
    inversion_latent = latent_list[-1].detach()

    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    rec_pil, edit_pil = edit_pipe(
        prompt_str,
        num_inference_steps=NUM_DDIM_STEPS,
        x_in=inversion_latent,
        edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
        guidance_amount=XA_GUIDANCE,
        guidance_scale=guidance_scale,
        negative_prompt=prompt_str,
    )
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    return Image.fromarray(np.concatenate((np.array(image_instruct), np.array(image_gt),
                                            np.array(rec_pil[0]), np.array(edit_pil[0])), 1))


def edit_image_lora_directinversion_pix2pix_zero(
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                inversion_guidance_scale=1.0,
                image_size=[512,512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    prompt_str = generate_caption(image_gt)

    # Algorithm 1, inverse: replace DDIM inversion with the learned LoRA
    # inversion while retaining its full latent trajectory.
    set_lora_enabled(True)
    try:
        latent_list, _, _ = pipe(
            prompt_str,
            guidance_scale=inversion_guidance_scale,
            num_inversion_steps=NUM_DDIM_STEPS,
            img=image_gt,
        )
    finally:
        set_lora_enabled(False)
    inversion_latent = latent_list[-1].detach()

    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    rec_pil, edit_pil = edit_pipe(
        prompt_str,
        num_inference_steps=NUM_DDIM_STEPS,
        x_in=inversion_latent,
        edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
        guidance_amount=XA_GUIDANCE,
        guidance_scale=guidance_scale,
        negative_prompt=prompt_str,
        # Algorithm 1, editing: this enables the existing base-UNet Direct
        # Inversion offset calculation and application.
        latent_list=latent_list,
    )
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    return Image.fromarray(np.concatenate((np.array(image_instruct), np.array(image_gt),
                                            np.array(rec_pil[0]), np.array(edit_pil[0])), 1))


def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

    
image_save_paths={
    "ddim+pix2pix-zero":"ddim+pix2pix-zero",
    "lora+pix2pix-zero":"lora+pix2pix-zero",
    "lora+directinversion+pix2pix-zero":"lora+directinversion+pix2pix-zero",
    "directinversion+pix2pix-zero":"directinversion+pix2pix-zero",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pix2pix-zero","directinversion+pix2pix-zero"]) # the editing methods that needed to run
    parser.add_argument('--lora_checkpoint', type=str, default=None)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_scale', type=float, default=1.0)
    parser.add_argument('--inversion_guidance_scale', type=float, default=1.0)
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    lora_methods = {
        "lora+pix2pix-zero",
        "lora+directinversion+pix2pix-zero",
    }
    use_lora = any(method in lora_methods for method in edit_method_list)
    if use_lora and args.lora_checkpoint is None:
        raise ValueError("--lora_checkpoint is required when using a LoRA edit method")
    model_key = "runwayml/stable-diffusion-v1-5" if use_lora else "CompVis/stable-diffusion-v1-4"
    pipe, edit_pipe = load_pipelines(model_key)
    if use_lora:
        load_lora_inversion(checkpoint_path=args.lora_checkpoint, rank=args.lora_rank,
                            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                            scale=args.lora_scale)

    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                if edit_method=="ddim+pix2pix-zero":
                    edited_image = edit_image_ddim_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                        inversion_guidance_scale=args.inversion_guidance_scale,
                    )
                elif edit_method=="directinversion+pix2pix-zero":
                    edited_image = edit_image_directinversion_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="lora+pix2pix-zero":
                    edited_image = edit_image_lora_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="lora+directinversion+pix2pix-zero":
                    edited_image = edit_image_lora_directinversion_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                else:
                    raise NotImplementedError(f"No edit method named {edit_method}")
                
                
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
