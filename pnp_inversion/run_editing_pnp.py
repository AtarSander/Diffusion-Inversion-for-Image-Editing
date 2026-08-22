
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T

from utils.utils import txt_draw,load_512,latent2image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

NUM_DDIM_STEPS = 50

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device,model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(
            model_key,
            subfolder="vae",
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key,
            subfolder="unet",
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.lora_loaded = False
        self.lora_adapter_name = "inversion"
        self.lora_mode = "single"
        self.lora_branch_adapter_names = None
        self.lora_scale = 1.0
        print(f'[INFO] loaded stable diffusion!')

    def load_lora(
        self,
        checkpoint_path,
        rank=16,
        lora_alpha=8,
        lora_dropout=0.0,
        adapter_name="inversion",
        scale=1.0,
    ):
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint does not exist: {checkpoint_path}")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            init_lora_weights=True,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        inject_adapter_in_model(lora_config, self.unet, adapter_name=adapter_name)
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "lora_state_dict" in state_dict:
            state_dict = state_dict["lora_state_dict"]
        set_peft_model_state_dict(self.unet, state_dict, adapter_name=adapter_name)

        if scale is not None and hasattr(self.unet, "set_adapters"):
            self.unet.set_adapters([adapter_name], weights=[float(scale)])

        self.lora_loaded = True
        self.lora_adapter_name = adapter_name
        self.lora_mode = "single"
        self.lora_branch_adapter_names = None
        self.lora_scale = float(scale)
        self.set_lora_enabled(False)
        print(f"[INFO] loaded inversion LoRA from {checkpoint_path}")

    def load_branch_pair_lora(
        self, checkpoint_path, rank=16, lora_alpha=8, lora_dropout=0.0, scale=1.0
    ):
        if self.lora_loaded:
            raise RuntimeError("An inversion LoRA checkpoint is already loaded.")
        from utils.lora_adapters import load_branch_pair_lora

        resolved_path, adapter_names = load_branch_pair_lora(
            self.unet, checkpoint_path, rank=rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, scale=scale,
        )
        self.lora_loaded = True
        self.lora_mode = "branch_pair"
        self.lora_branch_adapter_names = adapter_names
        self.lora_scale = float(scale)
        print(f"[INFO] loaded branch-pair inversion LoRAs from {resolved_path}")

    def set_lora_enabled(self, enabled):
        if not self.lora_loaded:
            return
        for module in self.unet.modules():
            if module is self.unet:
                continue
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled)


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(
        self, cond, latent, uncond=None, guidance_scale=1.0,
        lora_branch_adapter_names=None, lora_adapter_scale=1.0,
    ):
        latent_list=[latent]
        timesteps = list(reversed(self.scheduler.timesteps))
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                if lora_branch_adapter_names is not None:
                    if uncond is None:
                        raise ValueError("Branch-pair CFG inversion requires an empty-prompt embedding.")
                    from utils.lora_adapters import set_active_lora_adapter

                    uncond_adapter_name, cond_adapter_name = lora_branch_adapter_names
                    uncond_batch = uncond.repeat(latent.shape[0], 1, 1)
                    set_active_lora_adapter(self.unet, uncond_adapter_name, lora_adapter_scale)
                    eps_uncond = self.unet(
                        latent, t, encoder_hidden_states=uncond_batch
                    ).sample
                    set_active_lora_adapter(self.unet, cond_adapter_name, lora_adapter_scale)
                    eps_cond = self.unet(
                        latent, t, encoder_hidden_states=cond_batch
                    ).sample
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                elif uncond is None or guidance_scale == 1.0:
                    eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample
                else:
                    uncond_batch = uncond.repeat(latent.shape[0], 1, 1)
                    eps_uncond, eps_cond = self.unet(
                        torch.cat([latent, latent]), t,
                        encoder_hidden_states=torch.cat([uncond_batch, cond_batch]),
                    ).sample.chunk(2)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt='', use_lora_inversion=False,
                        inversion_guidance_scale=1.0):
        if use_lora_inversion and not self.lora_loaded:
            raise RuntimeError("LoRA inversion requested, but no LoRA checkpoint was loaded.")

        self.scheduler.set_timesteps(num_steps)

        uncond, cond = self.get_text_embeds(inversion_prompt, "").chunk(2)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        self.set_lora_enabled(use_lora_inversion)
        try:
            inverted_x = self.ddim_inversion(
                cond, latent, uncond=uncond, guidance_scale=inversion_guidance_scale,
                lora_branch_adapter_names=self.lora_branch_adapter_names,
                lora_adapter_scale=self.lora_scale,
            )
        finally:
            self.set_lora_enabled(False)

        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        return inverted_x, rgb_reconstruction, latent_reconstruction


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

class PNP(nn.Module):
    def __init__(self, model_key,n_timesteps=NUM_DDIM_STEPS,device="cuda"):
        super().__init__()
        self.device = device

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
        else:
            print("[INFO] xFormers is unavailable; using standard attention.")

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(n_timesteps, device=self.device)
        self.n_timesteps=NUM_DDIM_STEPS
        print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self,image_path):
        # load image
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, t,guidance_scale,noisy_latent):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self,image_path,noisy_latent,target_prompt,guidance_scale=7.5,pnp_f_t=0.8,pnp_attn_t=0.5):

        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        
        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps,guidance_scale,noisy_latent)
        
        return edited_img

    def sample_loop(self, x,guidance_scale,noisy_latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t,guidance_scale,noisy_latent[-1-i])

            decoded_latent = self.decode_latent(x)
                
        return decoded_latent


model_key = "runwayml/stable-diffusion-v1-5"
toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
toy_scheduler.set_timesteps(NUM_DDIM_STEPS)

timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=NUM_DDIM_STEPS,
                                                           strength=1.0,
                                                           device=device)
model = Preprocess(device, model_key=model_key)
pnp = PNP(model_key)


def edit_image_ddim_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    _, rgb_reconstruction, latent_reconstruction = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)
    
    edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))



def edit_image_directinversion_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    inverted_x, rgb_reconstruction, _ = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)

    edited_image=pnp.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))


def edit_image_lora_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    _, rgb_reconstruction, latent_reconstruction = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src,
                                         use_lora_inversion=True)

    edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))

def edit_image_lora_directinversion_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    inversion_guidance_scale=1.0,
    image_shape=[512,512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    inverted_x, _, _ = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src,
                                         use_lora_inversion=True,
                                         inversion_guidance_scale=inversion_guidance_scale)

    # Use the LoRA-produced inversion trajectory as the per-step source
    # reference, exactly as in the standard Direct Inversion + PnP path.
    edited_image=pnp.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))

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
    "ddim+pnp":"ddim+pnp",
    "directinversion+pnp":"directinversion+pnp",
    "lora+pnp":"lora+pnp",
    "lora+directinversion+pnp":"lora+directinversion+pnp",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pnp","directinversion+pnp"]) # the editing methods that needed to run
    parser.add_argument('--lora_checkpoint', type=str, default=None)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_scale', type=float, default=1.0)
    parser.add_argument('--lora_mode', choices=["single", "branch_pair"], default="single")
    parser.add_argument('--inversion_guidance_scale', type=float, default=1.0)
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list

    lora_methods = {"lora+pnp", "lora+directinversion+pnp"}
    if args.lora_mode == "branch_pair" and "lora+pnp" in edit_method_list:
        raise ValueError(
            "Branch-pair LoRAs are supported only by lora+directinversion+pnp."
        )
    if any(method in lora_methods for method in edit_method_list):
        if args.lora_checkpoint is None:
            raise ValueError("--lora_checkpoint is required when using a LoRA edit method")
        load_method = (
            model.load_branch_pair_lora
            if args.lora_mode == "branch_pair"
            else model.load_lora
        )
        load_method(
            checkpoint_path=args.lora_checkpoint,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            scale=args.lora_scale,
        )
    
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
                if edit_method=="ddim+pnp":
                    edited_image = edit_image_ddim_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                        inversion_guidance_scale=args.inversion_guidance_scale,
                    )
                elif edit_method=="directinversion+pnp":
                    edited_image = edit_image_directinversion_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="lora+pnp":
                    edited_image = edit_image_lora_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="lora+directinversion+pnp":
                    edited_image = edit_image_lora_directinversion_PnP(
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
