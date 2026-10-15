import pdb, sys

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from models.pix2pix_zero.base_pipeline import BasePipeline
from models.pix2pix_zero.cross_attention import prep_unet
from diff_inversion.eval.editor_controls import pix2pix_attention_alignment_loss
from utils.partial_inversion import denoising_timesteps


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EditingPipeline(BasePipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        attention_loss_reduction="legacy_sum",
        attention_loss_scale=1.0,
        attention_grad_clip_norm=None,
        edit_dir=None,
        x_in=None,
        only_sample=False, # only perform sampling, and no editing
        latent_list=None,
        reference_attention_maps=None,
        noise_loss_list_override=None,
        return_reference_artifacts=False,
        attention_gradient_diagnostics=None,
        diagnostic_guidance_amounts=(0.025, 0.05, 0.075, 0.1),
        diagnostic_clip_norm=1.0,
        diagnostic_normalized_loss_scales=(1.0,),
    ):

        if not np.isfinite(attention_loss_scale) or attention_loss_scale <= 0:
            raise ValueError("attention_loss_scale must be finite and positive")
        if any(not np.isfinite(scale) or scale <= 0 for scale in diagnostic_normalized_loss_scales):
            raise ValueError("diagnostic_normalized_loss_scales must be finite and positive")
        x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 0. modify the unet to be useful :D
        self.unet = prep_unet(self.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device)
        # 3. Encode input prompt = 2x77x1024
        prompt_embeds = self._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        denoising_steps = denoising_steps or num_inference_steps
        timesteps = denoising_timesteps(self.scheduler.timesteps, denoising_steps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)
        
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - denoising_steps * self.scheduler.order
        
        noise_loss_list=[]
        with torch.no_grad():
            with self.progress_bar(total=denoising_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(dtype=self.unet.dtype)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.to(dtype=self.unet.dtype)
                    if latent_list is not None:
                        noise_loss_list.append(latent_list[-2-i]-latents)
                        latents=latents+noise_loss_list[-1]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))

        # Crossing diagnostics may independently replace the reference-attention
        # cache and Direct-Inversion offsets while keeping the selected endpoint.
        if reference_attention_maps is not None:
            d_ref_t2attn = reference_attention_maps
        if noise_loss_list_override is not None:
            noise_loss_list = [value.to(device=device, dtype=self.unet.dtype) for value in noise_loss_list_override]

        if only_sample:
            if return_reference_artifacts:
                return image_rec, d_ref_t2attn, noise_loss_list
            return image_rec


        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        prompt_embeds_edit[1:2] += edit_dir
        
        latents = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - denoising_steps * self.scheduler.order
        with self.progress_bar(total=denoising_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual
                noise_pred = self.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),cross_attention_kwargs=cross_attention_kwargs,).sample

                attention_pairs = []
                attention_names = []
                for name, module in self.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention" and "attn2" in name:
                        curr = module.attn_probs
                        ref = d_ref_t2attn[t.item()][name].detach().to(
                            device=device, dtype=curr.dtype
                        )
                        attention_names.append(name)
                        attention_pairs.append((curr, ref))

                if attention_gradient_diagnostics is None:
                    loss = pix2pix_attention_alignment_loss(
                        attention_pairs, reduction=attention_loss_reduction
                    ) * float(attention_loss_scale)
                    loss.backward(retain_graph=False)
                else:
                    legacy_loss = pix2pix_attention_alignment_loss(
                        attention_pairs, reduction="legacy_sum"
                    )
                    normalized_loss = pix2pix_attention_alignment_loss(
                        attention_pairs, reduction="per_layer_mean"
                    )
                    legacy_grad = torch.autograd.grad(
                        legacy_loss, x_in, retain_graph=True
                    )[0]
                    scale_candidates = list(dict.fromkeys([
                        1.0,
                        *[float(scale) for scale in diagnostic_normalized_loss_scales],
                        *([float(attention_loss_scale)] if attention_loss_reduction == "per_layer_mean" else []),
                    ]))
                    normalized_grads = {}
                    for scale_index, scale in enumerate(scale_candidates):
                        normalized_grads[f"{scale:g}"] = torch.autograd.grad(
                            normalized_loss * scale, x_in,
                            retain_graph=scale_index < len(scale_candidates) - 1,
                        )[0]
                    normalized_grad = normalized_grads["1"]
                    legacy_norm = float(legacy_grad.float().norm().item())
                    normalized_norm = float(normalized_grad.float().norm().item())
                    denominator = legacy_norm * normalized_norm
                    cosine = (
                        float(
                            torch.sum(
                                legacy_grad.float() * normalized_grad.float()
                            ).item()
                            / denominator
                        )
                        if denominator > 0.0
                        else 0.0
                    )
                    clip_norm = (
                        None
                        if diagnostic_clip_norm is None
                        else float(diagnostic_clip_norm)
                    )
                    layer_records = []
                    for layer_name, (current, reference) in zip(
                        attention_names, attention_pairs
                    ):
                        squared_error = (
                            current.detach().float()
                            - reference.detach().float()
                        ).square()
                        layer_records.append(
                            {
                                "name": layer_name,
                                "shape": list(current.shape),
                                "numel": current.numel(),
                                "legacy_loss": float(
                                    squared_error.sum((1, 2)).mean(0).item()
                                ),
                                "mean_loss": float(squared_error.mean().item()),
                            }
                        )
                    scaled_gradient_records = {}
                    for scale_key, gradient in normalized_grads.items():
                        gradient_norm = float(gradient.float().norm().item())
                        scaled_denominator = legacy_norm * gradient_norm
                        scaled_gradient_records[scale_key] = {
                            "grad_norm": gradient_norm,
                            "cosine_to_legacy": (
                                float(torch.sum(legacy_grad.float() * gradient.float()).item() / scaled_denominator)
                                if scaled_denominator > 0.0 else 0.0
                            ),
                        }
                    attention_gradient_diagnostics.append(
                        {
                            "step_index": i,
                            "timestep": int(t.item()),
                            "legacy_loss": float(legacy_loss.detach().item()),
                            "normalized_loss": float(
                                normalized_loss.detach().item()
                            ),
                            "legacy_grad_norm": legacy_norm,
                            "normalized_grad_norm": normalized_norm,
                            "gradient_cosine": cosine,
                            "diagnostic_clip_norm": clip_norm,
                            "legacy_post_clip_norm": (
                                legacy_norm
                                if clip_norm is None
                                else min(legacy_norm, clip_norm)
                            ),
                            "normalized_post_clip_norm": (
                                normalized_norm
                                if clip_norm is None
                                else min(normalized_norm, clip_norm)
                            ),
                            "normalized_scaled_gradients": scaled_gradient_records,
                            "candidate_update_norms": {
                                f"{float(amount):g}": {
                                    "legacy": float(amount) * legacy_norm,
                                    "normalized": float(amount) * normalized_norm,
                                    "legacy_clipped": float(amount)
                                    * (
                                        legacy_norm
                                        if clip_norm is None
                                        else min(legacy_norm, clip_norm)
                                    ),
                                    "normalized_clipped": float(amount)
                                    * (
                                        normalized_norm
                                        if clip_norm is None
                                        else min(normalized_norm, clip_norm)
                                    ),
                                }
                                for amount in diagnostic_guidance_amounts
                            },
                            "layers": layer_records,
                        }
                    )
                    selected_grad = (
                        legacy_grad
                        if attention_loss_reduction == "legacy_sum"
                        else normalized_grads[f"{float(attention_loss_scale):g}"]
                    )
                    x_in.grad = selected_grad.detach()

                if attention_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [x_in], float(attention_grad_clip_norm)
                    )
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample
                
                latents = x_in.detach().chunk(2)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.to(dtype=self.unet.dtype)
                if latent_list is not None or noise_loss_list_override is not None:
                    latents=latents+noise_loss_list[i]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        # 8. Post-processing
        image = self.decode_latents(latents.detach())

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image_edit = self.numpy_to_pil(image)


        if return_reference_artifacts:
            return image_rec, image_edit, d_ref_t2attn, noise_loss_list
        return image_rec, image_edit
