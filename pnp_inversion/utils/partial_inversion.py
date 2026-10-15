"""Map a partial DDIM inversion to its matching denoising timesteps."""


def validate_inversion_steps(inversion_steps, total_steps):
    if not 1 <= inversion_steps <= total_steps:
        raise ValueError(
            f"inversion_steps must be between 1 and {total_steps}, got {inversion_steps}"
        )


def validate_lora_inversion_depth(mode, inversion_steps, total_steps):
    validate_inversion_steps(inversion_steps, total_steps)
    if mode == "one_step_endpoint" and inversion_steps != total_steps:
        raise ValueError("One-step endpoint inversion requires the full denoising schedule.")


def inversion_timesteps(timesteps, inversion_steps):
    validate_inversion_steps(inversion_steps, len(timesteps))
    return list(reversed(timesteps))[:inversion_steps]


def denoising_timesteps(timesteps, inversion_steps):
    validate_inversion_steps(inversion_steps, len(timesteps))
    return timesteps[-inversion_steps:]
