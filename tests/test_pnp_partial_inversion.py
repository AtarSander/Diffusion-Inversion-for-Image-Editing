import pytest

from pnp_inversion.utils.partial_inversion import (
    denoising_timesteps,
    inversion_timesteps,
    validate_lora_inversion_depth,
)


def test_partial_inversion_uses_matching_scheduler_timesteps():
    timesteps = list(range(981, 0, -20))

    assert inversion_timesteps(timesteps, 5) == [1, 21, 41, 61, 81]
    assert denoising_timesteps(timesteps, 5) == [81, 61, 41, 21, 1]
    assert len(inversion_timesteps(timesteps, 45)) == 45
    assert len(denoising_timesteps(timesteps, 45)) == 45


@pytest.mark.parametrize("steps", [0, 51])
def test_partial_inversion_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError):
        inversion_timesteps(list(range(50)), steps)


def test_endpoint_requires_full_schedule_but_iterative_allows_partial():
    validate_lora_inversion_depth("iterative", 5, 50)
    validate_lora_inversion_depth("one_step_endpoint", 50, 50)
    with pytest.raises(ValueError, match="full denoising schedule"):
        validate_lora_inversion_depth("one_step_endpoint", 5, 50)
