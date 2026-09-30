# ABOUTME: Tests that an unguided DDIM inversion step skips the zero-weight unconditional branch:
# ABOUTME: one denoiser call at scale 1 (the conditional prediction), two and the CFG mix above it.

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "editing/AudioEditingCode/code"))

# ddim_inversion imports get_text_embeddings from the editing utils, which pull in the whole model
# stack; get_noise_pred needs none of it. Stub the import, then drop the module again so a later
# import anywhere else gets the real one.
_saved_utils = sys.modules.get("utils")
sys.modules["utils"] = types.SimpleNamespace(get_text_embeddings=None)
try:
    from ddm_inversion.ddim_inversion import get_noise_pred  # noqa: E402
finally:
    sys.modules.pop("ddm_inversion.ddim_inversion", None)
    if _saved_utils is None:
        del sys.modules["utils"]
    else:
        sys.modules["utils"] = _saved_utils


class CountingLDM:
    """A denoiser whose output is the latent times the conditioning value, logging each call."""

    def __init__(self):
        self.calls = []

    def unet_forward(self, latent, timestep, encoder_hidden_states, class_labels,
                     encoder_attention_mask):
        self.calls.append(encoder_hidden_states)
        return types.SimpleNamespace(sample=latent * encoder_hidden_states), None, None


def embedding(value: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(embedding_hidden_states=value, embedding_class_lables=None,
                                 boolean_prompt_mask=None)


def test_scale_one_makes_one_conditional_call():
    ldm, x = CountingLDM(), torch.ones(3)
    out = get_noise_pred(ldm, x, 1, embedding(3.0), embedding(5.0), 1.0)
    assert ldm.calls == [3.0]
    assert torch.equal(out, 3.0 * x)


def test_guided_scale_still_mixes_both_branches():
    ldm, x = CountingLDM(), torch.ones(3)
    out = get_noise_pred(ldm, x, 1, embedding(3.0), embedding(5.0), 3.0)
    assert ldm.calls == [5.0, 3.0]  # unconditional first, as before the change
    assert torch.allclose(out, (5.0 + 3.0 * (3.0 - 5.0)) * x)
