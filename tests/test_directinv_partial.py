"""CPU checks for the Direct Inversion partial-depth control wiring."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("steps", [45, 46, 47, 48, 49, 50])
def test_direct_invert_forwards_depth_to_trajectory_and_corrections(steps):
    tree = ast.parse((ROOT / "pnp_inversion/models/p2p/inversion.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DirectInversion")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "invert")
    namespace = {"register_attention_control": lambda *args: None}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "inversion.py", "exec"), namespace)
    calls = []
    def inversion(image, inversion_steps):
        calls.append(("inverse", inversion_steps))
        return "reconstruction", list(range(inversion_steps + 1))
    def offsets(latents, *args, inversion_steps):
        assert len(latents) == inversion_steps + 1
        calls.append(("offsets", inversion_steps))
        return [0] * inversion_steps
    obj = SimpleNamespace(model=None, init_prompt=lambda _: None,
                          ddim_inversion=inversion, offset_calculate=offsets)
    result = namespace["invert"](obj, "image", ["source", "target"], 7.5, inversion_steps=steps)
    assert calls == [("inverse", steps), ("offsets", steps)]
    assert len(result[-1]) == steps


@pytest.mark.parametrize("path,function,required_calls", [
    ("pnp_inversion/models/p2p_editor.py", "edit_image_directinversion", {"invert": "inversion_steps", "direct_inversion_p2p_guidance_forward": "denoising_steps"}),
    ("pnp_inversion/run_editing_masactrl.py", "edit_image_directinversion_MasaCtrl", {"invert": "inversion_steps", "model": "denoising_steps"}),
    ("pnp_inversion/run_editing_pnp.py", "edit_image_directinversion_PnP", {"extract_latents": "inversion_steps", "run_pnp": "inversion_steps"}),
    ("pnp_inversion/run_editing_pix2pix_zero.py", "edit_image_directinversion_pix2pix_zero", {"pipe": "inversion_steps", "edit_pipe": "denoising_steps"}),
])
def test_all_direct_editor_paths_forward_depth(path, function, required_calls):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    assert "inversion_steps" in [a.arg for a in node.args.args]
    for name, keyword in required_calls.items():
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                 and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]
        assert calls, name
        assert all(any(k.arg == keyword for k in c.keywords) for c in calls), name
