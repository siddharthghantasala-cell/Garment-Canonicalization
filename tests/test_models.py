"""Model-plumbing tests on tiny random components (CPU, no downloads):  pytest -q

Skipped (not errors) on a CPU-only install without torch/diffusers.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")      # must come before anything that imports torch
pytest.importorskip("diffusers")
pytest.importorskip("peft")


def test_adapter_shapes_and_key_names():
    """Our TryOffDiff must use the authors' parameter names, or tryoffdiff.pth won't load."""
    from garmentcanon.models.components import _tiny_unet
    from garmentcanon.models.tryoffdiff_arch import TryOffDiff
    net = TryOffDiff(_tiny_unet())
    assert {k.split(".")[0] for k in net.state_dict()} == {"unet", "transformer", "proj", "norm"}
    assert net.proj.in_features == 1024 and net.proj.out_features == 77
    cond = torch.randn(2, 1024, 768)
    assert net.adapt(cond).shape == (2, 77, 768)
    out = net(torch.randn(2, 4, 64, 64), torch.tensor([10, 500]), cond)
    assert out.shape == (2, 4, 64, 64)


def test_lora_checkpoint_is_small_and_reloads(tmp_path):
    from garmentcanon.models.components import _tiny_unet, load_finetuned
    from garmentcanon.models.lora import configure_trainable, trainable_state_dict
    from garmentcanon.models.tryoffdiff_arch import TryOffDiff
    torch.manual_seed(0)
    net = TryOffDiff(_tiny_unet())
    configure_trainable(net, "lora", rank=4, alpha=4, verbose=False)
    sd = trainable_state_dict(net)
    assert any("lora" in k for k in sd) and all(not k.startswith("unet.") or "lora" in k for k in sd)
    for v in sd.values():
        v.add_(0.01)                                 # pretend training changed them
    torch.save({"net": sd, "mode": "lora", "cfg": {"lora_rank": 4, "lora_alpha": 4}}, tmp_path / "c.pt")
    torch.manual_seed(0)
    fresh = TryOffDiff(_tiny_unet())
    load_finetuned(fresh, str(tmp_path / "c.pt"))
    for k, v in sd.items():
        assert torch.allclose(fresh.state_dict()[k], v)


def test_visibility_mask_fold_hides_garment():
    import cv2
    from garmentcanon.data.crumple_procedural import CrumpleParams, crumple
    img = np.full((512, 384, 3), 255, np.uint8)
    cv2.rectangle(img, (90, 80), (294, 450), (40, 70, 160), -1)
    garment = img.min(2) < 250
    _, p, vis = crumple(img, params=CrumpleParams(seed=3, fold_prob=1.0), return_visibility=True)
    assert vis.shape == garment.shape
    assert not vis[~garment].any()                   # never "visible" outside the garment
    frac = vis[garment].mean()
    assert 0.2 < frac < 0.95, frac                   # a fold hides a real chunk, not everything
    _, _, vis2 = crumple(img, params=CrumpleParams(seed=3, fold_prob=0.0), return_visibility=True)
    assert vis2[garment].mean() > 0.9                # no fold: almost everything visible


def test_adapter_without_transformer_loads_linear_ln_checkpoint():
    """The BMVC paper's adapter is Linear+LN only: such a checkpoint must load (auto-detected)."""
    from garmentcanon.models.components import _clean_state, _tiny_unet
    from garmentcanon.models.tryoffdiff_arch import TryOffDiff
    src = TryOffDiff(_tiny_unet(), use_transformer=False)
    state = {"module." + k: v for k, v in src.state_dict().items()}       # as saved by Accelerate
    state = _clean_state({"state_dict": state})
    assert not any(k.startswith("transformer.") for k in state)
    dst = TryOffDiff(_tiny_unet(), use_transformer=False)
    missing, unexpected = dst.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    assert dst.adapt(torch.randn(1, 1024, 768)).shape == (1, 77, 768)


def test_clean_state_accepts_whole_module():
    from garmentcanon.models.components import _clean_state, _tiny_unet
    from garmentcanon.models.tryoffdiff_arch import TryOffDiff
    net = TryOffDiff(_tiny_unet())
    assert set(_clean_state(net)) == set(net.state_dict())
