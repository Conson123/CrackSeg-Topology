"""Model construction, ASPP channel lookup, and weight loading utilities."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .attention import AttentionUNet
from .aspp import ASPPUNet


# Bottleneck channel counts for common encoders (used to initialise ASPP).
_ENCODER_CHANNELS = {
    "efficientnet-b0": 1280, "efficientnet-b1": 1280, "efficientnet-b2": 1408,
    "efficientnet-b3": 1536, "efficientnet-b4": 1792, "efficientnet-b5": 2048,
    "efficientnet-b6": 2304, "efficientnet-b7": 2560,
    "resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048,
    "mobilenet_v2": 1280, "vgg16": 512,
}


def get_aspp_in_channels(encoder_name):
    """Return bottleneck output channels for *encoder_name*."""
    return _ENCODER_CHANNELS.get(encoder_name, 2048)


def _build_base(model_name, encoder_name, encoder_weights):
    builders = {
        "Unet": smp.Unet, "Linknet": smp.Linknet, "FPN": smp.FPN,
        "DeepLabV3": smp.DeepLabV3, "UnetPlusPlus": smp.UnetPlusPlus,
        "MAnet": smp.MAnet,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}")
    return builders[model_name](
        encoder_name=encoder_name, encoder_weights=encoder_weights,
        in_channels=3, classes=1,
    )


def get_model(model_name, encoder_name, encoder_weights="imagenet",
              load_weights_path=None, attention_type=None, use_aspp=False,
              device=None):
    """Construct the segmentation model with optional attention / ASPP.

    Parameters
    ----------
    model_name : str
        Architecture name (Unet, UnetPlusPlus, Linknet, ...).
    encoder_name : str
        Encoder backbone (mobilenet_v2, efficientnet-b4, ...).
    encoder_weights : str or None
        Pre-trained encoder weights (e.g. ``"imagenet"``).
    load_weights_path : str or None
        Path to a ``.pth`` checkpoint to load.
    attention_type : str or None
        One of ``"cbam"``, ``"se"``, ``"eca"``; ``None`` for no attention.
    use_aspp : bool
        Whether to insert an ASPP block at the encoder bottleneck.
    device : torch.device or None
        Device used for ``map_location`` when loading weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building model: {model_name}, encoder: {encoder_name}")
    if attention_type:
        print(f"  Attention: {attention_type.upper()}")
    if use_aspp:
        print(f"  ASPP module enabled")

    base_model = _build_base(model_name, encoder_name, encoder_weights)

    if attention_type in ("cbam", "se", "eca"):
        model = AttentionUNet(base_model, attention_type=attention_type,
                              attention_positions=["decoder"])
    else:
        model = base_model

    if use_aspp:
        aspp_in = get_aspp_in_channels(encoder_name)
        model = ASPPUNet(model, aspp_in_channels=aspp_in)
        print(f"  ASPP input channels: {aspp_in}")

    if load_weights_path:
        _load_weights(model, load_weights_path, attention_type, device)

    return model


def _load_weights(model, path, attention_type, device):
    """Load checkpoint with robust handling of DataParallel and attention
    structure mismatches."""
    print(f"Loading weights from {path} ...")
    try:
        state_dict = torch.load(path, map_location=device)

        # Strip 'module.' prefix left by DataParallel
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[7:] if k.startswith("module.") else k] = v
        state_dict = cleaned

        has_base_prefix = any(k.startswith("base_model.") for k in state_dict)

        if attention_type and not has_base_prefix:
            # Current model has attention wrapper; checkpoint does not.
            adapted = {f"base_model.{k}": v for k, v in state_dict.items()}
            missing, _ = model.load_state_dict(adapted, strict=False)
            print(f"  Loaded base weights ({len(adapted)} params); "
                  f"attention layers randomly initialised ({len(missing)} params).")
        elif not attention_type and has_base_prefix:
            # Checkpoint has attention wrapper; current model does not.
            adapted = {k[11:]: v for k, v in state_dict.items()
                       if k.startswith("base_model.")}
            model.load_state_dict(adapted, strict=True)
            print(f"  Loaded {len(adapted)} params (attention keys removed).")
        else:
            model.load_state_dict(state_dict, strict=True)
            print("  Weights loaded (exact match).")
    except Exception as e:
        print(f"Warning: weight loading failed ({e}). "
              "Falling back to ImageNet encoder weights.")
        import traceback
        traceback.print_exc()


def save_model_clean(model, path):
    """Save state dict without the ``module.`` prefix added by DataParallel."""
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
