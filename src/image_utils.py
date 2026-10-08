"""Device selection and RGB image loading/saving."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def load_image(path, max_size=128, device=None):
    """Load RGB in [0, 1] as (H, W, 3), preserving aspect ratio when resizing.

    max_size limits the longest side; small images are not enlarged.
    Pass None to keep the original resolution.
    """
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be positive or None")
    with Image.open(path) as source:
        image = source.convert("RGB")
        if max_size is not None:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        pixels = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).to(device=device)



def save_image(image, path):
    """Export an (H, W, 3) tensor as an 8-bit RGB image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (image.detach().cpu().numpy().clip(0, 1) * 255).round()
    Image.fromarray(pixels.astype(np.uint8)).save(path)

