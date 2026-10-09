"""Load the posed images and shared intrinsics from the spheres dataset."""

import json
from pathlib import Path

import torch

from image_utils import load_image


def load_cameras(folder, device="cpu", split="train"):
    """Return camera dictionaries; preserve the images' original resolution."""
    folder = Path(folder)
    data = json.loads((folder / "cameras.json").read_text(encoding="utf-8-sig"))
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    frames = data["frames" if split == "train" else "val_frames"]
    if not frames:
        raise ValueError(f"No {split} views in cameras.json")
    K = torch.tensor(data["K"], dtype=torch.float32, device=device)
    H, W = int(data["height"]), int(data["width"])
    if K.shape != (3, 3) or min(H, W) <= 0:
        raise ValueError("Expected a shared 3x3 K and positive image dimensions")

    cameras = []
    for frame in frames:
        image = load_image(folder / frame["file"], max_size=None, device=device)
        R = torch.tensor(frame["R_wc"], dtype=torch.float32, device=device)
        t = torch.tensor(frame["t"], dtype=torch.float32, device=device)
        if image.shape != (H, W, 3) or R.shape != (3, 3) or t.shape != (3,):
            raise ValueError(f"Image/pose dimensions do not match cameras.json: {frame['file']}")
        cameras.append(dict(file=frame["file"], image=image, R_wc=R, t=t, K=K, H=H, W=W))
    return cameras
