"""Evaluate a fitted scene without optimization on a chosen camera split."""

import math

import torch

from fitting_3d import render_view


@torch.no_grad()
def evaluate_scene(parameters, cameras, pixel_chunk=2048, save_indices=()):
    """Return per-view metrics, mean PSNR, and selected RGB renders."""
    if not cameras:
        raise ValueError("Evaluation needs at least one camera")
    rows, images = [], {}
    for i, camera in enumerate(cameras):
        image = render_view(parameters, camera, pixel_chunk)
        mse = ((image - camera["image"]) ** 2).mean().item()
        psnr = -10 * math.log10(mse) if mse > 0 else math.inf
        rows.append(dict(view=camera["file"], mse=mse, psnr=psnr))
        if i in save_indices:
            images[i] = image.cpu()
    return rows, sum(row["psnr"] for row in rows) / len(rows), images
