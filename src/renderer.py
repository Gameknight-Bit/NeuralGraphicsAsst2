"""Differentiable front-to-back rasterization over a black background."""

import torch

from gaussians import covariance_2d, gaussian_weight
from image_utils import get_device, save_image


def pixel_grid(H, W, device=None, dtype=torch.float32):
    """Return (H*W, 2) coordinates in (x, y) order, flattened row by row."""
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def render_pixels(xy, mu, Sigma, color, opacity, order):
    """Render (P, 2) pixel coordinates with differentiable over compositing."""
    order = torch.as_tensor(order, device=mu.device, dtype=torch.long)
    alpha = (gaussian_weight(xy, mu, Sigma) * opacity[None, :])[:, order]
    
    remaining = torch.cat((alpha.new_ones((len(xy), 1)), 1.0 - alpha), dim=1)
    T = torch.cumprod(remaining, dim=1)[:, :-1]
    return (T * alpha) @ color[order]


def render(mu, Sigma, color, opacity, order, H, W, pixel_chunk=2048):
    """Render N Gaussians as an (H, W, 3) floating-point RGB tensor.

    mu: (N, 2) gaussian centers in pixel coordinates.
    Sigma: (N, 2, 2) positive definite covariances from covariance_2d.
    color: (N, 3) straight (non-premultiplied) RGB in [0, 1].
    opacity: (N,) values in [0, 1].
    order: permutation of 0..N-1, sorted front to back.
    H, W: positive image height and width.
    """
    if H <= 0 or W <= 0 or pixel_chunk <= 0:
        raise ValueError("H, W, and pixel_chunk must be positive")
    xy = pixel_grid(H, W, device=mu.device, dtype=mu.dtype)
    chunks = [render_pixels(points, mu, Sigma, color, opacity, order)
              for points in xy.split(pixel_chunk)]

    return torch.cat(chunks, dim=0).reshape(H, W, 3)

#Can run this if we want to do small tests with renderer
def main():
    device = get_device()
    mu = torch.tensor([[50.0, 64.0], [78.0, 64.0]], device=device)
    scale = torch.tensor([[24.0, 14.0], [18.0, 24.0]], device=device)
    theta = torch.tensor([0.4, -0.3], device=device)
    color = torch.tensor([[1.0, 0.2, 0.1], [0.1, 0.4, 1.0]], device=device)
    opacity = torch.tensor([0.8, 0.7], device=device)
    order = torch.arange(2, device=device)

    Sigma = covariance_2d(scale, theta)
    image = render(mu, Sigma, color, opacity, order, H=128, W=128)

    save_image(image, "p2_render.png")
    print(f"Saved p2_render.png using {device}.")


if __name__ == "__main__":
    main()
