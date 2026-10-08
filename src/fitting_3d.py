"""P7/P8: fit a shared 3D Gaussian cloud, with optional densification."""

import math

import torch

from densification import rebuild_adam
from densification_3d import densify_3d
from gaussians_3d import covariance_3d, project_gaussian
from renderer import pixel_grid, render, render_pixels


def initialize_scene(count, device):
    """initialize: uniform [-1.5, 1.5]^3 centers and scale 0.08."""
    quat = torch.zeros(count, 4, device=device)
    quat[:, 0] = 1.0
    values = {
        "mu3": (torch.rand(count, 3, device=device) * 2 - 1) * 1.5,
        "log_s": torch.full((count, 3), math.log(0.08), device=device),
        "quat": quat,
        "color": torch.zeros(count, 3, device=device),
        "op_raw": torch.full((count,), -2.0, device=device),
    }
    return {name: torch.nn.Parameter(value) for name, value in values.items()}


def project_scene(parameters, camera):
    """Cull behind/near-camera centers, project, and sort for this camera."""
    p = parameters
    visible = (p["mu3"] @ camera["R_wc"].T + camera["t"])[:, 2] > 1e-3
    Sigma3 = covariance_3d(p["log_s"][visible].exp(), p["quat"][visible])
    mu2, Sigma2, depth = project_gaussian(
        p["mu3"][visible], Sigma3, camera["R_wc"], camera["t"], camera["K"],
    )
    # Tiny pixel-variance regularizer for the covariance solve; not a blur pass.
    Sigma2 = Sigma2 + 1e-4 * torch.eye(2, device=Sigma2.device, dtype=Sigma2.dtype)
    return mu2, Sigma2, p["color"][visible].sigmoid(), p["op_raw"][visible].sigmoid(), depth.argsort()


def render_view(parameters, camera, pixel_chunk=2048):
    """Render one camera using the shared 3D scene parameters."""
    projected = project_scene(parameters, camera)
    if len(projected[0]) == 0:
        return parameters["mu3"].new_zeros((camera["H"], camera["W"], 3))
    return render(*projected, camera["H"], camera["W"], pixel_chunk)


def fit_scene(cameras, num_gaussians=4000, iters=1500, lr=1e-2,
              pixel_chunk=2048, log_every=100, densification=False, budget=4000,
              densify_every=200, grad_threshold=2e-4, size_threshold=0.05,
              split_scale=1.6, prune_opacity=0.005, camera_seed=0):
    """Fit all five parameter groups; one random training camera per step.

    Pixel batches accumulate the full chosen view's MSE gradient before one
    Adam update, keeping memory manageable. Returns raw parameters and history.
    """
    if not cameras or min(num_gaussians, iters, pixel_chunk, log_every) <= 0 or lr <= 0:
        raise ValueError("Provide cameras and positive count, iterations, batch size, log interval, and lr")
    if densification and (budget < num_gaussians or densify_every <= 0
                          or grad_threshold < 0 or size_threshold <= 0
                          or split_scale <= 1 or not 0 <= prune_opacity < 1):
        raise ValueError("Invalid densification settings; budget must cover the starting count")
    p = initialize_scene(num_gaussians, cameras[0]["image"].device)
    opt = torch.optim.Adam(p.values(), lr=lr)
    history = []
    grad_sum, grad_steps = p["mu3"].new_zeros(num_gaussians), 0
    # Independent RNG makes plain/densified runs see the same camera sequence.
    camera_rng = torch.Generator().manual_seed(camera_seed)
    for step in range(1, iters + 1):
        index = torch.randint(len(cameras), (), generator=camera_rng).item()
        cam = cameras[index]
        target = cam["image"].reshape(-1, 3)
        xy = pixel_grid(cam["H"], cam["W"], target.device, target.dtype)
        opt.zero_grad(set_to_none=True)
        mse_sum = target.new_zeros(())
        for start in range(0, len(xy), pixel_chunk):
            # Recompute projection so every batch has its own autograd graph.
            projected = project_scene(p, cam)
            if len(projected[0]) == 0:
                raise RuntimeError(f"No Gaussians in front of training camera {cam['file']}")
            image = render_pixels(xy[start:start + pixel_chunk], *projected)
            loss = ((image - target[start:start + pixel_chunk]) ** 2).sum() / target.numel()
            loss.backward()
            mse_sum += loss.detach()
        mse = mse_sum.item()
        if not math.isfinite(mse):
            raise RuntimeError(f"Non-finite loss at step {step}; try a smaller learning rate")
        if densification:
            with torch.no_grad():
                grad_sum += p["mu3"].grad.norm(dim=1)
            grad_steps += 1
        opt.step()
        psnr = -10 * math.log10(mse) if mse > 0 else math.inf
        record = dict(step=step, camera=cam["file"], mse=mse, psnr=psnr, count=len(p["mu3"]))
        history.append(record)
        if step == 1 or step % log_every == 0 or step == iters:
            print(f"Step {step}/{iters}, view {index}, N={len(p['mu3'])}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")
        if densification and step % densify_every == 0 and step < iters:
            old = p
            p, sources, new_rows, stats = densify_3d(
                p, grad_sum / grad_steps, budget, grad_threshold, size_threshold,
                split_scale, prune_opacity,
            )
            if p is not old:
                opt = rebuild_adam(opt, old, p, sources, new_rows, lr)
            grad_sum, grad_steps = p["mu3"].new_zeros(len(p["mu3"])), 0
            record["densification"] = stats
            print(f"  Densify: {stats['before']} -> {stats['after']}; "
                  f"prune={stats['pruned']}, clone={stats['cloned']}, split={stats['split']}")
    return {name: value.detach() for name, value in p.items()}, history
