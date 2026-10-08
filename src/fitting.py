"""P3: initialize and optimize 2D Gaussians, optionally using P4 densification."""

import math

import torch

from densification import densify, rebuild_adam
from gaussians import covariance_2d
from renderer import pixel_grid, render, render_pixels


def fit_image(target, num_gaussians=256, steps=2000, lr=1e-2, log_every=100,
              densification=False, budget=1024, densify_every=200,
              grad_threshold=2e-4, size_threshold=0.02, split_scale=1.6,
              prune_opacity=0.005, pixel_chunk=2048):
    """Optimize all five Gaussian parameter groups against an RGB target.

    target: (H, W, 3) floating-point tensor in [0, 1].
    Returns (final_image, parameters, history). Parameters are detached raw
    tensors, including log scales and color/opacity logits. History records
    each training render's step, MSE, and PSNR before that step's update.
    """
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("target must have shape (H, W, 3)")
    if min(num_gaussians, steps, log_every, pixel_chunk) <= 0 or lr <= 0:
        raise ValueError("Counts, steps, pixel_chunk, log_every, and lr must be positive")
    if densification and (budget < num_gaussians or densify_every <= 0
                          or grad_threshold < 0 or size_threshold <= 0
                          or split_scale <= 1 or not 0 <= prune_opacity < 1):
        raise ValueError("Invalid densification settings; budget must cover the starting count")

    H, W, _ = target.shape
    device, dtype = target.device, target.dtype

    # Parameter wraps the initialized values as leaf tensors for Adam.
    # Centers are spread over the integer pixel-center coordinate domain.
    mu = torch.nn.Parameter(
        torch.rand(num_gaussians, 2, device=device, dtype=dtype)
        * target.new_tensor([max(W - 1, 0), max(H - 1, 0)])
    )
    log_s = torch.nn.Parameter(torch.full(
        (num_gaussians, 2), math.log(0.02 * max(H, W)),
        device=device, dtype=dtype,
    ))
    theta = torch.nn.Parameter(torch.zeros(num_gaussians, device=device, dtype=dtype))
    color = torch.nn.Parameter(torch.zeros(num_gaussians, 3, device=device, dtype=dtype))
    op_raw = torch.nn.Parameter(torch.full((num_gaussians,), -2.0, device=device, dtype=dtype))

    parameters = {"mu": mu, "log_s": log_s, "theta": theta,
                  "color": color, "op_raw": op_raw}
    opt = torch.optim.Adam(parameters.values(), lr=lr)
    depth_order = list(range(num_gaussians))  # Fixed front-to-back order in 2D.
    history = []
    grad_sum = mu.new_zeros(num_gaussians)
    grad_steps = 0
    xy = pixel_grid(H, W, device=device, dtype=dtype)
    target_pixels = target.reshape(-1, 3)

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        mse_sum = target.new_zeros(())
        for start in range(0, H * W, pixel_chunk):
            # Rebuild this small graph per batch; accumulate gradients at the
            # same parameters, then update Adam once for the whole image.
            Sigma = covariance_2d(log_s.exp(), theta)
            image = render_pixels(xy[start:start + pixel_chunk], mu, Sigma,
                                  color.sigmoid(), op_raw.sigmoid(), depth_order)
            loss = ((image - target_pixels[start:start + pixel_chunk]) ** 2).sum() / target.numel()
            loss.backward()
            mse_sum += loss.detach()
        mse = mse_sum.item()
        if not math.isfinite(mse):
            raise RuntimeError(f"Non-finite loss at step {step}; try a smaller learning rate")
        if densification:
            # Accumulate magnitudes, not gradient vectors (which can cancel).
            with torch.no_grad():
                grad_sum += parameters["mu"].grad.norm(dim=1)
            grad_steps += 1
        opt.step()

        psnr = -10.0 * math.log10(mse) if mse > 0 else math.inf
        record = {"step": step, "mse": mse, "psnr": psnr, "count": len(mu)}
        history.append(record)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"Step {step:4d}/{steps}: N={len(mu)}, MSE={mse:.6f}, PSNR={psnr:.2f} dB")

        # Leave at least one optimization step after the final count change.
        if densification and step % densify_every == 0 and step < steps:
            old_parameters = parameters
            parameters, sources, new_rows, stats = densify(
                parameters, grad_sum / grad_steps, budget, W,
                grad_threshold, size_threshold, split_scale, prune_opacity,
            )
            if parameters is not old_parameters:
                opt = rebuild_adam(opt, old_parameters, parameters, sources, new_rows, lr)
                mu, log_s, theta, color, op_raw = (parameters[name] for name in
                                                  ("mu", "log_s", "theta", "color", "op_raw"))
                depth_order = list(range(len(mu)))
            grad_sum = mu.new_zeros(len(mu))
            grad_steps = 0
            record["densification"] = stats
            print(f"  Densify: {stats['before']} -> {stats['after']}; "
                  f"clone={stats['cloned']}, split={stats['split']}, prune={stats['pruned']}")

    # Re-render after the last update so the saved image matches saved weights.
    with torch.no_grad():
        Sigma = covariance_2d(log_s.exp(), theta)
        image = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order, H, W, pixel_chunk)
    return image, {name: value.detach() for name, value in parameters.items()}, history

