"""P8: prune floaters and clone/split 3D Gaussians within a budget."""

import math

import torch

from gaussians_3d import quaternion_to_rotation


@torch.no_grad()
def densify_3d(p, grad_mag, budget, grad_threshold=2e-4, size_threshold=0.05,
               split_scale=1.6, prune_opacity=0.005):
    """Use mean position-gradient norms; size_threshold is in world units.

    Split children use opposite sampled 3D offsets, scaled and rotated by the
    parent. Quaternions, colors, and opacity logits are copied. Return the same
    source-row mapping used by P4's rebuild_adam helper.
    """
    count = len(p["mu3"])
    if not 0 < count <= budget or grad_mag.shape != (count,):
        raise ValueError("Expected a nonempty cloud within budget and one score per Gaussian")
    if grad_threshold < 0 or size_threshold <= 0 or split_scale <= 1 or not 0 <= prune_opacity < 1:
        raise ValueError("Invalid densification thresholds")
    opacity = p["op_raw"].sigmoid()
    keep = opacity >= prune_opacity
    if not keep.any().item():
        keep[opacity.argmax()] = True  # Keep one primitive so fitting can recover.
    alive = torch.nonzero(keep, as_tuple=True)[0]
    candidates = alive[grad_mag[alive] > grad_threshold]
    ranked = candidates[torch.argsort(grad_mag[candidates], descending=True)]
    chosen = ranked[:budget - len(alive)]  # Every clone/split costs one net row.
    scales = p["log_s"].exp()
    large = scales[chosen].max(dim=1).values > size_threshold
    splits, clones = chosen[large], chosen[~large]
    split_mask = torch.zeros(count, dtype=torch.bool, device=opacity.device)
    split_mask[splits] = True
    originals = alive[~split_mask[alive]]
    sources = torch.cat((originals, clones, splits.repeat_interleave(2)))
    new_rows = torch.arange(len(sources), device=opacity.device) >= len(originals)
    stats = dict(before=count, after=len(sources), pruned=count-len(alive),
                 cloned=len(clones), split=len(splits))
    if stats["pruned"] == len(clones) == len(splits) == 0:
        return p, sources, new_rows, stats

    values = {name: value.detach()[sources].clone() for name, value in p.items()}
    if len(splits):
        local = torch.randn_like(scales[splits]) * scales[splits]
        R = quaternion_to_rotation(p["quat"][splits])
        offsets = (R @ local[..., None]).squeeze(-1)
        children = 2 * len(splits)
        values["mu3"][-children:] += torch.stack((-offsets, offsets), dim=1).reshape(-1, 3)
        values["log_s"][-children:] -= math.log(split_scale)
    return {name: torch.nn.Parameter(value) for name, value in values.items()}, sources, new_rows, stats
