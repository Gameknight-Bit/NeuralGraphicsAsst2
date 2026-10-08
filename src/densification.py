"""Clone, split, prune, and preserve Adam state when counts change."""

import math
import torch

@torch.no_grad()
def densify(gaussians, grad_mag, budget, image_width, grad_threshold=2e-4,
            size_threshold=0.02, split_scale=1.6, prune_opacity=0.005):
    """Return updated parameters, source indices, new-row flags, and statistics.

    Classify first, then apply pruning and the budget before making children.
    Children stay beside their parent in the front-to-back order.
    """
    count = len(gaussians["mu"])
    if budget < count or count == 0 or image_width <= 0:
        raise ValueError("budget must cover the current nonempty set; image_width must be positive")
    if grad_mag.shape != (count,):
        raise ValueError("grad_mag must have shape (number of Gaussians,)")
    if grad_threshold < 0 or size_threshold <= 0 or split_scale <= 1 or not 0 <= prune_opacity < 1:
        raise ValueError("Invalid densification thresholds")

    # grad_mag: per-Gaussian g_i accumulated since the last pass
    # dense = grad_mag > grad_threshold (under-fit Gaussians)
    dense = grad_mag > grad_threshold
    scales = gaussians["log_s"].exp()
    max_scale = scales.max(dim=1).values.tolist()
    # Convert the width fraction to pixels; retain the original scalar comparisons.
    large = dense.new_tensor([s > size_threshold * image_width for s in max_scale])

    # clone = dense & (max_scale <= size_threshold) (duplicate in place)
    clone = dense & ~large

    # split = dense & (max_scale >  size_threshold)  (2 children, scale / split_scale)
    split = dense & large

    # prune Gaussians with opacity < prune_opacity
    opacity = gaussians["op_raw"].sigmoid()
    keep = opacity >= prune_opacity
    # Retain one primitive if all are transparent, allowing fitting to recover.
    if not keep.any().item():
        keep[opacity.argmax()] = True
    survivors = torch.nonzero(keep, as_tuple=True)[0]
    # keep the total count <= budget
    candidates = survivors[(clone | split)[survivors]]
    ranked = candidates[torch.argsort(grad_mag[candidates], descending=True)]
    # Both a clone and a two-child replacement increase the count by one.
    selected = set(ranked[:budget - len(survivors)].tolist())
    clone, split = clone.tolist(), split.tolist()

    ## Logging and Result processing Logic (splitting, cloning, etc.) ##
    sources, new_rows, shifts = [], [], []
    cloned = split_count = 0
    for i in survivors.tolist():
        do_clone, do_split = i in selected and clone[i], i in selected and split[i]
        sources.extend([i, i] if do_clone or do_split else [i])
        new_rows.extend([True, True] if do_split else [False, True] if do_clone else [False])
        shifts.extend([-0.5, 0.5] if do_split else [0.0, 0.0] if do_clone else [0.0])
        cloned += do_clone
        split_count += do_split

    stats = {"before": count, "after": len(sources),
             "pruned": count - len(survivors), "cloned": cloned, "split": split_count}
    source_indices = torch.tensor(sources, device=opacity.device, dtype=torch.long)
    new_rows = torch.tensor(new_rows, device=opacity.device, dtype=torch.bool)
    if stats["pruned"] == cloned == split_count == 0:
        return gaussians, source_indices, new_rows, stats

    values = {name: value.detach()[source_indices].clone() for name, value in gaussians.items()}
    shift = gaussians["mu"].new_tensor(shifts)
    split_rows = shift != 0
    parent_scales = scales[source_indices]
    angle = gaussians["theta"][source_indices]
    x_axis = torch.stack((angle.cos(), angle.sin()), dim=1)
    y_axis = torch.stack((-angle.sin(), angle.cos()), dim=1)
    major_axis = torch.where((parent_scales[:, 0] >= parent_scales[:, 1])[:, None], x_axis, y_axis)
    offset = shift[:, None] * parent_scales.max(dim=1).values[:, None] * major_axis
    values["mu"] += offset
    values["log_s"][split_rows] -= math.log(split_scale)
    parameters = {name: torch.nn.Parameter(value) for name, value in values.items()}
    return parameters, source_indices, new_rows, stats

def rebuild_adam(old_opt, old_parameters, parameters, sources, new_rows, lr):
    """Rebind Adam after count changes, retaining moments for original rows."""
    opt = torch.optim.Adam(parameters.values(), lr=lr)
    for name, parameter in parameters.items():
        old_parameter = old_parameters[name]
        old_state = old_opt.state.get(old_parameter)
        if not old_state:
            continue
        state = {}
        for key, value in old_state.items():
            if isinstance(value, torch.Tensor) and value.shape == old_parameter.shape:
                resized = value[sources].clone()
                resized[new_rows] = 0
                state[key] = resized
            else:
                state[key] = value.clone() if isinstance(value, torch.Tensor) else value
        opt.state[parameter] = state
    # Adam uses one step counter per parameter tensor. Children share that
    # counter while starting with zero first/second moments.
    return opt