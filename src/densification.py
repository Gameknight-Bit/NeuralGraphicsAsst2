"""P4: clone, split, prune, and preserve Adam state when counts change."""

import math

import torch

@torch.no_grad()
def densify(gaussians, grad_mag, budget, image_width, grad_threshold=2e-4,
            size_threshold=0.02, split_scale=1.6, prune_opacity=0.005):
    """Prune, then clone/split high-gradient Gaussians within a count budget.

    grad_mag is the mean position-gradient norm since the previous pass.
    size_threshold is a fraction of image width, converted to pixel units.
    Small selected Gaussians are duplicated exactly in place. Large ones are
    replaced by two children offset +/- half the parent's longest scale along
    its rotated axis; both child scales are divided by split_scale.

    Returns (parameters, source_indices, new_rows, stats). The row mapping lets
    Adam retain its moments for surviving originals. Clone/split children get
    zero moments. Existing front-to-back order is preserved, with children
    placed at their parent's location in the list.
    """
    count = len(gaussians["mu"])
    if budget < count or count == 0 or image_width <= 0:
        raise ValueError("budget must cover the current nonempty set; image_width must be positive")
    if grad_mag.shape != (count,):
        raise ValueError("grad_mag must have shape (number of Gaussians,)")
    if grad_threshold < 0 or size_threshold <= 0 or split_scale <= 1 or not 0 <= prune_opacity < 1:
        raise ValueError("Invalid densification thresholds")

    opacity = gaussians["op_raw"].sigmoid()
    keep = opacity >= prune_opacity
    # Retain one primitive if all are transparent, allowing fitting to recover.
    if not keep.any().item():
        keep[opacity.argmax()] = True
    survivors = torch.nonzero(keep, as_tuple=True)[0]
    candidates = survivors[grad_mag[survivors] > grad_threshold]
    ranked = candidates[torch.argsort(grad_mag[candidates], descending=True)]
    # Both a clone and a two-child replacement increase the count by one.
    selected = set(ranked[:budget - len(survivors)].tolist())
    scales = gaussians["log_s"].exp()
    max_scales = scales.max(dim=1).values.tolist()

    sources, new_rows, shifts = [], [], []
    cloned = split = 0
    for i in survivors.tolist():
        if i in selected and max_scales[i] > size_threshold * image_width:
            sources.extend([i, i])
            new_rows.extend([True, True])
            shifts.extend([-0.5, 0.5])
            split += 1
        else:
            sources.append(i)
            new_rows.append(False)
            shifts.append(0.0)
            if i in selected:
                sources.append(i)
                new_rows.append(True)
                shifts.append(0.0)
                cloned += 1

    stats = {"before": count, "after": len(sources),
             "pruned": count - len(survivors), "cloned": cloned, "split": split}
    source_indices = torch.tensor(sources, device=opacity.device, dtype=torch.long)
    new_rows = torch.tensor(new_rows, device=opacity.device, dtype=torch.bool)
    if stats["pruned"] == cloned == split == 0:
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

