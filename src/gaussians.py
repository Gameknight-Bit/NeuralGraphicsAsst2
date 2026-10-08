"""P1: 2D Gaussian covariance and weights."""

import torch

def covariance_2d(scale, theta):
    """Build Sigma = R S S^T R^T for N Gaussians.

    scale: (N, 2) strictly positive standard deviations along the local axes.
    theta: (N,) rotation angles in radians.
    Returns: (N, 2, 2) covariance matrices.

    Positive scales make Sigma positive definite, so it can be inverted when
    evaluating weights. For future optimization, obtain positive scales with
    torch.exp(log_scale) or torch.nn.functional.softplus(raw_scale).
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Each rotation is [[cos(theta), -sin(theta)],
    #                   [sin(theta),  cos(theta)]].
    rotation = torch.stack(
        (
            torch.stack((cos_theta, -sin_theta), dim=-1),
            torch.stack((sin_theta, cos_theta), dim=-1),
        ),
        dim=-2,
    )

    # S S^T = diag(sx^2, sy^2): scales are standard deviations, not variances.
    variance = torch.diag_embed(scale.square())
    return rotation @ variance @ rotation.transpose(-1, -2)



def gaussian_weight(xy, mu, Sigma):
    """Evaluate exp(-0.5 * (xy - mu)^T Sigma^-1 (xy - mu)).

    xy: (P, 2) pixel coordinates.
    mu: (N, 2) Gaussian centers.
    Sigma: (N, 2, 2) positive definite covariance matrices.
    Returns: (P, N), one weight for every pixel/Gaussian pair.

    Inputs must be floating-point tensors with matching dtype and device.
    These are unnormalized splat weights: each Gaussian has weight 1 at its
    center. There is no probability-density normalization factor.
    """
    delta = xy[:, None, :] - mu[None, :, :]  # (P, N, 2)

    # Solve Sigma * q = delta instead of explicitly computing Sigma^-1.
    # Group the P pixel displacements as columns for each Gaussian.
    rhs = delta.permute(1, 2, 0)  # (N, 2, P)
    q = torch.linalg.solve(Sigma, rhs).permute(2, 0, 1)  # (P, N, 2)
    squared_distance = (delta * q).sum(dim=-1)  # (P, N)
    return torch.exp(-0.5 * squared_distance)

