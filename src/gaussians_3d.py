"""3D Gaussian covariances and differentiable pinhole projection."""

import torch

def quaternion_to_rotation(q):
    """Convert (N, 4) quaternions in (w, x, y, z) order to (N, 3, 3).

    Normalize raw quaternions first. Zero quaternions produce the identity;
    initialize learned rotations to (1, 0, 0, 0) so gradients can change them.
    """
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    q = q / torch.where(norm > 0, norm, torch.ones_like(norm))
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z),     2 * (x*z + w*y),
        2 * (x*y + w*z),     1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y),     2 * (y*z + w*x),     1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)

def covariance_3d(scale, quat):
    """Positive scales (N, 3) and quaternions (N, 4) -> covariances (N, 3, 3)."""
    R = quaternion_to_rotation(quat)
    return R @ torch.diag_embed(scale.square()) @ R.transpose(-1, -2)

def project_gaussian(mu3, Sigma3, R_wc, t, K):
    """Project world-space means/covariances through one known camera.

    mu3: (N, 3); Sigma3: (N, 3, 3); R_wc: (3, 3); t: (3,); K: (3, 3).
    K follows the assignment: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    Inputs must share a floating-point dtype and device.
    Returns pixel means (N, 2), pixel covariances (N, 2, 2), and depths (N,).

    Perspective division is undefined at zero depth. Before calling this
    function for rendering, filter to camera-space z > a small near distance
    (e.g. 1e-3). Sort the returned depths ascending for front-to-back blending.
    """
    mu_cam = mu3 @ R_wc.T + t
    x, y, z = mu_cam.unbind(dim=-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu2 = torch.stack((fx * x / z + cx, fy * y / z + cy), dim=-1)

    # Jacobian of (fx*x/z + cx, fy*y/z + cy) with respect to (x, y, z).
    zero = torch.zeros_like(z)
    J = torch.stack((fx / z, zero, -fx * x / z.square(),
                     zero, fy / z, -fy * y / z.square()), dim=-1).reshape(-1, 2, 3)
    Sigma_cam = R_wc @ Sigma3 @ R_wc.T
    Sigma2 = J @ Sigma_cam @ J.transpose(-1, -2)
    return mu2, Sigma2, z
