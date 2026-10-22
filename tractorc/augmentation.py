import math

import torch

from .keypoint_aligners import AffineTransform


def _batched_eye(batch_size, device):
    return torch.eye(4, device=device).repeat(batch_size, 1, 1)


# Rotation plane (row_a, row_b, sign) for each axis. Filling the 2x2 block on
# rows a/b with [[cos, -sign*sin], [sign*sin, cos]] reproduces the standard
# right-handed rotation matrices about x, y, and z.
_ROTATION_PLANES = ((1, 2, 1), (0, 2, -1), (0, 1, 1))


def _axis_rotation(theta, plane, device):
    """Batched 4x4 rotation by ``theta`` (rad) around one axis."""
    row_a, row_b, sign = plane
    matrix = _batched_eye(theta.shape[0], device)
    cos, sin = torch.cos(theta), torch.sin(theta)
    matrix[:, row_a, row_a] = cos
    matrix[:, row_a, row_b] = -sign * sin
    matrix[:, row_b, row_a] = sign * sin
    matrix[:, row_b, row_b] = cos
    return matrix


class AffineDeformation:
    def __init__(self, device="cuda:0"):
        self.device = device

    def build_affine_matrix(self, batch_size, params):
        """Return batched 4x4 affines: M = Translation · Shear · Scale · Rz·Ry·Rx.

        params: tuple of torch.FloatTensor
          scales  (batch_size, 3)
          offsets (batch_size, 3)
          angles  (batch_size, 3)
          shear   (batch_size, 6)
        """
        scale, offset, theta, shear = params
        device = self.device

        scale_matrix = _batched_eye(batch_size, device)
        scale_matrix[:, :3, :3] = torch.diag_embed(scale)

        translation_matrix = _batched_eye(batch_size, device)
        translation_matrix[:, :3, 3] = offset

        shear_matrix = _batched_eye(batch_size, device)
        shear_matrix[:, 0, 1], shear_matrix[:, 0, 2] = shear[:, 0], shear[:, 1]
        shear_matrix[:, 1, 0], shear_matrix[:, 1, 2] = shear[:, 2], shear[:, 3]
        shear_matrix[:, 2, 0], shear_matrix[:, 2, 1] = shear[:, 4], shear[:, 5]

        rotation_x = _axis_rotation(theta[:, 0], _ROTATION_PLANES[0], device)
        rotation_y = _axis_rotation(theta[:, 1], _ROTATION_PLANES[1], device)
        rotation_z = _axis_rotation(theta[:, 2], _ROTATION_PLANES[2], device)
        rotation = torch.bmm(rotation_z, torch.bmm(rotation_y, rotation_x))

        return torch.bmm(
            translation_matrix,
            torch.bmm(shear_matrix, torch.bmm(scale_matrix, rotation)),
        )

    def __call__(self, points, return_affine_matrix=False, params=None):
        """Apply the batched affine to ``(B, N, P, 3)`` points."""
        batch_size, num_lines, num_points, dim = points.shape
        matrix = self.build_affine_matrix(batch_size, params)

        flat = points.reshape(batch_size, num_lines * num_points, dim)
        transformed = AffineTransform(matrix=matrix).get_forward_transformed_points(flat)

        out = transformed.reshape(points.shape)
        if return_affine_matrix:
            return out, matrix
        return out


# Convenience functions
def random_affine_augment(
    points,
    max_random_params=((0.15, 0.15), 20, (math.pi/12, math.pi/12, math.pi/12), 0.1),
    device="cuda:0",
    return_affine_matrix=False,
    same_scale=False,
):
    """Apply the configured random affine augmentation.

    ``max_random_params`` is ``(scale_range, offset, angle_range, shear)``:
    scale draws ``uniform(1 - scale_range[0], 1 + scale_range[1])`` per axis,
    offsets and shear draw ``uniform(-v, v)``, angles ``uniform(-v, v)`` per
    axis. ``same_scale=True`` with zero shear produces a similarity transform.
    """
    scale_range, offset_range, angle_range, shear_range = max_random_params
    batch_size = points.shape[0] if points.dim() == 4 else 1

    if same_scale:
        scale = torch.empty(batch_size, 1, device=device).uniform_(
            1 - scale_range[0], 1 + scale_range[1]
        ).repeat(1, 3)
    else:
        scale = torch.empty(batch_size, 3, device=device).uniform_(
            1 - scale_range[0], 1 + scale_range[1]
        )
    offset = torch.empty(batch_size, 3, device=device).uniform_(-offset_range, offset_range)
    shear = torch.empty(batch_size, 6, device=device).uniform_(-shear_range, shear_range)
    theta = torch.empty(batch_size, 3, device=device)
    for dim in range(3):
        theta[:, dim].uniform_(-angle_range[dim], angle_range[dim])

    augmenter = AffineDeformation(device=device)
    params = (scale, offset, theta, shear)
    return augmenter(points, return_affine_matrix=return_affine_matrix, params=params)


def transform_normalization_stats(center, scale, affine_matrix, eps=1e-6):
    """Propagate fixed normalization statistics through an affine transform.

    Args:
        center: (B, 3) full-tractogram feature centers.
        scale: (B,) full-tractogram isotropic feature scales.
        affine_matrix: (B, 4, 4) forward affine matrix from
            random_affine_augment.

    The center is exact for every affine.  A scalar 95th-percentile radius can
    only be propagated exactly through a similarity transform; for the
    existing non-uniform-scale/shear augmentation preset we use the RMS linear
    stretch.  This retains a fixed full-tractogram reference scale without
    silently recomputing statistics from the random training sample.
    (``random_affine_augment(..., same_scale=True)`` with zero shear produces
    a similarity transform, for which the propagated scale is exact.)
    """
    linear = affine_matrix[:, :3, :3]
    rms_stretch = torch.sqrt(
        linear.square().sum(dim=(1, 2)) / 3
    ).clamp_min(eps)

    transformed_center = (
        torch.bmm(linear, center.unsqueeze(-1)).squeeze(-1)
        + affine_matrix[:, :3, 3]
    )
    transformed_scale = scale * rms_stretch
    return transformed_center, transformed_scale
