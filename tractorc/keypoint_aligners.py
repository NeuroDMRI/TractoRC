"""
Get from https://github.com/alanqrwang/keymorph.git
"""
import math

import torch
import torch.nn as nn

from . import utilis


SOLVE_JITTER = 1e-6
MAX_SOLVE_ATTEMPTS = 4


def _stable_solve(matrix, rhs, *, name):
    """Solve batched linear systems with small, escalating diagonal jitter."""
    identity = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).unsqueeze(0)
    solutions = None
    solved = torch.zeros(matrix.shape[0], device=matrix.device, dtype=torch.bool)
    for attempt in range(MAX_SOLVE_ATTEMPTS):
        jitter = SOLVE_JITTER * (100.0**attempt)
        candidate, info = torch.linalg.solve_ex(
            matrix + jitter * identity, rhs, check_errors=False
        )
        finite = torch.isfinite(candidate).all(dim=(-1, -2))
        current = (info == 0) & finite
        if solutions is None:
            solutions = candidate
        else:
            solutions = torch.where(current[:, None, None], candidate, solutions)
        solved |= current
        if bool(solved.all()):
            return solutions

    failed = torch.nonzero(~solved, as_tuple=False).flatten().detach().cpu().tolist()
    raise RuntimeError(
        f"{name}: unable to solve degenerate keypoint system for batch rows "
        f"{failed} after {MAX_SOLVE_ATTEMPTS} regularized attempts."
    )


def _parse_transform_type(transform_type, batch_size, device, max_rand_tps_lmbda):
    if not isinstance(transform_type, str):
        raise ValueError("transform_type must be a string.")
    transform_type = transform_type.strip().lower()
    if transform_type in {"rigid", "affine"}:
        return transform_type, None
    if not transform_type.startswith("tps_"):
        raise ValueError(
            "Unsupported transform_type. Expected 'rigid', 'affine', "
            "'tps_<positive number>', 'tps_uniform', or 'tps_loguniform'."
        )

    specification = transform_type.removeprefix("tps_")
    if specification == "uniform":
        lmbda = torch.rand(batch_size, device=device) * max_rand_tps_lmbda
    elif specification == "loguniform":
        lower = math.log(1e-6)
        upper = math.log(max_rand_tps_lmbda)
        lmbda = torch.empty(batch_size, device=device).uniform_(lower, upper).exp()
    else:
        try:
            value = float(specification)
        except ValueError as error:
            raise ValueError(
                "TPS transform_type must be 'tps_<positive number>', "
                "'tps_uniform', or 'tps_loguniform'."
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError("TPS regularization must be a positive finite number.")
        lmbda = torch.full((batch_size,), value, device=device)
    return "tps", lmbda


def _normalized_weights(weights):
    """Row-normalize optional keypoint weights to unit mean.

    Rigid and affine fits are invariant to the overall weight scale. TPS
    reliability weighting divides lambda by each weight, so unit-mean weights
    keep the nominal lambda meaningful; uniform weights reproduce the
    unweighted fit exactly.
    """
    if weights is None:
        return None
    weights = weights.float()
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(utilis.EPS)


def build_keypoint_aligner(
    points_m, points_f, weights=None, transform_type="affine", max_rand_tps_lmbda=10
):
    """Build a moving-to-fixed aligner from normalized keypoints."""
    align_type, tps_lmbda = _parse_transform_type(
        transform_type, points_m.shape[0], points_m.device, max_rand_tps_lmbda
    )
    weights = _normalized_weights(weights)
    points_m = points_m.float()
    points_f = points_f.float()

    aligner_classes = {
        "rigid": RigidKeypointAligner,
        "affine": AffineKeypointAligner,
        "tps": TPS,
    }
    kwargs = {
        "points_m": points_m,
        "points_f": points_f,
        "w": weights,
        "dim": points_m.shape[-1],
    }
    if align_type == "tps":
        kwargs["lmbda"] = tps_lmbda
    return aligner_classes[align_type](**kwargs)


def _apply_affine(points, matrix):
    """Apply the (dim, dim+1) homogeneous part of ``matrix`` to ``points``."""
    batch_size, num_points, _ = points.shape
    ones = torch.ones(batch_size, num_points, 1, device=points.device, dtype=points.dtype)
    homogeneous = torch.cat([points, ones], dim=2)
    return torch.bmm(matrix[:, :-1, :], homogeneous.permute(0, 2, 1)).permute(0, 2, 1)


class AffineTransform(nn.Module):
    """
    Affine transformations.
    """

    def __init__(
        self,
        matrix=None,
        inverse_matrix=None,
        dim=3,
    ):
        r"""
        Args:
            One of matrix or inverse_matrix must be provided
            dim: Dimensionality of the points (2 or 3)
        r"""
        super().__init__()
        self.dim = dim
        if matrix is not None and inverse_matrix is None:
            self.transform_matrix = matrix
            identity = torch.eye(
                dim + 1, device=matrix.device, dtype=matrix.dtype
            ).expand(matrix.shape[0], -1, -1)
            self.inverse_transform_matrix = _stable_solve(
                matrix, identity, name="affine transform inverse"
            )
        elif matrix is None and inverse_matrix is not None:
            self.inverse_transform_matrix = inverse_matrix
            identity = torch.eye(
                dim + 1, device=inverse_matrix.device, dtype=inverse_matrix.dtype
            ).expand(inverse_matrix.shape[0], -1, -1)
            self.transform_matrix = _stable_solve(
                inverse_matrix, identity, name="affine transform forward"
            )
        else:
            raise ValueError("Only one of matrix or inverse_matrix should be provided")

    def _square(self, matrix):
        # matrix: (bs, dim, dim+1)
        square = torch.eye(
            self.dim + 1, device=matrix.device, dtype=matrix.dtype
        ).repeat(matrix.shape[0], 1, 1)  # (bs, dim+1, dim+1)
        square[:, : self.dim, : self.dim + 1] = matrix
        return square

    def get_forward_transformed_points(self, points):
        """
        Transforms a set of points in moving space to fixed space using the fitted matrix:
            p_f = A p_m.
        """
        return _apply_affine(points, self.transform_matrix)

    def get_inverse_transformed_points(self, points):
        """
        Transforms a set of points in fixed space to moving space using the fitted matrix:
            p_m = A^-1 p_f.
        """
        return _apply_affine(points, self.inverse_transform_matrix)


class AffineKeypointAligner(AffineTransform):
    """
    Keypoint aligner for transformations that can be represented as a matrix.

    All keypoints must be passed in with matrix/image/voxel ordering (aka 'ij' indexing).
    When indexing into a 3D volume, volume[points[0], points[1], points[2]]
    In a flattened grid, the first dimension varies the fastest.
    """

    def __init__(
        self,
        points_m,
        points_f,
        w=None,
        dim=3,
    ):
        """
        Args:
            points_m: Moving points (batch, num_points, dim)
            points_f: Fixed points (batch, num_points, dim)
            w: Weights for the points (batch, num_points)
            dim: Dimensionality of the points (2 or 3)
        """
        self.dim = dim
        self.points_f = points_f
        self.points_m = points_m

        # Note we flip the order of the points here
        inverse_transform_matrix = self._square(
            self.fit(self.points_f, self.points_m, w=w)
        ).to(points_m)
        super().__init__(
            inverse_matrix=inverse_transform_matrix,
            dim=dim,
        )

    def fit(self, x, y, w=None):
        r"""
        Find A which is the solution to argmin_A \sum_i ||y_i - Ax_i||_2 = argmin_A ||Ax - y||_F
        Computes the closed-form affine equation: A = y x^T (x x^T)^(-1).

        If w provided, solves the weighted affine equation:
          A = y diag(w) x^T  (x diag(w) x^T)^(-1).
          See https://www.wikiwand.com/en/Weighted_least_squares.

        Args:
          x, y: [n_batch, n_points, dim]
          w: [n_batch, n_points]
        Returns:
          A: [n_batch, dim, dim+1]
        """
        x = x.float()
        y = y.float()
        ones = torch.ones((*x.shape[:2], 1), device=x.device, dtype=x.dtype)
        design = torch.cat([x, ones], dim=-1)
        if w is not None:
            sqrt_w = w.float().clamp_min(0).sqrt().unsqueeze(-1)
            design = design * sqrt_w
            y = y * sqrt_w
        normal = torch.bmm(design.transpose(1, 2), design)
        rhs = torch.bmm(design.transpose(1, 2), y)
        coefficients = _stable_solve(normal, rhs, name="affine alignment")
        return coefficients.transpose(1, 2)


class RigidKeypointAligner(AffineKeypointAligner):
    def fit(self, p1, p2, w=None):
        r"""
        Find R and T which is the solution to argmin_{R, T} \sum_i ||p2_i - (R * p1_i + T)||_2
        See https://ieeexplore.ieee.org/document/4767965

        Args:
          x, y: [n_batch, n_points, dim]
          w: [n_batch, n_points]
        Returns:
          A: [n_batch, dim, dim+1]
        """
        p1 = p1.permute(0, 2, 1).float()
        p2 = p2.permute(0, 2, 1).float()

        # Calculate centroids
        if w is not None:
            w = w.float() / w.sum(dim=1, keepdim=True).clamp_min(utilis.EPS)
            w = w.unsqueeze(1)  # (B, 1, N): broadcast over the D rows of p1/p2
            p1_c = torch.sum(p1 * w, dim=2, keepdims=True)
            p2_c = torch.sum(p2 * w, dim=2, keepdims=True)
        else:
            p1_c = torch.mean(p1, dim=2, keepdim=True)
            p2_c = torch.mean(p2, dim=2, keepdim=True)

        # Subtract centroids
        q1 = p1 - p1_c
        q2 = p2 - p2_c
        if w is not None:
            q1 = q1 * w

        # Rotation from the covariance matrix's SVD, reflected to det=+1
        H = torch.bmm(q1, q2.transpose(1, 2))
        U, _, Vt = torch.linalg.svd(H)
        Ut = U.transpose(1, 2)
        V = Vt.transpose(1, 2)
        R = torch.bmm(V, Ut)
        correction = torch.eye(
            R.shape[-1], device=R.device, dtype=R.dtype
        ).expand(R.shape[0], -1, -1).clone()
        determinant = torch.det(R)
        correction[:, -1, -1] = torch.where(
            determinant < 0, -torch.ones_like(determinant), torch.ones_like(determinant)
        )
        R = torch.bmm(torch.bmm(V, correction), Ut)

        # Create augmented [R | T] matrix
        T = p2_c - torch.bmm(R, p1_c)
        return torch.cat([R, T], dim=-1)


class TPS(nn.Module):
    """
    See https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/numpy.py
    """
    def __init__(
        self,
        points_m,
        points_f,
        lmbda,
        w=None,
        dim=3,
        eval_chunk=8192,
    ):

        super().__init__()
        self.dim = dim
        self.lmbda = lmbda
        self.weights = w

        self.points_f = points_f
        self.points_m = points_m

        # Query points are evaluated in row blocks of ``eval_chunk``. The RBF
        # expansion materializes a (batch, control points, query points, dim)
        # intermediate, which for whole-tractogram query sets is larger than
        # everything else in the pipeline combined; blocking is compute
        # neutral and numerically identical.
        self.eval_chunk = int(eval_chunk) if eval_chunk else None

        # Both spline directions are fitted once; note the moving/fixed order
        # is flipped so the transforms map moving -> fixed and back.
        self.inverse_theta = self.fit(points_f, points_m, lmbda, weights=w)
        self.theta = self.fit(points_m, points_f, lmbda, weights=w)

    def fit_dim(self, c, lmbda, w=None):
        """
        Assumes last dimension of c contains target points.

          Set up and solve linear system:
            [K + lmbda*I   P] [w] = [v]
            [        P^T   0] [a]   [0]

          If w is provided, solve reliability-weighted TPS:
            [K + lmbda*diag(1 / w) P] [w] = [v]
            [                P^T   0] [a]   [0]

          See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=929618&tag=1, Eq. (8)
        Args:
          c: control points and target point (bs, T, d+1)
          lmbda: Lambda values per batch (bs)
        """
        device = c.device
        bs, T = c.shape[0], c.shape[1]
        ctrl, tgt = c[:, :, : self.dim], c[:, :, -1]

        # Build K matrix
        U = TPS.u(TPS.d(ctrl, ctrl), dim=self.dim)  # (bs, T, T)
        if w is not None:
            reliability = w.float().clamp_min(utilis.EPS)
            K = U + torch.diag_embed(lmbda.view(-1, 1) / reliability)
        else:
            I = torch.eye(T, device=device, dtype=c.dtype).expand(bs, -1, -1)
            K = U + I * lmbda.view(bs, 1, 1)

        # Build P matrix
        P = torch.ones((bs, T, self.dim + 1), device=device, dtype=c.dtype)
        P[:, :, 1:] = ctrl

        # Build v vector
        v = torch.zeros(bs, T + self.dim + 1, device=device, dtype=c.dtype)
        v[:, :T] = tgt

        A = torch.zeros(
            (bs, T + self.dim + 1, T + self.dim + 1),
            device=device,
            dtype=c.dtype,
        )
        A[:, :T, :T] = K
        A[:, :T, -(self.dim + 1) :] = P
        A[:, -(self.dim + 1) :, :T] = P.transpose(1, 2)

        return _stable_solve(A, v.unsqueeze(-1), name="TPS alignment").squeeze(-1)

    @staticmethod
    def d(a, b):
        """
        Compute pair-wise distances between points.

        Args:
          a: (bs, num_points, d)
          b: (bs, num_points, d)
        Returns:
          dist: (bs, num_points, num_points)
        """
        return torch.sqrt(
            torch.square(a[:, :, None, :] - b[:, None, :, :]).sum(-1) + 1e-6
        )

    @staticmethod
    def u(r, dim=3):
        """
        Compute radial basis function.
        """
        return r**2 * torch.log(r + 1e-6)

    def fit(self, c_src, c_dst, lmbda, weights=None):
        """
        Args:
          c_src: (bs, T, dim)
          c_dst: (bs, T, dim)
          lmbda: (bs)
        """
        stacked = [torch.cat((c_src, c_dst[..., i:i + 1]), dim=-1) for i in range(self.dim)]
        thetas = [self.fit_dim(c, lmbda, w=weights).to(c_src.device) for c in stacked]
        return torch.stack(thetas, -1)

    def transform_points(self, theta, ctrl, grid, chunk=None):
        r"""
        Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
        The TPS surface is a minimum bend interpolation surface defined by a set of control points.
        The function value for a x,y location is given by

          TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])

        Params
        ------
        theta: Nx(T+3)xd tensor, or Nx(T+2)xd tensor
          Batch size N, T+3 model parameters for T control points in dx and dy.
        ctrl: NxTxd tensor
          T control points in normalized image coordinates [0..1]
        grid: NxHxWx(d+1) tensor
          Grid locations to evaluate with homogeneous 1 in first coordinate.
        chunk: int, optional
          Evaluate the grid in row blocks of this size (default: the aligner's
          ``eval_chunk``). Pass ``0`` to force one full evaluation.

        Returns
        -------
        z: NxHxWxd tensor
          Function values at each grid location in dx and dy.
        """
        weights, affine = theta[:, : -(self.dim + 1), :], theta[:, -(self.dim + 1) :, :]
        N, T, _ = ctrl.shape
        num_points = grid.shape[1]
        if chunk is None:
            chunk = self.eval_chunk

        if not chunk or chunk >= num_points:
            U = TPS.u(TPS.d(ctrl, grid), dim=self.dim)  # NxT x (HxW)

            P = torch.ones(
                (N, num_points, self.dim + 1), device=theta.device, dtype=theta.dtype
            )
            P[:, :, 1:] = grid[:, :, : self.dim]

            # U is NxHxWxT
            b = torch.bmm(U.transpose(1, 2), weights)
            z = torch.bmm(P.view(N, -1, self.dim + 1), affine)
            return z + b

        # Blocked evaluation: numerically identical, but the (N, T, block, dim)
        # difference tensor stays small instead of spanning the whole query set.
        out = torch.empty(
            (N, num_points, self.dim), device=theta.device, dtype=theta.dtype
        )
        for start in range(0, num_points, chunk):
            block = grid[:, start : start + chunk]
            U = TPS.u(TPS.d(ctrl, block), dim=self.dim)

            P = torch.ones(
                (N, block.shape[1], self.dim + 1), device=theta.device, dtype=theta.dtype
            )
            P[:, :, 1:] = block[:, :, : self.dim]

            b = torch.bmm(U.transpose(1, 2), weights)
            z = torch.bmm(P.view(N, -1, self.dim + 1), affine)
            out[:, start : start + chunk] = z + b
        return out

    def get_inverse_transformed_points(self, points):
        # Control points in the inverse case are fixed points
        return self.transform_points(self.inverse_theta, self.points_f, points)

    def get_forward_transformed_points(self, points):
        # Control points in the forward case are moving points
        return self.transform_points(self.theta, self.points_m, points)
