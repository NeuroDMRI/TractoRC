import gzip
import io
import os
import pathlib

import dipy.io.vtk
import math
import nibabel as nib
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
import torch
import vtk
from dipy.io.dpy import Dpy
from dipy.tracking.streamline import length, set_number_of_points, transform_streamlines


EPS = 1e-6


def parse_tt(tinytrack):
    """Read every DSI Studio TinyTrack block as MNI RAS+ streamlines."""
    tinytrack = os.fspath(tinytrack)
    if tinytrack.endswith('.tt.gz'):
        # scipy.io.loadmat accepts a seekable binary file object. Keeping the
        # decompressed data in memory avoids a temporary file, filesystem
        # contention, and accidental removal of a pre-existing ``.tt`` file.
        with gzip.open(tinytrack, 'rb') as handle:
            mat = scipy.io.loadmat(io.BytesIO(handle.read()))
    else:
        mat = scipy.io.loadmat(tinytrack)

    if 'trans_to_mni' not in mat:
        raise ValueError(f"{tinytrack} is not a DSI Studio TinyTrack file")
    tt_affine = np.asarray(mat['trans_to_mni'], dtype=np.float64).reshape(4, 4)
    block_names = sorted(
        (name for name in mat if name == 'track' or (name.startswith('track') and name[5:].isdigit())),
        key=lambda name: 0 if name == 'track' else int(name[5:]) + 1,
    )
    if not block_names:
        raise ValueError(f"{tinytrack} contains no TinyTrack blocks")

    streamlines_lps_center = []
    for name in block_names:
        buffer = np.ascontiguousarray(np.asarray(mat[name]).ravel())
        if buffer.dtype != np.uint8:
            raise ValueError(f"{tinytrack}: {name} must be uint8, got {buffer.dtype}")
        signed_buffer = buffer.view(np.int8)
        position = 0
        while position < len(buffer):
            if position + 16 > len(buffer):
                raise ValueError(f"{tinytrack}: truncated {name} record")
            track_size = int(buffer[position:position + 4].view('<u4')[0])
            if track_size <= 0 or track_size % 3:
                raise ValueError(f"{tinytrack}: invalid {name} record length")
            next_position = position + track_size + 13
            if next_position <= position or next_position > len(buffer):
                raise ValueError(f"{tinytrack}: {name} record exceeds its block")

            point_count = track_size // 3
            start = buffer[position + 4:position + 16].view('<i4')[:3]
            points = np.empty((point_count, 3), dtype=np.float32)
            points[0] = start
            if point_count > 1:
                delta = signed_buffer[position + 16:position + 16 + 3 * (point_count - 1)]
                points[1:] = start + delta.astype(np.int32).reshape(-1, 3).cumsum(axis=0)
            streamlines_lps_center.append(points / 32.0)
            position = next_position

    return transform_streamlines(streamlines_lps_center, tt_affine)


def load_tract(path, lps=False):
    """
    Load a tractography file and return the streamlines in RASMM space.
    """
    if path.endswith(('.tck', '.trk')):
        tractogram = nib.streamlines.load(path)
        streamlines = tractogram.streamlines

    elif path.endswith(('.vtk', '.vtp', '.fib')):
        streamlines = dipy.io.vtk.load_vtk_streamlines(path, to_lps=lps)

    elif path.endswith('.dpy'):
        dpy_obj = Dpy(path, mode='r')
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()

    elif path.endswith(('.tt', '.tt.gz')):
        streamlines = parse_tt(path)

    else:
        raise ValueError(f"{path} is an unsupported file format")

    return streamlines


def save_tract(streamlines, path, lps=False):
    """
    Save streamlines to a tractography file.
    """
    if path.endswith(('.tck', '.trk')):
        new_tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(new_tractogram, path)

    elif path.endswith(('.vtk', '.vtp', '.fib')):
        binary = path.endswith(('.vtk', '.fib'))
        dipy.io.vtk.save_vtk_streamlines(streamlines, path, to_lps=lps, binary=binary)

    elif path.endswith('.dpy'):
        dpy_obj = Dpy(path, mode='w')
        dpy_obj.write_tracks(streamlines)
        dpy_obj.close()

    else:
        raise ValueError(f"{path} is an unsupported file format")


def streamline_feature_stats(streamlines, center=None, percentile=95, eps=EPS):
    """Return stable per-tractogram center and isotropic feature scale."""
    streamlines = np.asarray(streamlines)
    if center is None:
        center = np.median(streamlines.reshape(-1, 3), axis=0)
    center = np.asarray(center, dtype=np.float32)

    distances = np.linalg.norm(streamlines - center, axis=-1).reshape(-1)
    scale = np.float32(max(np.percentile(distances, percentile), eps))
    return center, scale


def resample_streamlines(streamlines, num_points):
    """Resample streamlines, preserving constant coordinates for zero lengths."""
    streamlines = list(streamlines)
    if not streamlines:
        return np.empty((0, num_points, 3), dtype=np.float32)
    zero_length = np.asarray(length(streamlines)) == 0
    if zero_length.any():
        resampled = np.empty((len(streamlines), num_points, 3), dtype=np.float32)
        nonzero_indices = np.flatnonzero(~zero_length)
        if len(nonzero_indices):
            resampled[nonzero_indices] = set_number_of_points(
                [streamlines[index] for index in nonzero_indices], num_points
            )
        # DIPY's zero-length interpolation can leave coordinates uninitialized.
        for index in np.flatnonzero(zero_length):
            resampled[index] = streamlines[index][0]
        return resampled
    return np.asarray(set_number_of_points(streamlines, num_points), dtype=np.float32)


TRACT_EXTENSIONS = ("*.trk", "*.tck", "*.vtk", "*.vtp", "*.fib", "*.dpy", "*.tt.gz")


def search_tract_paths(input_dir):
    """Recursively find tractography files under ``input_dir``, sorted."""
    input_dir = pathlib.Path(input_dir)
    return sorted(
        path
        for pattern in TRACT_EXTENSIONS
        for path in input_dir.glob(f"**/{pattern}")
    )


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    num_fibers, num_dims, num_points = x.size()

    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    idx = idx[:num_fibers].to(device=x.device, dtype=torch.long)

    idx_base = torch.arange(0, num_fibers, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)
    neighbors = x.view(num_fibers * num_points, -1)[idx, :]
    neighbors = neighbors.view(num_fibers, num_points, k, num_dims)
    center = x.view(num_fibers, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbors - center, center), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def convert_fiber_to_polydata(streamlines):
    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()

    for streamline in streamlines:
        cellptids = vtk.vtkIdList()
        for point in streamline:
            cellptids.InsertNextId(outpoints.InsertNextPoint(point))
        outlines.InsertNextCell(cellptids)

    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    return outpd


def convert_point_to_polydata(points):
    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()

    for p in points:
        outpoints.InsertNextPoint(p)

    outpd.SetPoints(outpoints)
    n = outpoints.GetNumberOfPoints()
    verts = vtk.vtkCellArray()
    verts.AllocateExact(n, 1)
    for i in range(n):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    outpd.SetVerts(verts)
    return outpd


def write_polydata(polydata, filename):
    extension = os.path.splitext(filename)[1]

    if extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
    elif extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def cluster_outlier_indices(
    confidences,
    *,
    mode="mad",
    min_confidence=None,
    mad_z=3.5,
    min_cluster_size=20,
    min_keep=5,
):
    """Return rejected positions using an explicit absolute and/or MAD rule.

    MAD filtering is only applied when a cluster has enough valid, non-rejected
    streamlines. min_keep limits additional MAD rejections; explicit
    absolute-confidence and non-finite-score rejections are never restored.
    """
    if mode not in {"none", "absolute", "mad", "hybrid"}:
        raise ValueError(f"Unsupported outlier mode: {mode}")

    confidences = np.asarray(confidences, dtype=np.float64)
    if mode == "none":
        return np.empty(0, dtype=np.int64)

    rejected = ~np.isfinite(confidences)
    if mode in {"absolute", "hybrid"}:
        rejected |= confidences < min_confidence
    if mode == "absolute":
        return np.flatnonzero(rejected)

    valid_indices = np.flatnonzero(~rejected)
    if len(valid_indices) < min_cluster_size:
        return np.flatnonzero(rejected)

    valid_confidences = confidences[valid_indices]
    median = np.median(valid_confidences)
    robust_scale = 1.4826 * np.median(np.abs(valid_confidences - median))
    if robust_scale <= EPS:
        return np.flatnonzero(rejected)

    cutoff = median - mad_z * robust_scale
    candidates = valid_indices[valid_confidences < cutoff]
    max_additional_rejections = max(
        0, len(confidences) - min_keep - int(rejected.sum())
    )
    if len(candidates) and max_additional_rejections:
        lowest_confidence_first = candidates[
            np.argsort(confidences[candidates], kind="stable")
        ]
        rejected[lowest_confidence_first[:max_additional_rejections]] = True
    return np.flatnonzero(rejected)


def write_cluster_results(
    streamline_batches,
    label_batches,
    confidence_batches,
    output_dir,
    num_clusters,
    outlier_mode="mad",
    min_confidence=None,
    mad_z=3.5,
    min_cluster_size=20,
    min_keep=5,
):
    """Write clean and rejected cluster files without materializing all subjects.

    Each batch usually represents one subject. Processing one cluster at a time
    avoids the atlas path creating a second full copy of every streamline.
    """
    batches = [
        (np.asarray(streamlines), np.asarray(labels), np.asarray(confidences))
        for streamlines, labels, confidences in zip(
            streamline_batches, label_batches, confidence_batches
        )
    ]
    sample_shape = batches[0][0].shape[1:]

    output_dir = pathlib.Path(output_dir)
    cluster_dir = output_dir / "clusters"
    outlier_dir = cluster_dir / "outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)

    outlier_count = 0
    for cluster_index in range(num_clusters):
        parts = [
            (streamlines[labels == cluster_index], confidences[labels == cluster_index])
            for streamlines, labels, confidences in batches
            if np.any(labels == cluster_index)
        ]
        if parts:
            cluster = np.concatenate([part[0] for part in parts], axis=0)
            confidences = np.concatenate([part[1] for part in parts], axis=0)
        else:
            cluster = np.empty((0, *sample_shape), dtype=np.float32)
            confidences = np.empty(0, dtype=np.float32)

        rejected = cluster_outlier_indices(
            confidences,
            mode=outlier_mode,
            min_confidence=min_confidence,
            mad_z=mad_z,
            min_cluster_size=min_cluster_size,
            min_keep=min_keep,
        )
        outlier_count += len(rejected)
        clean_cluster = np.delete(cluster, rejected, axis=0)
        outlier_cluster = cluster[rejected]
        write_polydata(
            convert_fiber_to_polydata(clean_cluster),
            str(cluster_dir / f"cluster_{cluster_index + 1:05d}.vtp"),
        )
        write_polydata(
            convert_fiber_to_polydata(outlier_cluster),
            str(outlier_dir / f"cluster_{cluster_index + 1:05d}_outliers.vtp"),
        )

    return {"clusters": num_clusters, "outliers": outlier_count}


def calculate_distance(a, b, method="RMS"):
    """
    Compute direction-invariant pairwise distance between fibers.

    Args:
        a: (..., Sa, P, D)
        b: (..., Sb, P, D)
        method: 'RMS' or 'MeanSquared'; both take the minimum over the two
            point orders, so streamlines are orientation-invariant.
    """
    if method not in ("RMS", "MeanSquared"):
        raise ValueError(f"Unsupported method: {method}")

    unbatched = a.ndim == 3
    if unbatched:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)

    B, Sa, P, D = a.shape
    a_flat = a.reshape(B, Sa, -1)
    b_flat = b.reshape(B, b.shape[1], -1)
    # Flipping before flattening reverses the point order of each streamline,
    # so the distance is invariant to streamline direction.
    b_reversed_flat = b.flip(dims=[2]).reshape(B, b.shape[1], -1)

    dist = torch.minimum(
        torch.cdist(a_flat, b_flat),
        torch.cdist(a_flat, b_reversed_flat),
    )  # (B, Sa, Sb)

    result = dist / math.sqrt(P * D) if method == "RMS" else dist.square() / (P * D)
    return result.squeeze(0) if unbatched else result


def get_dgcnn_idx(k, num_points, batch):
    """Sequential neighbor indices (the point itself and k-1 // 2 on each side)."""
    if k not in (3, 5):
        raise ValueError(f"Only k=3 or k=5 is supported for sequential DGCNN, got {k}.")
    offsets = {3: (0, -1, 1), 5: (0, -2, -1, 1, 2)}[k]
    index = torch.arange(num_points)
    index = (index.unsqueeze(1) + torch.tensor(offsets)).clamp(0, num_points - 1)
    return index.repeat(batch, 1, 1)


def geometric_median(points, max_iters=1000, eps=EPS):
    y = points.mean(axis=0, keepdims=True)  # (1, D)

    for _ in range(max_iters):
        distance = cdist(points, y, metric='euclidean').ravel()  # (N,)
        if not (distance > 1e-10).all():
            return y.ravel()

        weights = 1.0 / distance
        y_next = (points * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
        if cdist(y_next[np.newaxis], y, metric='euclidean').item() < eps:
            return y_next
        y = y_next[np.newaxis]

    return y.ravel()


def shared_tps_normalization_stats(
    moving_center,
    moving_scale,
    fixed_center,
    fixed_scale,
    *,
    device=None,
    dtype=torch.float32,
    eps=1e-6,
):
    """Build a stable, shared coordinate frame for a registration pair.

    Feature statistics are computed once from each complete tractogram.  TPS
    uses their midpoint and larger robust radius so its radial basis and
    regularization lambda are independent of millimetre units and of the
    random streamline sample used to extract keypoints.

    Centers accept (D,) or (B, D); scales accept a scalar or (B,).

    Returns:
        center: ``(B, 1, 3)``
        scale: ``(B, 1, 1)``
    """
    center = (
        torch.as_tensor(moving_center, device=device, dtype=dtype).reshape(-1, 1, 3)
        + torch.as_tensor(fixed_center, device=device, dtype=dtype).reshape(-1, 1, 3)
    ) / 2
    scale = torch.maximum(
        torch.as_tensor(moving_scale, device=device, dtype=dtype).reshape(-1, 1, 1).abs(),
        torch.as_tensor(fixed_scale, device=device, dtype=dtype).reshape(-1, 1, 1).abs(),
    ).clamp_min(eps)
    return center, scale


def normalize_pairwise_points(points, center, scale):
    """Map pairwise keypoints into their shared robust coordinate frame."""
    return (points - center) / scale


def denormalize_pairwise_points(points, center, scale):
    """Map points from a shared robust coordinate frame back to geometry space."""
    return points * scale + center


def norm_coords(points, center, scale, eps=1e-6):
    """Normalize coordinates with an isotropic per-subject scale.

    ``center``/``scale`` come from the complete tractogram (see
    :func:`streamline_feature_stats`), never from a random streamline
    sample: sampling must not change feature coordinates. ``points`` is
    ``(B, N, P, D)`` or ``(N, P, D)``; ``center`` accepts ``(B, D)`` or
    ``(D,)`` and ``scale`` a scalar or ``(B,)``. Both broadcast.
    """
    squeeze_batch = points.dim() == 3
    if squeeze_batch:
        points = points.unsqueeze(0)
    center = torch.as_tensor(center, device=points.device, dtype=points.dtype).reshape(
        -1, 1, 1, points.shape[-1]
    )
    scale = torch.as_tensor(scale, device=points.device, dtype=points.dtype).reshape(
        -1, 1, 1, 1
    )
    normalized = (points - center) / scale.clamp_min(eps)
    return normalized.squeeze(0) if squeeze_batch else normalized
