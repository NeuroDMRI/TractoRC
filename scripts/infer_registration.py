import argparse
import pathlib
import sys

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractorc import dataset, keypoint_aligners, model, runtime, train_utilis, utilis


torch.set_float32_matmul_precision("high")

DEFAULT_NB_POINTS = 14
DEFAULT_NUM_SAMPLES = 2200
DEFAULT_MIN_LENGTH = 20.0
DEFAULT_MAX_LENGTH = 260.0
DEFAULT_APPLY_BATCH_STREAMLINES = 10_000
TRANSFORM_FORMAT_VERSION = 1
TRACT_EXTENSIONS = (".vtk", ".vtp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register moving tractography to fixed tractography."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--moving", type=pathlib.Path, required=True)
    parser.add_argument("--fixed", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--transform_output",
        type=pathlib.Path,
        default=None,
        help="Path for the fitted transform (.pt); defaults next to --output.",
    )
    parser.add_argument(
        "--regis_type",
        nargs="+",
        default=("tps_0.5",),
        metavar="TYPE",
        help=(
            "Ordered registration stages, e.g. '--regis_type affine tps_0.5'. "
            "Comma-separated stages are also accepted. In the "
            "affine_refine_tps pipeline this specifies the fine (residual) "
            "stage fitted after moving re-extraction."
        ),
    )
    parser.add_argument(
        "--registration_pipeline",
        choices=model.REGISTRATION_PIPELINES,
        default="affine_refine_tps",
        help=(
            "single_tps fits the --regis_type stages on one keypoint set; "
            "affine_refine_tps (default) fits an affine on the original "
            "keypoints, re-extracts "
            "moving features/keypoints from the coarse-aligned sample, then "
            "fits the --regis_type stages on the new keypoints."
        ),
    )
    parser.add_argument(
        "--refine_feature_stats",
        choices=model.REFINE_FEATURE_STATS,
        default="fixed",
        help=(
            "Statistics used to re-encode coarse-aligned moving fibers in "
            "the affine_refine_tps pipeline (default: fixed)."
        ),
    )
    parser.add_argument("--nb_points", type=int, default=DEFAULT_NB_POINTS)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--min_length", type=float, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--max_length", type=float, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device: "auto", "cpu", a GPU index, or e.g. "cuda:0".',
    )
    parser.add_argument(
        "--apply_dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory of Anatomical Tracts transformed with the fitted transform.",
    )
    parser.add_argument(
        "--apply_output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory corresponding to --apply_dir.",
    )
    parser.add_argument(
        "--apply_batch_streamlines",
        type=int,
        default=DEFAULT_APPLY_BATCH_STREAMLINES,
        help="Number of complete streamlines transformed per batch.",
    )
    parser.add_argument(
        "--output_keypoints",
        action="store_true",
        help=(
            "Save moving, fixed, and aligned moving keypoints as VTP files "
            "next to --output."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def validate_args(args):
    for path_name in ("checkpoint", "moving", "fixed"):
        path = getattr(args, path_name)
        if not path.is_file():
            raise ValueError(f"{path_name} file does not exist: {path}")

    if (args.apply_dir is None) != (args.apply_output_dir is None):
        raise ValueError("--apply_dir and --apply_output_dir must be provided together")
    if args.apply_dir is not None and not args.apply_dir.is_dir():
        raise ValueError(f"Apply directory does not exist: {args.apply_dir}")
    if args.nb_points < 2:
        raise ValueError("--nb_points must be at least 2")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.min_length < 0 or args.max_length < args.min_length:
        raise ValueError("Require 0 <= --min_length <= --max_length")
    if args.apply_batch_streamlines <= 0:
        raise ValueError("--apply_batch_streamlines must be positive")
    args.regis_type = parse_registration_types(args.regis_type)


def resolve_sample_count(args, checkpoint):
    """Resolve the keypoint-extraction sample count for registration inference."""
    resolved = runtime.resolve_num_samples(
        args.num_samples, DEFAULT_NUM_SAMPLES, checkpoint
    )
    if resolved != args.num_samples:
        print(
            f"[num_samples] Using checkpoint training value {resolved} "
            f"instead of the {DEFAULT_NUM_SAMPLES} default."
        )
    return resolved


def warn_inference_config_mismatch(args, checkpoint):
    """Warn when inference settings differ from the checkpoint's training run.

    The checkpoint records the training configuration, so a silent
    single_tps/per_side evaluation of an affine_refine_tps/fixed model is
    reported instead of producing misleading scores. The fine-stage lambda is
    deliberately not compared: it is a deployment operating point to be
    tuned on the validation set after training, not part of the training
    identity.
    """
    training_config = checkpoint.get("training_config") or {}
    checks = (
        ("registration_pipeline", "registration_pipeline"),
        ("refine_feature_stats", "refine_feature_stats"),
    )
    for arg_name, config_key in checks:
        trained = training_config.get(config_key)
        if trained is None:
            continue
        current = getattr(args, arg_name)
        current = tuple(current) if isinstance(current, (list, tuple)) else (current,)
        trained = tuple(trained) if isinstance(trained, (list, tuple)) else (trained,)
        if current != trained:
            print(
                f"[Checkpoint] WARNING: inference --{arg_name}={current} "
                f"differs from the checkpoint's trained {config_key}={trained}."
            )


def load_registration_data(path, args):
    return dataset.TractographyData(
        input_path=str(path),
        point_per_streamline=args.nb_points,
        min_length=args.min_length,
        max_length=args.max_length,
        recenter=False,
    )


def parse_registration_types(regis_type):
    """Return the ordered, non-empty registration-stage sequence."""
    if regis_type is None:
        raise ValueError("At least one registration type is required.")
    if isinstance(regis_type, str):
        regis_type = (regis_type,)

    stages = tuple(
        stage.strip()
        for value in regis_type
        for stage in str(value).split(",")
        if stage.strip()
    )
    if not stages:
        raise ValueError("At least one registration type is required.")
    return stages


class SequentialAligner:
    """Apply fitted keypoint aligners in their registration order."""

    def __init__(self, aligners):
        if not aligners:
            raise ValueError("SequentialAligner requires at least one aligner.")
        self.aligners = tuple(aligners)


def _aligner_stages(aligner):
    return getattr(aligner, "aligners", (aligner,))


def _cpu_tensor(tensor):
    return tensor.detach().cpu()


def save_transform(aligner, output_path, pipeline=None):
    """Save a fitted single- or multi-stage transform in the current format."""
    stages = []
    for stage in _aligner_stages(aligner):
        transform_type = getattr(stage, "transform_type", None)
        if transform_type is None:
            raise ValueError("Transform stage does not record its transform type.")

        stage_data = {
            "transform_type": transform_type,
            "pair_center": _cpu_tensor(stage.pair_center),
            "pair_scale": _cpu_tensor(stage.pair_scale),
        }
        if isinstance(stage, keypoint_aligners.TPS):
            stage_data.update(
                kind="tps",
                points_m=_cpu_tensor(stage.points_m),
                points_f=_cpu_tensor(stage.points_f),
                lmbda=_cpu_tensor(stage.lmbda),
                weights=None if stage.weights is None else _cpu_tensor(stage.weights),
                dim=stage.dim,
            )
        else:
            stage_data.update(
                kind="affine",
                transform_matrix=_cpu_tensor(stage.transform_matrix),
            )
        stages.append(stage_data)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform_data = {
        "format_version": TRANSFORM_FORMAT_VERSION,
        "stages": stages,
    }
    if pipeline is not None:
        transform_data["pipeline"] = pipeline
    torch.save(transform_data, output_path)
    return output_path


def load_transform(input_path, device):
    """Load a transform written by :func:`save_transform`."""
    transform_data = torch.load(input_path, map_location=device, weights_only=True)
    if transform_data.get("format_version") != TRANSFORM_FORMAT_VERSION:
        raise ValueError(f"Unsupported transform file: {input_path}")

    aligners = []
    for stage_data in transform_data.get("stages", ()):
        transform_type = stage_data["transform_type"]
        if stage_data["kind"] == "tps":
            weights = stage_data["weights"]
            aligner = keypoint_aligners.TPS(
                points_m=stage_data["points_m"].to(device),
                points_f=stage_data["points_f"].to(device),
                lmbda=stage_data["lmbda"].to(device),
                w=None if weights is None else weights.to(device),
                dim=stage_data["dim"],
            )
        elif stage_data["kind"] == "affine":
            matrix = stage_data["transform_matrix"].to(device)
            aligner = keypoint_aligners.AffineTransform(
                matrix=matrix,
                dim=matrix.shape[-1] - 1,
            )
        else:
            raise ValueError(f"Unsupported transform stage: {stage_data['kind']}")

        aligner.transform_type = transform_type
        aligner.pair_center = stage_data["pair_center"].to(device)
        aligner.pair_scale = stage_data["pair_scale"].to(device)
        aligners.append(aligner)

    if not aligners:
        raise ValueError(f"Transform file has no stages: {input_path}")
    return aligners[0] if len(aligners) == 1 else SequentialAligner(aligners)


def default_transform_path(output_path):
    output_path = pathlib.Path(output_path)
    return output_path.with_name(f"{output_path.stem}_transform.pt")


def sample_both(moving_data, fixed_data, num_samples):
    """Sample the same number of streamlines once from both sides."""
    sample_count = min(num_samples, moving_data.length, fixed_data.length)
    moving_sample, _ = moving_data.random_sample(sample_count)
    fixed_sample, _ = fixed_data.random_sample(sample_count)
    return moving_sample, fixed_sample


def _extract_pair_keypoints(moving_data, fixed_data, registration_model, args, device):
    """Sample both sides once and extract their keypoints.

    The coarse-to-fine pipeline reuses this single sample in every stage, so
    both stages see the same original fiber indices.
    """
    moving_sample, fixed_sample = sample_both(moving_data, fixed_data, args.num_samples)
    points_moving = extract_keypoints(
        moving_sample,
        registration_model,
        args.nb_points,
        device,
        moving_data.feature_center,
        moving_data.feature_scale,
    )
    points_fixed = extract_keypoints(
        fixed_sample,
        registration_model,
        args.nb_points,
        device,
        fixed_data.feature_center,
        fixed_data.feature_scale,
    )
    return moving_sample, fixed_sample, points_moving, points_fixed


def estimate_keypoint_transform(
    moving_data,
    fixed_data,
    registration_model,
    args,
    device,
):
    """Fit an aligner from sampled keypoints; also return the keypoints."""
    _, _, points_moving, points_fixed = _extract_pair_keypoints(
        moving_data, fixed_data, registration_model, args, device
    )

    aligner = build_aligner(
        points_moving,
        points_fixed,
        args.regis_type,
        device,
        moving_center=moving_data.feature_center,
        moving_scale=moving_data.feature_scale,
        fixed_center=fixed_data.feature_center,
        fixed_scale=fixed_data.feature_scale,
    )
    return aligner, points_moving, points_fixed


def estimate_affine_refine_transform(
    moving_data,
    fixed_data,
    registration_model,
    args,
    device,
):
    """Fit the coarse-to-fine transform (E2): affine, re-extract, residual.

    The fixed-side keypoints are extracted once and reused by both stages.
    """
    moving_sample, _, points_moving, points_fixed = _extract_pair_keypoints(
        moving_data, fixed_data, registration_model, args, device
    )
    num_lines, num_points, _ = moving_sample.shape

    coarse_aligner = fit_stage_aligner(
        points_moving,
        points_fixed,
        "affine",
        device,
        moving_center=moving_data.feature_center,
        moving_scale=moving_data.feature_scale,
        fixed_center=fixed_data.feature_center,
        fixed_scale=fixed_data.feature_scale,
    )

    coarse_sample = transform_points(
        torch.from_numpy(moving_sample).reshape(1, num_lines * num_points, 3).to(device),
        coarse_aligner,
    ).cpu().numpy().reshape(num_lines, num_points, 3)

    refine_stats = (
        (fixed_data.feature_center, fixed_data.feature_scale)
        if args.refine_feature_stats == "fixed"
        else (moving_data.feature_center, moving_data.feature_scale)
    )
    points_moving_refined = extract_keypoints(
        coarse_sample,
        registration_model,
        args.nb_points,
        device,
        *refine_stats,
    )

    fine_aligner = build_aligner(
        points_moving_refined,
        points_fixed,
        args.regis_type,
        device,
        moving_center=moving_data.feature_center,
        moving_scale=moving_data.feature_scale,
        fixed_center=fixed_data.feature_center,
        fixed_scale=fixed_data.feature_scale,
    )
    aligner = SequentialAligner([coarse_aligner, *_aligner_stages(fine_aligner)])
    return aligner, points_moving, points_fixed, points_moving_refined


def extract_keypoints(
    streamlines,
    registration_model,
    nb_points,
    device,
    feature_center,
    feature_scale,
):
    tract_tensor = torch.from_numpy(streamlines).unsqueeze(0).to(device)
    idx = train_utilis._build_backbone_idx(
        registration_model.backbone,
        nb_points,
        tract_tensor.shape[1],
        device,
    )

    with torch.no_grad():
        points = registration_model.encode_keypoints(
            tract_tensor,
            (feature_center, feature_scale),
            idx=idx,
        )[0]

    return points


def fit_stage_aligner(
    points_moving,
    points_fixed,
    transform_type,
    device,
    *,
    moving_center,
    moving_scale,
    fixed_center,
    fixed_scale,
):
    """Fit one keypoint aligner in the pair's full-tractogram coordinate frame."""
    pair_center, pair_scale = utilis.shared_tps_normalization_stats(
        moving_center,
        moving_scale,
        fixed_center,
        fixed_scale,
        device=device,
        dtype=torch.float32,
    )
    aligner = keypoint_aligners.build_keypoint_aligner(
        points_m=utilis.normalize_pairwise_points(
            points_moving.float(), pair_center, pair_scale
        ),
        points_f=utilis.normalize_pairwise_points(
            points_fixed.float(), pair_center, pair_scale
        ),
        weights=None,
        transform_type=transform_type,
    )
    aligner.transform_type = transform_type
    aligner.pair_center = pair_center
    aligner.pair_scale = pair_scale
    return aligner


def build_aligner(
    points_moving,
    points_fixed,
    regis_type,
    device,
    *,
    moving_center,
    moving_scale,
    fixed_center,
    fixed_scale,
):
    """Fit ordered single-stage aligners from one keypoint set."""
    points_current = points_moving.float()
    points_fixed = points_fixed.float()
    aligners = []
    for transform_type in parse_registration_types(regis_type):
        aligner = fit_stage_aligner(
            points_current,
            points_fixed,
            transform_type,
            device,
            moving_center=moving_center,
            moving_scale=moving_scale,
            fixed_center=fixed_center,
            fixed_scale=fixed_scale,
        )
        aligners.append(aligner)
        points_current = transform_points(points_current, aligner)

    return aligners[0] if len(aligners) == 1 else SequentialAligner(aligners)


def transform_points(points, aligner):
    for stage in _aligner_stages(aligner):
        pair_center = stage.pair_center.to(device=points.device, dtype=points.dtype)
        pair_scale = stage.pair_scale.to(device=points.device, dtype=points.dtype)
        points_norm = utilis.normalize_pairwise_points(
            points, pair_center, pair_scale
        )
        aligned_norm = stage.get_forward_transformed_points(points_norm.float())
        points = utilis.denormalize_pairwise_points(
            aligned_norm, pair_center, pair_scale
        )
    return points


def save_keypoints(points_moving, points_fixed, aligner, output_path, points_refined=None):
    """Save keypoints used to fit the registration transform as VTP files."""
    output_path = pathlib.Path(output_path)
    with torch.no_grad():
        points_aligned = transform_points(points_moving, aligner)

    keypoint_sets = {
        "moving": points_moving,
        "fixed": points_fixed,
        "aligned": points_aligned,
    }
    if points_refined is not None:
        keypoint_sets["moving_refined"] = points_refined
    saved_paths = {}
    for name, points in keypoint_sets.items():
        path = output_path.parent / f"{output_path.stem}_{name}_keypoints.vtp"
        polydata = utilis.convert_point_to_polydata(
            points.detach().reshape(-1, 3).cpu().numpy()
        )
        utilis.write_polydata(polydata, str(path))
        saved_paths[name] = path
    return saved_paths


def transform_streamline_batch(
    streamlines,
    aligner,
    device,
):
    """Transform a batch of complete streamlines, preserving their lengths."""
    arrays = [np.asarray(streamline, dtype=np.float32) for streamline in streamlines]
    lengths = [len(streamline) for streamline in arrays]
    if sum(lengths) == 0:
        return [streamline.copy() for streamline in arrays]

    points = np.concatenate(arrays, axis=0)
    with torch.no_grad():
        points_tensor = torch.from_numpy(points).to(device).view(1, -1, 3)
        aligned = (
            transform_points(points_tensor, aligner)
            .view(-1, 3)
            .cpu()
            .numpy()
        )

    offsets = np.cumsum(lengths[:-1])
    return [part.copy() for part in np.split(aligned, offsets)]


def transform_streamlines(
    streamlines,
    aligner,
    device,
    batch_streamlines,
):
    aligned_streamlines = []
    for start in range(0, len(streamlines), batch_streamlines):
        aligned_streamlines.extend(
            transform_streamline_batch(
                streamlines[start : start + batch_streamlines],
                aligner,
                device,
            )
        )
    return aligned_streamlines


def transform_tract_directory(
    input_dir,
    output_dir,
    aligner,
    device,
    batch_streamlines,
):
    tract_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.name.endswith(TRACT_EXTENSIONS)
    )
    if not tract_paths:
        raise RuntimeError(f"No .vtk/.vtp tract files found in {input_dir}")

    for input_path in tract_paths:
        output_path = output_dir / input_path.relative_to(input_dir)
        streamlines = list(utilis.load_tract(str(input_path)))
        if not streamlines:
            print(f"Skip empty anatomical tract: {input_path}")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aligned = transform_streamlines(
            streamlines,
            aligner,
            device,
            batch_streamlines,
        )
        utilis.save_tract(aligned, str(output_path))


def main():
    args = parse_args()
    validate_args(args)
    if args.seed is not None:
        np.random.seed(args.seed)

    device = runtime.resolve_device(args.device)
    registration_model, checkpoint = runtime.load_template_model(args.checkpoint, device)
    args.num_samples = resolve_sample_count(args, checkpoint)
    warn_inference_config_mismatch(args, checkpoint)
    moving_data = load_registration_data(args.moving, args)
    fixed_data = load_registration_data(args.fixed, args)
    points_refined = None
    if args.registration_pipeline == "affine_refine_tps":
        aligner, points_moving, points_fixed, points_refined = (
            estimate_affine_refine_transform(
                moving_data,
                fixed_data,
                registration_model,
                args,
                device,
            )
        )
    else:
        aligner, points_moving, points_fixed = estimate_keypoint_transform(
            moving_data,
            fixed_data,
            registration_model,
            args,
            device,
        )

    transform_output = args.transform_output or default_transform_path(args.output)
    save_transform(aligner, transform_output, pipeline=args.registration_pipeline)
    print(f"Saved transform: {transform_output}")

    aligned_moving = transform_streamlines(
        moving_data.streamlines,
        aligner,
        device,
        args.apply_batch_streamlines,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    utilis.save_tract(aligned_moving, str(args.output))
    if args.output_keypoints:
        saved_keypoints = save_keypoints(
            points_moving, points_fixed, aligner, args.output, points_refined
        )
        print(
            "Saved keypoints: "
            + ", ".join(str(path) for path in saved_keypoints.values())
        )

    if args.apply_dir is not None:
        transform_tract_directory(
            args.apply_dir,
            args.apply_output_dir,
            aligner,
            device,
            args.apply_batch_streamlines,
        )


if __name__ == "__main__":
    main()
