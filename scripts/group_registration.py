"""Group-wise registration using keypoints from a trained TractoRC model."""

import argparse
import pathlib
import sys

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import infer_registration
from tractorc import runtime, utilis


DEFAULT_NB_POINTS = 14
DEFAULT_NUM_SAMPLES = 2200
DEFAULT_MIN_LENGTH = 20.0
DEFAULT_MAX_LENGTH = 260.0
DEFAULT_TRANSFORM_BATCH_STREAMLINES = 10_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register a group of tractographies to their mean keypoint template."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--input_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--regis_type",
        nargs="+",
        default=("tps_0.05",),
        metavar="TYPE",
        help=(
            "Ordered registration stages, e.g. '--regis_type affine tps_0.05'. "
            "Comma-separated stages are also accepted."
        ),
    )
    parser.add_argument("--nb_points", type=int, default=DEFAULT_NB_POINTS)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--min_length", type=float, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--max_length", type=float, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--transform_batch_streamlines",
        type=int,
        default=DEFAULT_TRANSFORM_BATCH_STREAMLINES,
        help="Number of complete streamlines transformed per batch.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_each_iter", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    if args.nb_points < 2:
        raise ValueError("--nb_points must be at least 2")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.min_length < 0 or args.max_length < args.min_length:
        raise ValueError("Require 0 <= --min_length <= --max_length")
    if args.transform_batch_streamlines <= 0:
        raise ValueError("--transform_batch_streamlines must be positive")
    if args.num_iters <= 0:
        raise ValueError("--num_iters must be positive")
    args.regis_type = infer_registration.parse_registration_types(args.regis_type)


def transform_keypoints(points, aligner):
    with torch.no_grad():
        return infer_registration.transform_points(points, aligner)


def groupwise_registration_step(
    current_points,
    regis_type,
    device,
    subject_centers,
    subject_scales,
    template_center,
    template_scale,
):
    mean_points = current_points.mean(dim=0, keepdim=True).to(device)
    aligned_subjects = []

    for index, subject_points in enumerate(current_points):
        subject_points = subject_points.unsqueeze(0).to(device)
        aligner = infer_registration.build_aligner(
            subject_points,
            mean_points,
            regis_type,
            device,
            moving_center=subject_centers[index],
            moving_scale=subject_scales[index],
            fixed_center=template_center,
            fixed_scale=template_scale,
        )
        aligned_subjects.append(
            transform_keypoints(subject_points, aligner).squeeze(0).cpu()
        )

    return torch.stack(aligned_subjects), mean_points.cpu()


def output_stem(path):
    name = path.name
    for suffix in (".tt.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def save_results(
    mean_points,
    original_group_points,
    tract_data_items,
    subject_centers,
    subject_scales,
    template_center,
    template_scale,
    result_dir,
    device,
    args,
):
    result_dir.mkdir(parents=True, exist_ok=True)
    mean_polydata = utilis.convert_point_to_polydata(mean_points.squeeze(0).numpy())
    utilis.write_polydata(mean_polydata, str(result_dir / "mean_keypoints.vtp"))

    for index, tract_data in enumerate(tract_data_items):
        points_moving = original_group_points[index].unsqueeze(0).to(device)
        aligner = infer_registration.build_aligner(
            points_moving,
            mean_points.to(device),
            args.regis_type,
            device,
            moving_center=subject_centers[index],
            moving_scale=subject_scales[index],
            fixed_center=template_center,
            fixed_scale=template_scale,
        )
        infer_registration.save_transform(
            aligner,
            result_dir
            / f"{output_stem(pathlib.Path(tract_data.input_path))}_transform.pt",
        )
        aligned = infer_registration.transform_streamlines(
            tract_data.streamlines,
            aligner,
            device,
            args.transform_batch_streamlines,
        )
        utilis.save_tract(
            aligned,
            str(result_dir / f"{output_stem(pathlib.Path(tract_data.input_path))}_aligned.vtp"),
        )


def group_registration(input_paths, args):
    device = runtime.resolve_device(args.device)
    registration_model, checkpoint = runtime.load_template_model(args.checkpoint, device)
    args.num_samples = infer_registration.resolve_sample_count(args, checkpoint)

    group_points = []
    tract_data_items = []
    print("Extracting keypoints for each tractography...")

    for input_path in input_paths:
        tract_data = infer_registration.load_registration_data(input_path, args)
        tract_data_items.append(tract_data)
        sample_count = min(args.num_samples, tract_data.length)
        tract_sample, _ = tract_data.random_sample(sample_count)
        points = infer_registration.extract_keypoints(
            tract_sample,
            registration_model,
            args.nb_points,
            device,
            tract_data.feature_center,
            tract_data.feature_scale,
        )
        group_points.append(points.squeeze(0).cpu())

    original_group_points = torch.stack(group_points)
    current_points = original_group_points.clone()
    subject_centers = torch.from_numpy(
        np.stack([item.feature_center for item in tract_data_items])
    ).to(device=device, dtype=torch.float32)
    subject_scales = torch.as_tensor(
        [item.feature_scale for item in tract_data_items],
        device=device,
        dtype=torch.float32,
    )
    # The group has no independently stored tractogram yet.  Its reference
    # frame is therefore formed from complete input-tractogram statistics,
    # never from potentially collapsed learned keypoints.
    template_center = subject_centers.mean(dim=0, keepdim=True)
    template_scale = subject_scales.max().reshape(1)

    for iteration in range(args.num_iters):
        print(f"Group registration iteration {iteration + 1}/{args.num_iters}...")
        current_points, mean_points = groupwise_registration_step(
            current_points,
            args.regis_type,
            device,
            subject_centers,
            subject_scales,
            template_center,
            template_scale,
        )

        if args.save_each_iter:
            result_dir = args.output_dir / f"iter_{iteration + 1:02d}"
            if iteration == args.num_iters - 1:
                result_dir = args.output_dir / "final"
            save_results(
                mean_points,
                original_group_points,
                tract_data_items,
                subject_centers,
                subject_scales,
                template_center,
                template_scale,
                result_dir,
                device,
                args,
            )

    if not args.save_each_iter:
        save_results(
            mean_points,
            original_group_points,
            tract_data_items,
            subject_centers,
            subject_scales,
            template_center,
            template_scale,
            args.output_dir,
            device,
            args,
        )


def main():
    args = parse_args()
    validate_args(args)
    if args.seed is not None:
        np.random.seed(args.seed)

    input_paths = utilis.search_tract_paths(args.input_dir)
    if len(input_paths) < 2:
        raise RuntimeError(
            f"Group registration requires at least two tractographies, found {len(input_paths)}"
        )
    group_registration(input_paths, args)


if __name__ == "__main__":
    main()
