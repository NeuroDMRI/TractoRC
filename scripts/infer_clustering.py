import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractorc import runtime, utilis


def parse_args():
    parser = argparse.ArgumentParser(description="Tractography clustering inference.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Input tractography file (.tt.gz).")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output directory for clustered tractography.")
    parser.add_argument("--nb_points", type=int, default=14, help="Number of points per streamline.")
    parser.add_argument("--batchsize", type=int, default=50000, help="Inference batch size.")
    runtime.add_outlier_filter_args(parser)
    parser.add_argument("--device", type=str, default="0", help='Torch device (e.g. "cuda:0", "cpu", or "auto").')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batchsize < 1:
        raise ValueError("--batchsize must be positive")
    if args.nb_points < 2:
        raise ValueError("--nb_points must be at least 2")
    runtime.validate_outlier_filter_args(args)

    device = runtime.resolve_device(args.device)

    args.output.mkdir(parents=True, exist_ok=True)

    dgcnn_model, _ = runtime.load_template_model(args.checkpoint, device)

    input_tract = utilis.load_tract(str(args.input))
    input_tract = utilis.resample_streamlines(input_tract, args.nb_points)
    if len(input_tract) == 0:
        raise ValueError(f"No streamlines found in {args.input}")
    feature_center, feature_scale = utilis.streamline_feature_stats(input_tract)
    labels, confidences = runtime.predict_streamline_clusters(
        dgcnn_model,
        input_tract,
        feature_center,
        feature_scale,
        args.batchsize,
        device,
    )
    result = utilis.write_cluster_results(
        [input_tract],
        [labels],
        [confidences],
        args.output,
        dgcnn_model.num_clusters,
        outlier_mode=args.outlier_mode,
        min_confidence=args.min_confidence,
        mad_z=args.mad_z,
        min_cluster_size=args.min_cluster_size,
        min_keep=args.min_keep,
    )
    print(f"Wrote {result['clusters']} clusters; rejected {result['outliers']} streamlines.")

if __name__ == "__main__":
    main()
