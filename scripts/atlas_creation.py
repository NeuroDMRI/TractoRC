import argparse
import pathlib
import sys
import tqdm

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractorc import dataset, runtime, utilis


def parse_args():
    parser = argparse.ArgumentParser(description="Create an atlas from a set of tractograms using a trained model.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--inputdir", type=pathlib.Path, required=True, help="Input tractography dataset directory.")
    parser.add_argument("--outdir", type=pathlib.Path, required=True, help="Output model path.")
    parser.add_argument("--nb_points", type=int, default=30, help="Number of points per streamline.")
    parser.add_argument("--batchsize", type=int, default=30000, help="Inference batch size.")
    runtime.add_outlier_filter_args(parser)
    parser.add_argument("--device", type=str, default="auto", help='Torch device (e.g. "cuda:0", "cpu", or "auto").')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batchsize < 1:
        raise ValueError("--batchsize must be positive")
    if args.nb_points < 2:
        raise ValueError("--nb_points must be at least 2")
    runtime.validate_outlier_filter_args(args)

    device = runtime.resolve_device(args.device)

    args.outdir.mkdir(parents=True, exist_ok=True)

    dgcnn_model, _ = runtime.load_template_model(args.checkpoint, device)

    # pretrain=True skips pair construction: this script only iterates over
    # tract_data subject-by-subject and never draws __getitem__ samples.
    tract_dataset = dataset.PairedTractographyDataset(
        path=str(args.inputdir),
        point_per_streamline=args.nb_points,
        min_length=20,
        max_length=260,
        pretrain=True,
    )

    streamline_batches = []
    label_batches = []
    confidence_batches = []

    print(f"Start processing {len(tract_dataset.tract_data)} tract files...")

    for tract_data in tqdm.tqdm(tract_dataset.tract_data, desc="Processing Tract Files"):
        labels, confidences = runtime.predict_streamline_clusters(
            dgcnn_model,
            tract_data.streamlines,
            tract_data.feature_center,
            tract_data.feature_scale,
            args.batchsize,
            device,
        )
        streamline_batches.append(tract_data.streamlines)
        label_batches.append(labels)
        confidence_batches.append(confidences)

    result = utilis.write_cluster_results(
        streamline_batches,
        label_batches,
        confidence_batches,
        args.outdir,
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
