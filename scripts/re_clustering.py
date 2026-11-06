import pathlib
import argparse
import sys

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractorc import dataset, model, runtime, train_utilis, utilis


def parse_args():
    parser = argparse.ArgumentParser(description="Tractography clustering inference.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--inputdir", type=pathlib.Path, required=True, help="Input tractography dataset directory.")
    parser.add_argument("--clusters", type=int, default=800, help="Number of clusters used in the trained model.")
    parser.add_argument("--output_path", type=pathlib.Path, required=True, help="Output model path.")
    parser.add_argument("--clustering_fiber_interval", type=int, default=4, help="Interval for fibers used in clustering.")
    parser.add_argument("--cluster_kmeans_n_init", type=int, default=10, help="Number of kmeans initializations.")
    parser.add_argument("--nb_points", type=int, default=14, help="Number of points per streamline.")
    parser.add_argument("--batchsize", type=int, default=50000, help="Inference batch size.")
    parser.add_argument("--init", type=str, default=None, help="Initialize algorithm: kmeans or GMM")
    parser.add_argument("--device", type=str, default="auto", help='Torch device (e.g. "cuda:0", "cpu", or "auto").')
    return parser.parse_args()


def main():
    args = parse_args()

    device = runtime.resolve_device(args.device)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = runtime.validate_model_config(ckpt["model_config"])

    dgcnn_model = model.TractoRC(
        num_keypoints=model_config["num_keypoints"],
        num_clusters=args.clusters,
        dim=model_config["dim"],
        feat_len=model_config["feat_len"],
        use_amp=device.type == "cuda",
        cluster_temperature=model_config["cluster_temperature"],
        keypoint_temperature=model_config["keypoint_temperature"],
        keypoint_head_append_coords=model_config["keypoint_head_append_coords"],
    ).to(device)

    state_dict = ckpt["model_state_dict"]
    if state_dict["cluster_head.weight"].shape != dgcnn_model.cluster_head.weight.shape:
        del state_dict["cluster_head.weight"]
    dgcnn_model.load_state_dict(state_dict, strict=False)
    dgcnn_model.eval()

    tract_dataset = dataset.PairedTractographyDataset(
        path=str(args.inputdir),
        point_per_streamline=args.nb_points,
        min_length=0,
        max_length=300,
        num_samples=args.batchsize,
        pretrain=True
    )

    dgcnn_model = train_utilis.init_cluster_centroid(dgcnn_model, tract_dataset, device, args.batchsize, args.clustering_fiber_interval, args.cluster_kmeans_n_init, args.init)

    torch.save({
        'model_state_dict': dgcnn_model.state_dict(),
        'model_config': train_utilis._checkpoint_model_config(dgcnn_model),
    }, args.output_path)
    print(f"Re-clustered model saved to {args.output_path}")


if __name__ == "__main__":
    main()
