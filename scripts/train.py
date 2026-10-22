import argparse
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractorc import dataset, distributed, model, runtime, train_utilis

torch.set_float32_matmul_precision("high")
# Convolution autotuning: with drop_last the training shapes are constant, so
# the per-shape benchmark cost is paid once and later steps use faster kernels.
torch.backends.cudnn.benchmark = True


DEFAULT_CONFIG = {
    "train_path": None,
    "val_path": None,
    # Training TPS regularization: sampled per fit so the keypoint head is
    # robust across a lambda range. "tps_loguniform" spreads mass uniformly
    # over [1e-6, 10] on the log scale; "tps_<value>" fixes lambda.
    "regis_type": "tps_loguniform",
    "eval_regis_type": "tps_0.5",
    # Registration pipeline: "single_tps" fits one transform (per regis_type)
    # on the original keypoints; "affine_refine_tps" fits an affine on the
    # original keypoints, re-extracts moving features/keypoints from the
    # coarse-aligned fibers, then fits a residual TPS (per regis_type).
    "registration_pipeline": "single_tps",
    # Weight of the coarse-stage registration loss in the affine_refine_tps
    # pipeline; the final-stage loss keeps weight loss_weights.registration.
    "coarse_loss_weight": 0.25,
    # Feature statistics for re-encoding coarse-aligned moving fibers.
    "refine_feature_stats": "per_side",
    "output_path": None,
    "model_name": "TractoRC.pth",

    "epochs": 1000,
    "batch_size": 4,
    "feat_len": 512,
    "cluster_temperature": 1.0,
    # Temperature 1.0 keeps keypoints as soft weighted averages: with 0.1 the
    # attention is near-argmax, keypoints get pinned to sampled points and
    # gradients through the softmax nearly vanish.
    "keypoint_temperature": 0.6,
    "keypoint_head_append_coords": True,
    "learning_rate": 1e-5,

    "num_keypoints": 128,
    "num_clusters": 800,
    "clustering_fiber_interval": 4,
    "cluster_kmeans_n_init": 10,


    "nb_points": 14,
    "min_length": 20,
    "max_length": 260,
    "num_samples": 5000,
    "dataset_load_workers": 4,
    "dataloader_workers": 0,
    "dataloader_prefetch_factor": 2,
    # "chain": one subject pair per batch -- each epoch shuffles the subjects
    # once and batch b trains shuffled[b] (fixed) against shuffled[b + 1]
    # (moving), wrapping around at the end; every epoch sees fresh adjacent
    # pairs and each subject appears once on each side. With chain pairing,
    # batch_size is the number of samples drawn from that pair per step (1 is
    # typical -- grow num_samples instead for per-pair coverage, at
    # quadratically growing chamfer/embedding cost). "all": the previous
    # enumeration of every unordered subject pair per epoch.
    "pairing": "chain",
    # Shared-template geometry losses retain world coordinates by default.
    "recenter": False,

    "use_amp": False,
    # Compile the DGCNN backbone on CUDA for speed. Checkpoints are unaffected
    # (the compiled wrapper shares the backbone's parameters), and any graph
    # the compiler cannot handle falls back to eager execution.
    "compile_backbone": True,
    "pretrain": False,
    "init_cluster_after_pretrain": True,
    "gpu": 0,
    "use_checkpoint": None,
    "resume_training": False,
    "seed": 42,
    "val_seed": 43,
    # Registration and keypoint repulsion normalize by complete-tractogram
    # scale. Embedding metric targets remain in shared-world millimetres.
    "pretrain_registration_weight": 1.0,
    "pretrain_metric_weight": 5e-4,
    "pretrain_embedding_consistency_weight": 0.02,
    "embedding_metric_delta": 1.0,
    # Shared by pretraining and formal registration.  The loss operates in
    # the complete-tractogram coordinate frame, so one value is meaningful
    # across both stages and across subjects.
    "keypoint_regularization_weight": 0.02,

    "use_keypoint_weights": False,

    "distance": "RMS",
    "pretrain_init": "kmeans",
    "freeze_clustering_after_epoch": -1,

    "precluster_epochs": 0,
    "cluster_iterations": 2,
    "cluster_outlier_std_threshold": 3.0,
    "cluster_subject_threshold": 0.6,

    "loss_components": "registration,assignment,embedding,balance",
    "loss_weights": {
            "registration": 1.0,
            "assignment": 0.01,
            "balance": 0.005,
            "embedding": 5e-4,
        },
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, learning_rate, weight_decay=0.01, fused=None):
    """AdamW with weight decay excluded from norm affine params and centroids.

    Cluster centroids are geometric initializations (kmeans / geometric
    median) positioned by the assignment loss; decaying them pulls them
    toward the origin. Normalization affine parameters are scale/shift
    statistics, not weights to regularize.

    ``fused`` defaults to True on CUDA, which fuses the optimizer's elementwise
    updates into one kernel.
    """
    no_decay = {
        parameter
        for module in model.modules()
        if isinstance(
            module,
            (
                nn.LayerNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
            ),
        )
        for parameter in module.parameters(recurse=False)
    }
    no_decay.add(model.cluster_head.weight)
    decay_parameters = [p for p in model.parameters() if p not in no_decay]
    no_decay_parameters = [p for p in model.parameters() if p in no_decay]
    if fused is None:
        fused = any(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        fused=fused,
    )



def _parse_args():
    parser = argparse.ArgumentParser(description="Train TractoRC Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help="Local CUDA rank supplied by torchrun.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    if user_config is None:
        user_config = {}

    config = DEFAULT_CONFIG.copy()
    config.update(user_config)

    if "loss_weights" in user_config:
        config["loss_weights"] = DEFAULT_CONFIG["loss_weights"].copy()
        config["loss_weights"].update(user_config["loss_weights"] or {})


    if isinstance(config.get("loss_components"), list):
        config["loss_components"] = ",".join(config["loss_components"])

    if config.get("pairing") not in ("chain", "all"):
        raise ValueError("`pairing` must be 'chain' or 'all'")

    if config["train_path"] is None:
        raise ValueError("`train_path` must be specified in config.yaml")

    if config["output_path"] is None:
        raise ValueError("`output_path` must be specified in config.yaml")

    if config["pretrain_init"] not in [None, "kmeans", "GMM"]:
        raise ValueError("`pretrain_init` must be None, 'kmeans', or 'GMM'")
    if config["registration_pipeline"] not in model.REGISTRATION_PIPELINES:
        raise ValueError(
            "`registration_pipeline` must be one of "
            f"{model.REGISTRATION_PIPELINES}"
        )
    if config["refine_feature_stats"] not in model.REFINE_FEATURE_STATS:
        raise ValueError(
            "`refine_feature_stats` must be one of "
            f"{model.REFINE_FEATURE_STATS}"
        )
    if config["coarse_loss_weight"] < 0:
        raise ValueError("`coarse_loss_weight` cannot be negative")

    if config["dataset_load_workers"] < 1:
        raise ValueError("`dataset_load_workers` must be at least 1")
    if config["dataloader_workers"] < 0:
        raise ValueError("`dataloader_workers` cannot be negative")
    if config["dataloader_prefetch_factor"] < 1:
        raise ValueError("`dataloader_prefetch_factor` must be at least 1")
    if config["cluster_temperature"] <= 0 or config["keypoint_temperature"] <= 0:
        raise ValueError("Both cluster and keypoint temperatures must be positive")
    for name in (
        "pretrain_registration_weight",
        "pretrain_metric_weight",
        "pretrain_embedding_consistency_weight",
        "keypoint_regularization_weight",
    ):
        if config[name] < 0:
            raise ValueError(f"`{name}` cannot be negative")
    if config["embedding_metric_delta"] <= 0:
        raise ValueError("`embedding_metric_delta` must be positive")
    for name, value in config["loss_weights"].items():
        if value < 0:
            raise ValueError(f"`loss_weights.{name}` cannot be negative")
    return argparse.Namespace(**config)


def _restore_model_config(args):
    if args.use_checkpoint is None:
        return

    checkpoint = torch.load(args.use_checkpoint, map_location="cpu", weights_only=False)
    model_config = runtime.validate_model_config(checkpoint["model_config"])
    for name in (
        "num_keypoints",
        "num_clusters",
        "feat_len",
        "cluster_temperature",
        "keypoint_temperature",
        "keypoint_head_append_coords",
    ):
        setattr(args, name, model_config[name])


def _configure_batch_size(args, context):
    """Keep YAML batch_size global while every DDP worker receives its shard."""
    args.global_batch_size = args.batch_size
    if context.enabled:
        if not args.pretrain and args.pairing == "chain":
            raise ValueError(
                "DDP supports only pairing='all'. Chain pairing keeps one "
                "complete subject pair per step and must run on one GPU."
            )
        if args.batch_size % context.world_size:
            raise ValueError(
                "DDP requires global batch_size to be divisible by WORLD_SIZE: "
                f"{args.batch_size} % {context.world_size} != 0."
            )
    args.local_batch_size = args.batch_size // context.world_size
    if args.local_batch_size < 1:
        raise ValueError("Per-rank batch size must be at least one.")
    args.world_size = context.world_size

def main():
    cli_args = _parse_args()
    args = load_config(cli_args.config)
    context = None
    try:
        context = distributed.initialize(args.gpu, cli_args.local_rank)
        _configure_batch_size(args, context)
        seed_everything(args.seed + context.rank)
        _restore_model_config(args)

        device = runtime.set_default_device(context.device)
        train_dataset = dataset.PairedTractographyDataset(
            path=args.train_path,
            point_per_streamline=args.nb_points,
            min_length=args.min_length,
            max_length=args.max_length,
            num_samples=args.num_samples,
            pretrain=args.pretrain,
            seed=args.seed,
            load_workers=args.dataset_load_workers,
            recenter=args.recenter,
            pairing=args.pairing,
            batch_size=args.local_batch_size,
        )
        val_dataset = None
        if args.val_path and not args.pretrain:
            val_dataset = dataset.PairedTractographyDataset(
                path=args.val_path,
                point_per_streamline=args.nb_points,
                min_length=args.min_length,
                max_length=args.max_length,
                num_samples=args.num_samples,
                seed=args.val_seed,
                deterministic=True,
                random_flip=False,
                random_swap=False,
                load_workers=args.dataset_load_workers,
                recenter=args.recenter,
                pairing="all",
            )

        registration_model = model.TractoRC(
            num_keypoints=args.num_keypoints,
            num_clusters=args.num_clusters,
            feat_len=args.feat_len,
            use_amp=args.use_amp,
            cluster_temperature=args.cluster_temperature,
            keypoint_temperature=args.keypoint_temperature,
            keypoint_head_append_coords=args.keypoint_head_append_coords,
        ).to(device)
        if args.compile_backbone and device.type == "cuda":
            registration_model.enable_compiled_backbone()
        if context.enabled:
            registration_model = DistributedDataParallel(
                registration_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        optimizer = build_optimizer(
            distributed.unwrap_model(registration_model), args.learning_rate
        )

        if args.pretrain:
            train_utilis.run_pretrain(
                train_dataset, registration_model, optimizer, args, context
            )
        else:
            train_utilis.run_train(
                train_dataset, val_dataset, registration_model, optimizer, args, context
            )
    finally:
        distributed.destroy(context)


if __name__ == "__main__":
    main()
