"""Shared runtime helpers for command-line training and inference scripts."""

from pathlib import Path

import torch
import numpy as np

from . import model
from . import utilis


CHECKPOINT_ARCHITECTURE_VERSION = 3


MODEL_CONFIG_FIELDS = (
    "architecture_version",
    "model_type",
    "num_keypoints",
    "num_clusters",
    "dim",
    "feat_len",
    "cluster_temperature",
    "keypoint_temperature",
    "keypoint_head_append_coords",
)


def set_default_device(device):
    """Make ``device`` the process-wide default CUDA device (no-op on CPU).

    PyTorch initializes CUDA against the default device (``cuda:0``), so a
    run whose tensors all live on ``cuda:5`` still creates a ~400 MB primary
    context on ``cuda:0`` the first time any CUDA call resolves without an
    index. Binding the default device before the first CUDA allocation keeps
    every implicitly-resolved allocation on the intended GPU.
    """
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def resolve_device(value="auto"):
    """Resolve ``auto``, an ordinal, or a standard PyTorch device string."""
    value = str(value)
    if value == "auto":
        return set_default_device(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    if value.isdigit():
        value = f"cuda:{value}"

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({value}), but CUDA is unavailable")
    return set_default_device(device)


def validate_model_config(config):
    """Validate that a checkpoint belongs to the current TractoRC model."""
    missing = [name for name in MODEL_CONFIG_FIELDS if name not in config]
    if missing:
        raise ValueError(
            "Checkpoint does not contain the current model architecture metadata "
            f"(missing {missing}). It predates architecture version "
            f"{CHECKPOINT_ARCHITECTURE_VERSION} and must be retrained."
        )
    version = config["architecture_version"]
    if version != CHECKPOINT_ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint architecture version {version!r} is incompatible with "
            f"the current version {CHECKPOINT_ARCHITECTURE_VERSION}. "
            "Retrain from the affine pretraining stage."
        )
    if config["model_type"] != "TractoRC":
        raise ValueError(
            f"Checkpoint model type {config['model_type']!r} is incompatible with "
            "the current TractoRC model. Retrain from the affine pretraining stage."
        )
    return config


def load_template_model(checkpoint_path, device, *, use_amp=None):
    """Load a :class:`TractoRC` and its saved architecture config."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = validate_model_config(checkpoint["model_config"])
    if use_amp is None:
        use_amp = device.type == "cuda"

    template_model = model.TractoRC(
        num_keypoints=config["num_keypoints"],
        num_clusters=config["num_clusters"],
        dim=config["dim"],
        feat_len=config["feat_len"],
        use_amp=use_amp,
        cluster_temperature=config["cluster_temperature"],
        keypoint_temperature=config["keypoint_temperature"],
        keypoint_head_append_coords=config["keypoint_head_append_coords"],
    ).to(device)
    template_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    template_model.eval()
    return template_model, checkpoint


def resolve_num_samples(num_samples, default, checkpoint):
    """Prefer the num_samples the checkpoint was trained with.

    Keypoints are extracted over a random streamline sample, so the sample
    size the model saw at training time is part of its operating point. When
    the caller did not override the default, fall back to the value saved in
    the checkpoint's training config.
    """
    if num_samples != default:
        return num_samples
    training_config = checkpoint.get("training_config") or {}
    value = training_config.get("num_samples")
    if value is None:
        return num_samples
    resolved = int(value)
    return resolved if resolved > 0 else num_samples


def autocast_dtype(device):
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def add_outlier_filter_args(parser):
    """Add the shared cluster-outlier filtering options to an argparse parser."""
    parser.add_argument(
        "--outlier-mode",
        choices=("none", "absolute", "mad", "hybrid"),
        default="mad",
        help="Outlier rule: none, absolute confidence, robust MAD, or both.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Reject confidence below this value in absolute or hybrid mode.",
    )
    parser.add_argument(
        "--mad-z",
        type=float,
        default=3.5,
        help="Reject scores below median - mad_z * 1.4826 * MAD.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="Do not apply MAD filtering to clusters smaller than this.",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=5,
        help="Minimum streamlines retained after MAD filtering within each cluster.",
    )


def validate_outlier_filter_args(args):
    """Validate the options added by :func:`add_outlier_filter_args`."""
    if args.min_confidence is not None and not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be in [0, 1]")
    if args.outlier_mode in ("absolute", "hybrid") and args.min_confidence is None:
        raise ValueError("--min-confidence is required in absolute or hybrid mode")
    if args.mad_z < 0:
        raise ValueError("--mad-z cannot be negative")
    if args.min_cluster_size < 1 or args.min_keep < 0:
        raise ValueError(
            "--min-cluster-size must be positive and --min-keep cannot be negative"
        )


def predict_streamline_clusters(
    template_model,
    streamlines,
    feature_center,
    feature_scale,
    batch_size,
    device,
):
    """Predict a label and confidence for every fixed-length streamline."""
    streamlines = np.asarray(streamlines, dtype=np.float32)

    idx = utilis.get_dgcnn_idx(
        k=template_model.backbone.k,
        num_points=streamlines.shape[1],
        batch=batch_size,
    ).to(device)

    label_chunks = []
    confidence_chunks = []
    with torch.inference_mode():
        for start in range(0, len(streamlines), batch_size):
            batch = torch.from_numpy(streamlines[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            normalized = utilis.norm_coords(
                batch, center=feature_center, scale=feature_scale
            )
            with torch.autocast(
                device_type=device.type,
                enabled=template_model.use_amp,
                dtype=autocast_dtype(device),
            ):
                features = template_model.extract_features(normalized, idx=idx)
                _, probabilities = template_model.get_streamline_label(
                    features, num_lines=len(batch)
                )
            confidence, labels = probabilities.squeeze(0).max(dim=1)
            label_chunks.append(labels.cpu().numpy())
            confidence_chunks.append(confidence.cpu().numpy())

    return (
        np.concatenate(label_chunks, axis=0),
        np.concatenate(confidence_chunks, axis=0),
    )
