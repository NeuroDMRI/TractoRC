import os
import pathlib

import math

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import augmentation
from . import distributed
from . import runtime
from . import utilis
from .keypoint_aligners import AffineTransform


# Similarity-transform augmentation (uniform scale, zero shear) for the
# equivariant pretraining stage. Uniform scale propagates exactly through the
# feature-statistic update, so the target K(T·x) = T·K(x) is exactly
# achievable; non-uniform scale or shear would leave a systematic
# normalization residual that the network cannot learn away.
PRETRAIN_AUGMENT_PARAMS = (
    (0.15, 0.15),
    20,
    (math.pi / 12, math.pi / 12, math.pi / 12),
    0.0,
)


def _build_backbone_idx(backbone, num_points, batch, device):
    return utilis.get_dgcnn_idx(k=backbone.k, num_points=num_points, batch=batch).to(device)


def _checkpoint_model_config(model):
    model = distributed.unwrap_model(model)
    return {
        "architecture_version": runtime.CHECKPOINT_ARCHITECTURE_VERSION,
        "model_type": getattr(model, "MODEL_TYPE", type(model).__name__),
        "num_keypoints": getattr(model, "num_keypoints", None),
        "num_clusters": getattr(model, "num_clusters", None),
        "feat_len": getattr(model, "feat_len", None),
        "dim": getattr(model, "dim", None),
        "cluster_temperature": getattr(model, "cluster_temperature", None),
        "keypoint_temperature": getattr(model, "keypoint_temperature", None),
        "keypoint_head_append_coords": getattr(model, "append_coords", None),
    }


def _serialize_config(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _read_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    runtime.validate_model_config(checkpoint["model_config"])
    return checkpoint


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = _read_checkpoint(checkpoint_path, device)
    distributed.unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Checkpoints taken at a K-means snapshot carry no scheduler state
            # (the scheduler is not passed at that point). Fast-forward the
            # freshly built scheduler so the resumed epochs use the tail of the
            # original schedule instead of restarting it at the peak LR.
            for _ in range(int(checkpoint.get("next_epoch", 0))):
                scheduler.step()

    next_epoch = int(checkpoint["next_epoch"])
    best_val = float(checkpoint["val_loss"])
    return model, optimizer, scheduler, next_epoch, best_val


def load_model_weights(model, checkpoint_path, device="cpu"):
    """Initialize a new training stage from model weights only."""
    checkpoint = _read_checkpoint(checkpoint_path, device)
    distributed.unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    return model


def save_checkpoint(
    model,
    checkpoint_path,
    optimizer=None,
    scheduler=None,
    epoch=None,
    val_loss=None,
    metrics=None,
    config=None,
    is_best=False,
    next_epoch=None,
):
    model = distributed.unwrap_model(model)
    checkpoint = {
        "model_config": _checkpoint_model_config(model),
        "model_state_dict": model.state_dict(),
        "val_loss": float("inf") if val_loss is None else float(val_loss),
    }
    if config is not None:
        checkpoint["training_config"] = _serialize_config(vars(config))
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
        checkpoint["next_epoch"] = next_epoch if next_epoch is not None else epoch + 1
    elif next_epoch is not None:
        # Snapshot taken *before* training ``next_epoch`` (e.g. a kmeans
        # initialization checkpoint): record only the restart epoch instead of
        # a negative "last completed epoch".
        checkpoint["next_epoch"] = next_epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, checkpoint_path)
    if is_best:
        checkpoint_path = pathlib.Path(checkpoint_path)
        best_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
        )
        torch.save(checkpoint, best_path)


def _freeze_backbone_and_cluster(model):
    """Freeze the backbone and clustering head; only the keypoint head trains.

    This is all-or-nothing: it cannot express "freeze the clustering head but
    keep fine-tuning the backbone". While frozen, clustering outputs are fixed
    as well; if the backbone is unfrozen later, clustering outputs can drift,
    so re-run the clustering regression checks after such a phase.
    """
    for module in (model.backbone, model.cluster_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def _seed_worker(_):
    np.random.seed(torch.initial_seed() % (2**32))


def optimizer_step(loss, optimizer, gradscaler):
    """One backward/step/update, AMP-scaled when a scaler is active."""
    if gradscaler is None:
        loss.backward()
        optimizer.step()
    else:
        gradscaler.scale(loss).backward()
        gradscaler.step(optimizer)
        gradscaler.update()


def _prepare_ddp_gradient_layout(model, context):
    """Give DDP's 1x1 convolution gradients their canonical layout.

    CUDA's 1x1 Conv2d backward can create gradients with arbitrary strides in
    singleton spatial dimensions. DDP's bucket views use parameter strides,
    so preallocating these always-used DGCNN gradients avoids a layout
    conversion during all-reduce.
    """
    if not context.enabled:
        return
    for module in distributed.unwrap_model(model).backbone.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            module.weight.grad = torch.zeros_like(
                module.weight, memory_format=torch.contiguous_format
            )


def make_grad_scaler(device, enabled):
    """Return an AMP gradient scaler, or None when AMP is disabled.

    The scaler is pinned to an indexed device: a bare ``"cuda"`` would let it
    resolve against the process default device, which may differ from
    ``device``.
    """
    if not enabled:
        return None
    if device.type == "cuda":
        return torch.GradScaler(f"cuda:{device.index if device.index is not None else 0}")
    return torch.GradScaler("cpu")


def _pair_batch_to_device(batch_data, device):
    """Move one (fixed, moving, stats) training batch to ``device``."""
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch_data)


def build_scheduler(optimizer, epochs):
    """Cosine-anneal one optimizer continuously across the full run."""
    if epochs <= 1:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _accumulate_totals(totals, terms):
    for name, value in terms.items():
        totals[name] = totals.get(name, 0.0) + value.item()


def _running_means(totals, count):
    return {name: total / count for name, total in totals.items()}


def _make_dataloader(
    dataset,
    batch_size,
    shuffle,
    seed,
    sampler=None,
    num_workers=0,
    prefetch_factor=2,
    drop_last=False,
    persistent_workers=True,
):
    """Create a deterministic loader without duplicating resident tractograms.

    The dataset keeps all resampled tractograms in memory, so worker processes
    can use substantially more RAM than the samples they prepare. Zero workers
    is therefore the safe default; callers can opt into parallel loading.

    ``drop_last`` keeps the training input shape constant, which stabilizes
    BatchNorm statistics (no tiny final batch) and lets torch.compile reuse
    one graph. It is ignored when the dataset does not fill a batch.

    ``persistent_workers`` must be False for datasets whose sampling state the
    training loop mutates per epoch (chain pairing's ``set_epoch``): forked
    workers only observe the dataset as it was when the epoch iterator was
    created.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "drop_last": bool(drop_last) and len(dataset) >= batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker if num_workers else None,
        "generator": generator,
    }
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=bool(persistent_workers),
        )
    return DataLoader(dataset, **kwargs)


def _make_train_sampler(dataset, context, shuffle, seed):
    if not context.enabled:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )


def _compute_loss_terms(
    outputs,
    fixed_coords,
    metric_coords,
    active_losses,
    loss_weights,
    distance,
    registration_pipeline="single_tps",
    coarse_loss_weight=0.0,
    refine_feature_stats="per_side",
    embedding_metric_delta=1.0,
    keypoint_regularization_weight=0.0,
    fixed_scale=None,
    fixed_center=None,
    moving_center=None,
    moving_scale=None,
    distributed_context=None,
    gather_cluster_stats=False,
):
    """Compute one batch of joint loss terms.

    ``Reg`` always scores the final-stage aligned fibers and is the
    best-checkpoint criterion; the coarse-to-fine pipeline additionally
    reports ``RegCoarse`` on the coarse affine output, scaled by
    ``coarse_loss_weight``. Clustering terms are computed once from the
    original two-sided features, never from warped fibers. Re-encoded moving
    keypoints are normalized with the statistics selected by
    ``refine_feature_stats`` — the same statistics the model used at
    re-extraction.
    """
    loss_terms = {}
    if "registration" in active_losses:
        loss_terms["Reg"] = registration_loss(
            outputs["points_aligned"], fixed_coords, method=distance,
        ) * loss_weights["registration"]
        if registration_pipeline == "affine_refine_tps":
            loss_terms["RegCoarse"] = registration_loss(
                outputs["points_aligned_coarse"], fixed_coords, method=distance,
            ) * coarse_loss_weight
    if active_losses & {"assignment", "balance"}:
        combined_probs = torch.cat((outputs["prob_f"], outputs["prob_m"]), dim=1)
        if gather_cluster_stats:
            combined_probs = distributed.all_gather_cat(
                combined_probs, distributed_context
            )
    if "assignment" in active_losses:
        loss_terms["Assign"] = cluster_assignment_loss_dec(
            combined_probs
        ) * loss_weights["assignment"]
    if "balance" in active_losses:
        loss_terms["Balance"] = cluster_balance_loss(combined_probs) * loss_weights["balance"]
    if "embedding" in active_losses:
        emb_cat = torch.cat((outputs["emb_f"], outputs["emb_m"]), dim=1)
        loss_terms["Emb"] = emb_metric_loss(
            emb_cat, metric_coords, method=distance, delta=embedding_metric_delta
        ) * loss_weights["embedding"]
    if keypoint_regularization_weight > 0 and "registration" in active_losses:
        keypoint_loss = 0.5 * (
            keypoint_repulsion_loss(
                outputs["points_f"],
                fixed_center,
                fixed_scale,
            )
            + keypoint_repulsion_loss(
                outputs["points_m"],
                moving_center,
                moving_scale,
            )
        )
        if registration_pipeline == "affine_refine_tps":
            # Average the keypoint constraint over stages so the added refine
            # stage does not double its weight; the fixed keypoints are shared
            # with the coarse stage and counted only once. Re-encoded moving
            # keypoints live in the frame selected by ``refine_feature_stats``,
            # so normalize their repulsion with the same statistics the model
            # used at re-extraction.
            refine_center, refine_scale = (
                (fixed_center, fixed_scale)
                if refine_feature_stats == "fixed"
                else (moving_center, moving_scale)
            )
            keypoint_loss = 0.5 * (
                keypoint_loss
                + keypoint_repulsion_loss(
                    outputs["points_m_refined"], refine_center, refine_scale
                )
            )
        loss_terms["Keypoint"] = keypoint_loss * keypoint_regularization_weight
    if not loss_terms:
        raise ValueError("No active loss in this phase. Check args.loss_components.")
    return loss_terms


def extract_pretraining_outputs(
    model,
    fixed_coords,
    moving_coords,
    fixed_feature_stats,
    moving_feature_stats,
    *,
    idx=None,
    include_embeddings=True,
):
    """Extract pretraining outputs through ``forward`` for DDP synchronization."""
    return model(
        fixed_coords,
        moving_coords,
        clustering=False,
        registration=False,
        idx=idx,
        fixed_feature_stats=fixed_feature_stats,
        moving_feature_stats=moving_feature_stats,
        return_pretraining_outputs=True,
        return_pretraining_embeddings=include_embeddings,
    )


def run_train(train_dataset, val_dataset, model, optimizer, args, context):
    device = runtime.set_default_device(context.device)
    init_epoch = 0
    best_val = float("inf")

    out_dir = pathlib.Path(args.output_path)
    if context.is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier(context)

    if args.freeze_clustering_after_epoch == -1:
        args.freeze_clustering_after_epoch = args.epochs

    out_model_path = out_dir / f"{args.model_name}"
    if out_model_path.suffix != '.pth':
        out_model_path = out_model_path.with_suffix('.pth')

    enabled_losses = {
        name.strip().lower()
        for name in args.loss_components.split(",")
        if name.strip()
    }

    loss_weights = {k: float(v) for k, v in args.loss_weights.items()}

    invalid = enabled_losses - loss_weights.keys()
    if invalid:
        raise ValueError(
            f"Unsupported loss components specified: {invalid}. "
            f"Supported components are: {set(loss_weights)}"
        )

    precluster_epochs = max(0, int(args.precluster_epochs))

    # Returns None for runs of a single epoch, which need no annealing.
    scheduler = build_scheduler(optimizer, args.epochs)

    if args.use_checkpoint is not None:
        if args.resume_training:
            model, optimizer, scheduler, init_epoch, best_val = load_checkpoint(
                model, args.use_checkpoint, optimizer, scheduler, device
            )
        else:
            model = load_model_weights(model, args.use_checkpoint, device)
            if context.is_master:
                print("[Checkpoint] Initialized formal training from model weights; epoch and optimizer reset.")

    loader_workers = getattr(args, "dataloader_workers", 0)
    prefetch_factor = getattr(args, "dataloader_prefetch_factor", 2)
    chain_paired = (
        not train_dataset.pretrain
        and getattr(train_dataset, "pairing", "all") == "chain"
    )
    if chain_paired and train_dataset.batch_size != args.local_batch_size:
        raise ValueError(
            "Chain pairing requires the dataset batch_size to equal the "
            "loader batch_size."
        )
    train_sampler = _make_train_sampler(
        train_dataset, context, shuffle=not chain_paired, seed=args.seed
    )
    train_loader = _make_dataloader(
        train_dataset,
        args.local_batch_size,
        shuffle=not chain_paired,
        seed=args.seed + context.rank,
        sampler=train_sampler,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
        drop_last=not chain_paired,
        persistent_workers=not chain_paired,
    )
    val_sampler = (
        distributed.DistributedEvalSampler(
            val_dataset, context.rank, context.world_size, args.local_batch_size
        )
        if val_dataset is not None
        else None
    )
    val_loader = (
        _make_dataloader(
            val_dataset,
            args.local_batch_size,
            False,
            args.val_seed + context.rank,
            sampler=val_sampler,
            num_workers=loader_workers,
            prefetch_factor=prefetch_factor,
        )
        if val_dataset is not None
        else None
    )
    gradscaler = make_grad_scaler(device, args.use_amp)
    idx = _build_backbone_idx(
        distributed.unwrap_model(model).backbone,
        args.nb_points,
        args.local_batch_size * args.num_samples,
        device,
    )

    def initialize_clusters(epoch):
        """Initialize DEC centroids before the first joint clustering epoch."""
        if context.is_master:
            print("[Two-Stage] Initializing kmeans cluster centroids..")
            # Cap the extraction batch: features do not depend on the chunk
            # size, but a 64k-streamline forward does not fit next to the
            # training allocator's reserved memory.
            torch.cuda.empty_cache()
            base_model = distributed.unwrap_model(model)
            # Extraction batches have their own shapes; run them on the eager
            # backbone so the compiled graph is never re-specialized.
            compiled_backbone = base_model._compiled_backbone
            object.__setattr__(base_model, "_compiled_backbone", None)
            try:
                init_cluster_centroid(
                    base_model,
                    train_dataset,
                    device,
                    min(args.global_batch_size * args.num_samples, 32000),
                    args.clustering_fiber_interval,
                    args.cluster_kmeans_n_init,
                    args.pretrain_init,
                    args.cluster_iterations,
                    args.cluster_outlier_std_threshold,
                    args.cluster_subject_threshold,
                )
            finally:
                object.__setattr__(base_model, "_compiled_backbone", compiled_backbone)
            kmeans_init_checkpoint_path = out_model_path.with_name(
                f"{out_model_path.stem}_kmeans_init{out_model_path.suffix}"
            )
            save_checkpoint(
                model,
                kmeans_init_checkpoint_path,
                optimizer,
                None,
                val_loss=best_val if best_val != float("inf") else None,
                config=args,
                next_epoch=epoch,
            )
            print(
                "[Two-Stage] Saved kmeans initialization checkpoint to "
                f"{kmeans_init_checkpoint_path}."
            )
        distributed.broadcast_model_state(model, context)
        distributed.barrier(context)

    # A zero precluster phase means DEC should be active from epoch 0. The
    # old schedule skipped centroid initialization in this case, leaving the
    # requested short joint run to train against random cluster centroids.
    if precluster_epochs == 0 and init_epoch == 0:
        initialize_clusters(epoch=0)

    if context.is_master:
        print(
            "[Loss Schedule] "
            f"registration/embedding warm-up epochs: 0-{max(precluster_epochs - 1, 0)}, "
            f"assignment epochs: {precluster_epochs}-{max(args.freeze_clustering_after_epoch - 1, precluster_epochs - 1)}, "
            f"assignment/rebalance stop epoch: {args.freeze_clustering_after_epoch}"
        )
    best_start_epoch = 0 if precluster_epochs == 0 else precluster_epochs
    if context.is_master:
        print(
            "[Loss Schedule] per-phase cosine LR restarts at each phase boundary; "
            f"best-checkpoint tracking starts at epoch {best_start_epoch}, after "
            "cluster centroid initialization"
        )

    for epoch in range(init_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        in_phase1 = epoch < precluster_epochs
        assignment_active = (
            epoch >= precluster_epochs
            and epoch < args.freeze_clustering_after_epoch
            and bool(enabled_losses & {"assignment", "balance"})
        )

        if epoch == precluster_epochs and precluster_epochs > 0:
            initialize_clusters(epoch)

        model.train()

        if in_phase1:
            active_losses = {"registration", "embedding"} & enabled_losses
        elif assignment_active:
            active_losses = {"registration", "embedding", "assignment", "balance"} & enabled_losses
        else:
            active_losses = {"registration"} & enabled_losses

        training_clustering = bool(active_losses & {"assignment", "balance", "embedding"})
        training_registration = "registration" in active_losses
        needs_emb = ("embedding" in active_losses) and training_clustering
        # The DEC centroids exist only from the kmeans initialization at
        # ``precluster_epochs`` (or from epoch 0 when the warm-up is
        # disabled). Earlier epochs would record a clustering head whose
        # centroids are still random, so they never update ``best_val``.
        best_tracking = precluster_epochs == 0 or epoch >= precluster_epochs

        if not training_clustering:
            _freeze_backbone_and_cluster(distributed.unwrap_model(model))

        train_loss = 0.0
        train_loss_meters = {}

        with tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} - Training",
            disable=not context.is_master,
        ) as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                fcoords, mcoords, fcenters, fscales, mcenters, mscales = (
                    _pair_batch_to_device(batch_data, device)
                )
                mcoords_aug, aug_affine = augmentation.random_affine_augment(
                    mcoords, device=device, return_affine_matrix=True
                )
                mcenters_aug, mscales_aug = augmentation.transform_normalization_stats(
                    mcenters, mscales, aug_affine
                )

                # Metric supervision is defined in the common world frame.
                # DGCNN normalization remains internal to the model via the
                # full-tractogram feature statistics passed below.
                metric_coords = torch.cat(
                    (fcoords, mcoords),
                    dim=1,
                )

                optimizer.zero_grad(set_to_none=True)
                _prepare_ddp_gradient_layout(model, context)

                outputs = model(
                    fcoords,
                    mcoords_aug,
                    args.regis_type,
                    clustering=training_clustering,
                    registration=training_registration,
                    return_emb=needs_emb,
                    use_weights=args.use_keypoint_weights,
                    idx=idx,
                    fixed_feature_stats=(fcenters, fscales),
                    moving_feature_stats=(mcenters_aug, mscales_aug),
                    registration_pipeline=args.registration_pipeline,
                    refine_feature_stats=args.refine_feature_stats,
                )

                with torch.autocast(
                    device_type=device.type,
                    enabled=args.use_amp,
                    dtype=runtime.autocast_dtype(device),
                ):
                    loss_terms = _compute_loss_terms(
                        outputs,
                        fcoords,
                        metric_coords,
                        active_losses,
                        loss_weights,
                        args.distance,
                        registration_pipeline=args.registration_pipeline,
                        coarse_loss_weight=args.coarse_loss_weight,
                        refine_feature_stats=args.refine_feature_stats,
                        embedding_metric_delta=args.embedding_metric_delta,
                        keypoint_regularization_weight=(
                            args.keypoint_regularization_weight
                            if training_registration
                            else 0.0
                        ),
                        fixed_scale=fscales,
                        fixed_center=fcenters,
                        moving_center=mcenters_aug,
                        moving_scale=mscales_aug,
                        distributed_context=context,
                        gather_cluster_stats=context.enabled,
                    )
                    loss = sum(loss_terms.values())

                optimizer_step(loss, optimizer, gradscaler)

                train_loss += loss.item()
                _accumulate_totals(train_loss_meters, loss_terms)
                pbar.set_postfix(
                    {
                        "Loss": train_loss / (batch_idx + 1),
                        **_running_means(train_loss_meters, batch_idx + 1),
                    }
                )

        train_summary = distributed.sum_across_ranks(
            [train_loss, *train_loss_meters.values(), len(train_loader)], device, context
        )
        total_train_batches = train_summary[-1]
        if context.is_master and total_train_batches:
            global_terms = {
                name: value / total_train_batches
                for name, value in zip(train_loss_meters, train_summary[1:-1])
            }
            print(
                f"[Training] epoch={epoch + 1} Loss={train_summary[0] / total_train_batches:.6f} "
                + " ".join(f"{name}={value:.6f}" for name, value in global_terms.items())
            )

        if scheduler is not None:
            scheduler.step()

        if val_loader is None:
            if context.is_master:
                save_checkpoint(
                    model, out_model_path, optimizer, scheduler, epoch=epoch, config=args
                )
            distributed.barrier(context)
            continue

        distributed.broadcast_model_state(model, context)
        eval_model = distributed.unwrap_model(model)
        eval_model.eval()
        total_loss = 0.0
        validation_term_names = []
        if "registration" in active_losses:
            validation_term_names.append("Reg")
            if args.registration_pipeline == "affine_refine_tps":
                validation_term_names.append("RegCoarse")
            if args.keypoint_regularization_weight > 0:
                validation_term_names.append("Keypoint")
        if "assignment" in active_losses:
            validation_term_names.append("Assign")
        if "balance" in active_losses:
            validation_term_names.append("Balance")
        if "embedding" in active_losses:
            validation_term_names.append("Emb")
        term_totals = {name: 0.0 for name in validation_term_names}
        validation_examples = 0
        # inference_mode is faster than no_grad: it also disables view/inplace
        # version tracking, which nothing in the validation path needs.
        with torch.inference_mode():
            with tqdm.tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} - Validation",
                disable=not context.is_master,
            ) as pbar:
                for batch_idx, batch_data in enumerate(pbar):
                    fcoords, mcoords, fcenters, fscales, mcenters, mscales = (
                        _pair_batch_to_device(batch_data, device)
                    )
                    metric_coords = torch.cat((fcoords, mcoords), dim=1)
                    outputs = eval_model(
                        fcoords,
                        mcoords,
                        args.eval_regis_type,
                        clustering=training_clustering,
                        registration=training_registration,
                        return_emb=needs_emb,
                        use_weights=args.use_keypoint_weights,
                        idx=idx,
                        fixed_feature_stats=(fcenters, fscales),
                        moving_feature_stats=(mcenters, mscales),
                        registration_pipeline=args.registration_pipeline,
                        refine_feature_stats=args.refine_feature_stats,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        enabled=args.use_amp,
                        dtype=runtime.autocast_dtype(device),
                    ):
                        loss_terms = _compute_loss_terms(
                            outputs,
                            fcoords,
                            metric_coords,
                            active_losses,
                            loss_weights,
                            args.distance,
                            registration_pipeline=args.registration_pipeline,
                            coarse_loss_weight=args.coarse_loss_weight,
                            refine_feature_stats=args.refine_feature_stats,
                            embedding_metric_delta=args.embedding_metric_delta,
                            keypoint_regularization_weight=(
                                args.keypoint_regularization_weight
                                if training_registration
                                else 0.0
                            ),
                            fixed_scale=fscales,
                            fixed_center=fcenters,
                            moving_center=mcenters,
                            moving_scale=mscales,
                        )
                        loss = sum(loss_terms.values())
                    batch_size = fcoords.shape[0]
                    validation_examples += batch_size
                    total_loss += loss.item() * batch_size
                    for name, value in loss_terms.items():
                        term_totals[name] = term_totals.get(name, 0.0) + value.item() * batch_size

                    divisor = validation_examples
                    pbar.set_postfix(
                        {"Loss": total_loss / divisor, **_running_means(term_totals, divisor)}
                    )

        validation_summary = distributed.sum_across_ranks(
            [total_loss, *term_totals.values(), validation_examples], device, context
        )
        validation_examples = validation_summary[-1]
        if "Reg" not in term_totals or not validation_examples:
            raise RuntimeError(
                "Validation checkpoint selection requires the registration loss."
            )
        standard_loss = validation_summary[0] / validation_examples
        global_term_totals = dict(zip(term_totals, validation_summary[1:-1]))
        validation_reg = global_term_totals["Reg"] / validation_examples
        metrics = {
            "validation": {
                "standard": {
                    "loss": standard_loss,
                    **{
                        key: value / validation_examples
                        for key, value in global_term_totals.items()
                    },
                },
                "checkpoint_selection": {"Reg": validation_reg},
            }
        }
        tracking_note = (
            "best-checkpoint criterion" if best_tracking
            else "best tracking inactive before centroid init"
        )
        if context.is_master:
            print(
                f"[Validation] standard={standard_loss:.6f}, "
                f"Reg={validation_reg:.6f} ({tracking_note})"
            )
        if context.is_master and best_tracking and validation_reg < best_val:
            best_val = validation_reg
            save_checkpoint(
                model, out_model_path, optimizer, scheduler, epoch=epoch, val_loss=best_val,
                metrics=metrics, config=args, is_best=True,
            )
        if context.is_master:
            save_checkpoint(
                model, out_model_path, optimizer, scheduler, epoch=epoch, val_loss=validation_reg,
                metrics=metrics, config=args,
            )
        distributed.barrier(context)


def run_pretrain(train_dataset, model, optimizer, args, context):
    device = runtime.set_default_device(context.device)
    init_epoch = 0

    # Equivariance and keypoint repulsion use complete-tractogram statistics.
    # Metric supervision is deliberately raw-world millimetres, with its own
    # configurable Huber transition (`embedding_metric_delta`).
    lambda_reg = float(args.pretrain_registration_weight)
    lambda_metric = float(args.pretrain_metric_weight)
    lambda_consistency = float(args.pretrain_embedding_consistency_weight)
    lambda_keypoint = float(args.keypoint_regularization_weight)

    out_dir = pathlib.Path(args.output_path)
    if context.is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier(context)

    out_model_path = out_dir / f"{args.model_name}"
    if out_model_path.suffix != '.pth':
        out_model_path = out_model_path.with_suffix('.pth')

    if args.epochs <= 1:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    if args.use_checkpoint is not None:
        model, optimizer, scheduler, init_epoch, _ = load_checkpoint(model, args.use_checkpoint, optimizer, scheduler, device)

    train_sampler = _make_train_sampler(train_dataset, context, shuffle=True, seed=args.seed)
    train_loader = _make_dataloader(
        train_dataset,
        args.local_batch_size,
        True,
        args.seed + context.rank,
        sampler=train_sampler,
        num_workers=getattr(args, "dataloader_workers", 0),
        prefetch_factor=getattr(args, "dataloader_prefetch_factor", 2),
        drop_last=True,
    )
    gradscaler = make_grad_scaler(device, args.use_amp)

    idx = _build_backbone_idx(
        distributed.unwrap_model(model).backbone,
        args.nb_points,
        args.local_batch_size * args.num_samples,
        device,
    )

    for epoch in range(init_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()

        train_loss = 0.0
        total_reg_loss = 0.0
        total_metric_loss = 0.0
        total_consistency_loss = 0.0
        total_keypoint_loss = 0.0

        with tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} - Pretraining",
            disable=not context.is_master,
        ) as pbar:
            for batch_idx, (coords, centers, scales) in enumerate(pbar):
                coords = coords.to(device, non_blocking=True)
                centers = centers.to(device, non_blocking=True)
                scales = scales.to(device, non_blocking=True)
                aug_coords, aug_affine = augmentation.random_affine_augment(
                    coords,
                    device=device,
                    return_affine_matrix=True,
                    same_scale=True,
                    max_random_params=PRETRAIN_AUGMENT_PARAMS,
                )
                aug_centers, aug_scales = augmentation.transform_normalization_stats(
                    centers, scales, aug_affine
                )

                optimizer.zero_grad(set_to_none=True)
                _prepare_ddp_gradient_layout(model, context)

                # Both embeddings are supervised by the unaugmented raw-world
                # geometry. The affine augmentation changes only model input.
                metric_coords = coords
                outputs = extract_pretraining_outputs(
                    model,
                    coords,
                    aug_coords,
                    (centers, scales),
                    (aug_centers, aug_scales),
                    idx=idx,
                    include_embeddings=True,
                )

                with torch.autocast(
                    device_type=device.type,
                    enabled=args.use_amp,
                    dtype=runtime.autocast_dtype(device),
                ):
                    reg_loss = reg_invariant_loss(
                        outputs['points_m'], outputs['points_f'], aug_affine,
                        scale=aug_scales, delta=0.1,
                    ) * lambda_reg

                    # The unaugmented world geometry is the target for both
                    # views, while consistency removes embedding ambiguity.
                    metric_loss = (
                        emb_metric_loss(
                            outputs["emb_f"],
                            metric_coords,
                            method=args.distance,
                            delta=args.embedding_metric_delta,
                        )
                        + emb_metric_loss(
                            outputs["emb_m"],
                            metric_coords,
                            method=args.distance,
                            delta=args.embedding_metric_delta,
                        )
                    ) * lambda_metric / 2

                    consistency_loss = (
                        embedding_consistency_loss(
                            outputs["emb_f"], outputs["emb_m"]
                        )
                        * lambda_consistency
                    )

                    keypoint_loss = 0.5 * (
                        keypoint_repulsion_loss(
                            outputs["points_m"],
                            aug_centers,
                            aug_scales,
                        )
                        + keypoint_repulsion_loss(
                            outputs["points_f"],
                            centers,
                            scales,
                        )
                    ) * lambda_keypoint
                    loss = reg_loss + metric_loss + consistency_loss + keypoint_loss

                optimizer_step(loss, optimizer, gradscaler)

                train_loss += loss.item()
                total_reg_loss += reg_loss.item()
                total_metric_loss += metric_loss.item()
                total_consistency_loss += consistency_loss.item()
                total_keypoint_loss += keypoint_loss.item()

                divisor = batch_idx + 1
                pbar.set_postfix(
                    {
                        "Loss": train_loss / divisor,
                        "Reg": total_reg_loss / divisor,
                        "Metric": total_metric_loss / divisor,
                        "Consistency": total_consistency_loss / divisor,
                        "Keypoint": total_keypoint_loss / divisor,
                    }
                )

        if scheduler is not None:
            scheduler.step()
        if context.is_master:
            save_checkpoint(model, out_model_path, optimizer, scheduler, epoch, config=args)
        distributed.barrier(context)

    if not args.init_cluster_after_pretrain:
        if context.is_master:
            print("[Pretrain] Skip cluster centroid initialization.")
        return

    cluster_model_path = out_model_path.with_name(
        f"{out_model_path.stem}_with_clusters{out_model_path.suffix}"
    )
    if context.is_master:
        initialized_model = init_cluster_centroid(
            distributed.unwrap_model(model),
            train_dataset,
            device,
            args.global_batch_size * args.num_samples,
            args.clustering_fiber_interval,
            args.cluster_kmeans_n_init,
            args.pretrain_init,
            args.cluster_iterations,
            args.cluster_outlier_std_threshold,
            args.cluster_subject_threshold,
        )
        save_checkpoint(
            initialized_model, cluster_model_path, optimizer, scheduler,
            args.epochs - 1, config=args,
        )
        print(
            f"Initialized {initialized_model.cluster_head.out_features} cluster centroids "
            f"and saved model to {cluster_model_path}."
        )
    distributed.broadcast_model_state(model, context)
    distributed.barrier(context)


def _extract_cluster_embeddings(model, dataset, device, batch_size):
    """Extract normalized streamline embeddings with their subject IDs."""
    embedding_chunks = []
    subject_chunks = []
    idx_cache = {}

    print("Extracting embeddings for initialization...")
    with torch.no_grad():
        for subject_id, tract_data in enumerate(
            tqdm.tqdm(dataset.tract_data, desc="Extracting")
        ):
            data = tract_data.streamlines  # (N, P, 3)
            num_lines, num_points, _ = data.shape
            subject_chunks.append(
                np.full((num_lines,), subject_id, dtype=np.int32)
            )

            if num_points not in idx_cache:
                idx_cache[num_points] = _build_backbone_idx(
                    model.backbone, num_points, batch_size, device
                )
            idx = idx_cache[num_points]

            for start_idx in range(0, num_lines, batch_size):
                end_idx = min(start_idx + batch_size, num_lines)
                batch_data = torch.from_numpy(data[start_idx:end_idx]).to(device)
                batch_data = utilis.norm_coords(
                    batch_data,
                    center=tract_data.feature_center,
                    scale=tract_data.feature_scale,
                )
                with torch.autocast(
                    device_type=device.type,
                    enabled=model.use_amp,
                    dtype=runtime.autocast_dtype(device),
                ):
                    features = model.extract_features(batch_data, idx=idx)
                    embeddings = model.cluster_head._extract_embedding(features)
                embedding_chunks.append(embeddings.cpu().numpy())

    return (
        np.concatenate(embedding_chunks, axis=0).astype(np.float32, copy=False),
        np.concatenate(subject_chunks, axis=0),
    )


def init_cluster_centroid(
    model,
    dataset,
    device,
    batchsize,
    fiber_interval=4,
    n_inits=10,
    algorithm=None,
    cluster_iterations=2,
    cluster_outlier_std_threshold=3.0,
    subject_threshold=0.6,
    num_threads=None,
):
    """Run centroid initialization with a raised CPU thread limit.

    torchrun pins every rank to ``OMP_NUM_THREADS=1``, which confines
    sklearn's KMeans/GMM to a single core and stretches the centroid
    initialization into the longest phase of a run. OpenMP/BLAS threads are
    raised only inside this function -- ``num_threads`` defaults to
    ``min(100, cpu_count)`` -- so training and data loading elsewhere keep
    their usual single-threaded-BLAS behavior. The IsolationForest stage
    already parallelizes itself with ``n_jobs=-1``; the geometric-median
    refinement stays sequential on purpose: per-cluster cost is a few
    milliseconds, and a large pool measured slower than the plain loop.
    """
    if num_threads is None:
        num_threads = min(100, os.cpu_count() or 1)
    num_threads = max(1, int(num_threads))
    with threadpool_limits(limits=num_threads):
        return _init_cluster_centroid_impl(
            model,
            dataset,
            device,
            batchsize,
            fiber_interval=fiber_interval,
            n_inits=n_inits,
            algorithm=algorithm,
            cluster_iterations=cluster_iterations,
            cluster_outlier_std_threshold=cluster_outlier_std_threshold,
            subject_threshold=subject_threshold,
        )


def _init_cluster_centroid_impl(model, dataset, device, batchsize, fiber_interval=4, n_inits=10, algorithm=None, cluster_iterations=2, cluster_outlier_std_threshold=3.0, subject_threshold=0.6):
    model.eval()

    embedding_array, subject_array = _extract_cluster_embeddings(
        model, dataset, device, batchsize
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if fiber_interval > 1:
        embedding_array = embedding_array[::fiber_interval]
        subject_array = subject_array[::fiber_interval]

    n_clusters = getattr(model, "num_clusters", model.cluster_head.weight.shape[0])
    if len(embedding_array) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} embeddings to initialize clusters, "
            f"but only {len(embedding_array)} are available."
        )

    print(f"Embedding array shape (original): {embedding_array.shape}")

    print('[Pre-processing] Removing global outliers before Clustering with Isolation Forest (3-Sigma)...')
    iso_forest = IsolationForest(
        n_estimators=2000,
        max_samples=512,
        n_jobs=-1,
        random_state=42
    )
    iso_forest.fit(embedding_array)

    scores = iso_forest.decision_function(embedding_array)

    threshold = scores.mean() - cluster_outlier_std_threshold * scores.std()

    is_inlier = scores > threshold
    n_outliers = (~is_inlier).sum()

    print(f"[Isolation Forest] Removed {n_outliers} extreme outliers ({(n_outliers/len(scores))*100:.2f}%)")

    embedding_array = embedding_array[is_inlier]
    subject_array = subject_array[is_inlier]
    del iso_forest, scores, is_inlier

    if len(embedding_array) < n_clusters:
        raise ValueError(
            "Outlier filtering left fewer embeddings than requested clusters: "
            f"{len(embedding_array)} < {n_clusters}."
        )

    number_of_subjects = len(dataset.tract_data)

    selected_algorithm = "kmeans" if algorithm is None else str(algorithm).lower()
    if selected_algorithm not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _run_cluster(data):
        if selected_algorithm == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                n_init=n_inits,
                max_iter=1000,
                tol=1e-6,
                random_state=42,
            )
        else:
            clusterer = GaussianMixture(
                n_components=n_clusters,
                n_init=5,
                init_params="kmeans",
                max_iter=100,
                random_state=42,
                covariance_type="full",
                reg_covar=1e-5,
            )

        print(f"[Sklearn {type(clusterer).__name__}] Initializing cluster centroids with n_clusters={n_clusters}")
        labels_local = clusterer.fit_predict(data)
        centroid_local = (
            clusterer.means_ if selected_algorithm == "gmm" else clusterer.cluster_centers_
        )
        return labels_local, centroid_local

    active_embeddings = embedding_array
    active_subjects = subject_array

    for iteration in range(cluster_iterations):
        print(f"[init_cluster_centroid] Iteration {iteration + 1}/{cluster_iterations} - clustering {len(active_embeddings)} fibers")
        labels, centroids = _run_cluster(active_embeddings)

        reject_mask = np.zeros(len(active_embeddings), dtype=bool)
        fiber_mean_sim = np.zeros(len(active_embeddings), dtype=np.float32)

        for cidx in range(n_clusters):
            cluster_mask = (labels == cidx)
            if not np.any(cluster_mask):
                continue

            fiber_indices = np.nonzero(cluster_mask)[0]

            if len(fiber_indices) <= 2:
                reject_mask[fiber_indices] = True
                continue

            points = active_embeddings[fiber_indices]
            subject_ids = active_subjects[fiber_indices]

            subjects_per_cluster = len(np.unique(subject_ids))
            subject_ratio = subjects_per_cluster / number_of_subjects

            centroid = centroids[cidx]

            diff = points - centroid
            dist2 = np.einsum("ij,ij->i", diff, diff)
            local_sigma = max(utilis.EPS, np.sqrt(np.median(dist2)))
            sim_scores = np.exp(-dist2 / (2 * local_sigma ** 2))

            fiber_mean_sim[fiber_indices] = sim_scores

            sim_mu = sim_scores.mean()
            sim_std = sim_scores.std()
            cutoff = sim_mu - cluster_outlier_std_threshold * sim_std

            local_reject = fiber_indices[sim_scores < cutoff]
            reject_mask[local_reject] = True

            if subject_ratio < subject_threshold:
                reject_mask[fiber_indices] = True

        valid_sim_mask = fiber_mean_sim > 0
        if valid_sim_mask.any():
            valid_scores = fiber_mean_sim[valid_sim_mask]
            global_cutoff = valid_scores.mean() - cluster_outlier_std_threshold * valid_scores.std()
            global_reject = np.nonzero(
                (fiber_mean_sim < global_cutoff) & valid_sim_mask
            )[0]
            reject_mask[global_reject] = True

        keep_mask = ~reject_mask

        removed_count = int(np.sum(~keep_mask))
        print(
            f"[init_cluster_centroid] Iteration {iteration + 1} removed {removed_count} outliers "
            f"({removed_count / max(len(keep_mask), 1) * 100:.2f}%)."
        )

        if np.sum(keep_mask) < n_clusters:
            print("[init_cluster_centroid] Remaining samples are fewer than n_clusters. Stop pruning.")
            break

        active_embeddings = active_embeddings[keep_mask]
        active_subjects = active_subjects[keep_mask]

    embedding_array = active_embeddings
    labels, _ = _run_cluster(embedding_array)

    print("Refining cluster centers using Geometric Median...")

    refined_centers = []
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique_labels, counts))

    avg_cluster_size = len(embedding_array) // n_clusters
    min_samples_threshold = max(10, int(0.01 * avg_cluster_size))

    for i in tqdm.tqdm(range(n_clusters), desc="Refining Centers"):
        mask = (labels == i)
        points = embedding_array[mask]

        if len(points) < min_samples_threshold:
            valid_large_clusters = {
                cid: cnt for cid, cnt in cluster_counts.items()
                if cnt >= 2 * min_samples_threshold
            }

            if len(valid_large_clusters) > 0:
                largest_cluster_id = max(valid_large_clusters, key=valid_large_clusters.get)
            else:
                largest_cluster_id = max(cluster_counts, key=cluster_counts.get)

            largest_mask = labels == largest_cluster_id
            largest_points = embedding_array[largest_mask]

            if len(largest_points) >= 2:
                sub_kmeans = KMeans(
                    n_clusters=2,
                    n_init=5,
                    max_iter=200,
                    random_state=42,
                )
                sub_labels = sub_kmeans.fit_predict(largest_points)
                sub_counts = np.bincount(sub_labels, minlength=2)

                # Give the smallest sub-cluster to the empty cluster and keep
                # the majority sub-cluster in the donor cluster.
                new_sub_id = int(np.argmin(sub_counts))
                largest_indices = np.nonzero(largest_mask)[0]
                labels[largest_indices[sub_labels == new_sub_id]] = i
                cluster_counts[i] = int(np.sum(sub_labels == new_sub_id))
                cluster_counts[largest_cluster_id] = int(np.max(sub_counts))

                points = embedding_array[labels == i]
                refined_center = utilis.geometric_median(points)

                print(
                    f"[Info] Cluster {i} is too small or empty ({len(points)}). "
                    f"Split cluster {largest_cluster_id}: "
                    f"{sub_counts[0]} / {sub_counts[1]}."
                )

            else:
                refined_center = embedding_array[np.random.randint(len(embedding_array))]

            refined_centers.append(refined_center)
            continue

        refined_center = utilis.geometric_median(points)
        refined_centers.append(refined_center)

    refined_centers = np.stack(refined_centers, axis=0)

    if np.isnan(refined_centers).any():
        print("[Warning] NaN detected in centroids! Replacing with random samples.")
        nan_rows = np.isnan(refined_centers).any(axis=1)
        refined_centers[nan_rows] = embedding_array[np.random.randint(len(embedding_array), size=nan_rows.sum())]

    print(f"Updating model weights with shape: {refined_centers.shape}")
    with torch.no_grad():
        weight_tensor = torch.from_numpy(refined_centers).to(device=device, dtype=model.cluster_head.weight.dtype)
        model.cluster_head.set_weight(weight_tensor)

    return model


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    # batch: (BN, C)
    weight = (batch ** 2) / torch.sum(batch, 0).clamp_min(utilis.EPS) # (BN, C)
    target = (weight.t() / torch.sum(weight, 1).clamp_min(utilis.EPS)).t() # (BN, C)

    return target


def cluster_assignment_loss_dec(probs, target=None):
    """
    Compute cluster assignment loss based on KL-divergence between predicted probabilities and target distribution.

    probs: (B, N, C)
    """
    C = probs.size(-1)
    probs = probs.reshape(-1, C).float().clamp_min(utilis.EPS)  # (BN, C)

    if target is None:
        target = target_distribution(probs).detach()  # (BN, C)
    else:
        target = target.to(probs.device).float()  # (BN, C)

    assign_loss = torch.nn.functional.kl_div(
        probs.log(), #  (BN, C)
        target,
        reduction='batchmean',
    )

    return assign_loss


def cluster_balance_loss(probs):
    """Penalize a batch-level cluster distribution that collapses to a few IDs."""
    num_clusters = probs.size(-1)
    probs = probs.reshape(-1, num_clusters).float().clamp_min(utilis.EPS)
    cluster_prob = probs.mean(dim=0, keepdim=True)
    cluster_prob = cluster_prob / cluster_prob.sum(
        dim=-1, keepdim=True
    ).clamp_min(utilis.EPS)
    uniform = torch.full_like(cluster_prob, 1.0 / num_clusters)

    return torch.nn.functional.kl_div(
        cluster_prob.clamp_min(utilis.EPS).log(),
        uniform,
        reduction="batchmean",
    )


def registration_loss(A_points, B_points, method="RMS"):
    """
    Compute registration loss between two sets of points.

    A_points: (B, S, P, D)
    B_points: (B, S, P, D)
    """
    D = utilis.calculate_distance(A_points, B_points, method=method)  # (B, Sa, Sb)

    min_dist_a2b = D.min(dim=2).values  # (B, Sa)
    min_dist_b2a = D.min(dim=1).values  # (B, Sb)
    loss = (min_dist_a2b.mean() + min_dist_b2a.mean()) / 2

    return loss


def reg_invariant_loss(A_points, B_points, aug_affine, scale=None, delta=0.1):
    """
    Equivariance loss for keypoint extraction: keypoints extracted from the
    transformed view must equal the known transform applied to keypoints
    extracted from the unaugmented view, K(T·x) = T·K(x).

    A_points: (B, P, D) keypoints of the augmented view (world coordinates).
    B_points: (B, P, D) keypoints of the unaugmented view.
    aug_affine: (B, 4, 4) forward affine augmentation T.
    scale: (B,) optional full-tractogram scale of the transformed view.  When
        supplied, residuals are dimensionless and comparable across subjects.
    delta: Huber threshold in normalized residual units.
    """
    transformed = AffineTransform(matrix=aug_affine).get_forward_transformed_points(
        B_points.float()
    )
    residual = A_points.float() - transformed
    if scale is not None:
        scale = torch.as_tensor(scale, device=residual.device, dtype=residual.dtype)
        residual = residual / scale.reshape(-1, 1, 1).clamp_min(utilis.EPS)

    return torch.nn.functional.huber_loss(
        residual,
        torch.zeros_like(residual),
        delta=delta,
    )


def embedding_consistency_loss(reference_embeddings, augmented_embeddings):
    """Make embeddings of the same streamline invariant to augmentation."""
    reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=-1)
    augmented_embeddings = torch.nn.functional.normalize(augmented_embeddings, dim=-1)
    return (1.0 - (reference_embeddings * augmented_embeddings).sum(dim=-1)).mean()


def emb_metric_loss(embeddings, coords, method="RMS", delta=1.0, chunk_rows=1024):
    """Match embedding and streamline distances within each batch item.

    ``coords`` must be raw coordinates in a common world frame (millimetres).
    The target therefore preserves both relative streamline geometry and
    cross-subject absolute position. DGCNN feature normalization is separate.

    Pairwise distances are evaluated in row blocks over columns ``j >= i``
    only, so the mirrored half of each symmetric matrix is never computed.
    The block Huber sums are accumulated under a precomputed 0/1 mask, which
    keeps the whole path free of boolean indexing and its GPU-CPU
    synchronizations. Results match the full-matrix triu formulation up to
    floating-point reassociation.
    """
    num_streamlines = embeddings.shape[1]
    if num_streamlines < 2:
        return embeddings.sum() * 0.0

    batch = embeddings.shape[0]
    upper_cache = {}
    total = None
    for start in range(0, num_streamlines - 1, chunk_rows):
        end = min(start + chunk_rows, num_streamlines - 1)
        rows, cols = end - start, num_streamlines - start
        # The mask content only depends on the block shape, not on the
        # absolute row offset: both ranges shift by the same ``start``.
        key = (rows, cols)
        if key not in upper_cache:
            r = torch.arange(start, end, device=coords.device)
            c = torch.arange(start, num_streamlines, device=coords.device)
            upper_cache[key] = (c.view(1, -1) > r.view(-1, 1)).to(
                device=coords.device, dtype=embeddings.dtype
            )
        mask = upper_cache[key]

        with torch.no_grad():
            geometric_block = utilis.calculate_distance(
                coords[:, start:end], coords[:, start:], method=method
            )  # (B, rows, cols), orientation-invariant

        embedding_block = torch.cdist(
            embeddings[:, start:end], embeddings[:, start:], p=2
        )
        if method == "MeanSquared":
            embedding_block = embedding_block.square()
        block_huber = torch.nn.functional.huber_loss(
            embedding_block, geometric_block, reduction="none", delta=delta
        )
        block_sum = (block_huber * mask).sum()
        total = block_sum if total is None else total + block_sum

    # reduction="mean" divides by batch items AND pairs.
    return total / ((num_streamlines * (num_streamlines - 1) // 2) * batch)


def _normalize_keypoint_coordinates(points, center, scale):
    """Normalize keypoints in a complete-tractogram coordinate frame."""
    center = torch.as_tensor(center, device=points.device, dtype=points.dtype)
    if center.ndim == 1:
        center = center.unsqueeze(0)
    scale = torch.as_tensor(scale, device=points.device, dtype=points.dtype)
    return (points - center[:, None]) / scale.reshape(-1, 1, 1).clamp_min(utilis.EPS)


def keypoint_repulsion_loss(keypoints, center, scale):
    """Penalize coincident learned keypoints without prescribing their layout.

    Pairwise distances are measured after subject-level normalization.  The
    potential ``log(1 + 1 / d²)`` rises sharply for duplicate keypoints and
    decays smoothly for separated ones, so it discourages collapse without a
    hand-chosen minimum physical separation.
    """
    keypoints_norm = _normalize_keypoint_coordinates(keypoints.float(), center, scale)
    if keypoints_norm.shape[1] < 2:
        return keypoints_norm.sum() * 0.0
    pairwise = torch.cdist(keypoints_norm, keypoints_norm, p=2)
    off_diagonal = ~torch.eye(
        keypoints_norm.shape[1], device=keypoints_norm.device, dtype=torch.bool
    )
    repulsion = torch.log1p(
        torch.reciprocal(pairwise[:, off_diagonal].square() + utilis.EPS)
    ).mean()
    return repulsion
