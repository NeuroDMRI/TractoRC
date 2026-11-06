# TractoRC

TractoRC learns a shared tractography representation for keypoint-based registration and streamline clustering. Training reads a directory of subject tractograms, samples and resamples streamlines, and preserves their supplied world-coordinate positions (`recenter: false` in the example configs). It does not require a pre-built atlas.

## Installation

Create and activate the Python 3.10 environment, then install the project dependencies:

```bash
git clone https://github.com/yishengpoxiao/TractoRC.git
cd TractoRC
conda create -n TractoRC python=3.10
conda activate TractoRC
pip install -r requirements.txt
```

For CUDA training, install a PyTorch build compatible with the installed CUDA drivers if the requirements install does not provide one. The example training settings use AMP and large samples and can require substantial GPU memory, CPU RAM, and training time.

## Data layout

Place one tractogram per subject under `data/train/` (nested directories are searched recursively). Training and directory-based workflows discover `.trk`, `.tck`, `.vtk`, `.vtp`, `.fib`, `.dpy`, and `.tt.gz` files recursively. Single-file inference also accepts uncompressed `.tt`. Provide coordinates in RAS+ millimetres in a shared physical/world frame across subjects; DSI Studio TinyTrack coordinates are transformed using the file's `trans_to_mni` affine. Tract lengths are interpreted in millimetres. Keep training and inference coordinates consistent.

## Pretrained checkpoint

Download [the pretrained release](https://github.com/yishengpoxiao/TractoRC/releases/tag/pretrained-2026-11-06) and place these assets in `checkpoints/`:

- `TractoRC_pretrained_k256_c800.pth` — the trained registration and clustering model (26,209,918 bytes).
- `TractoRC_pretrained_k256_c800.pth.sha256` — its SHA256 checksum.

With the [GitHub CLI](https://cli.github.com/) installed and authenticated where required:

```bash
mkdir -p checkpoints
gh release download pretrained-2026-11-06 \
  --repo yishengpoxiao/TractoRC \
  --pattern 'TractoRC_pretrained_k256_c800.pth*' --dir checkpoints
(cd checkpoints && sha256sum -c TractoRC_pretrained_k256_c800.pth.sha256)
```

The checkpoint uses architecture version 3, 256 keypoints, 800 clusters, 1,024-dimensional features, and a coordinate-conditioned keypoint head. It was trained with 14 points per streamline and 3,200 sampled streamlines per subject. The release preserves the original checkpoint bytes, including its training metadata and optimizer state; only the filename changes. Its training run is separate from the example YAML schedules below.

The inference examples use **MAD z = 0.80** for clustering and **TPS regularization = 0.05** for registration. These are inference settings passed on the command line; downloading the checkpoint does not set script arguments. Keep `--nb_points 14` with this checkpoint, including for atlas creation, whose script default is 30.

Current architecture metadata and a compatible state dictionary are required when loading a checkpoint. The YAML files configure training; they do not contain model weights.

## Script reference

Run all commands from the repository root. Relative paths in commands and YAML files are resolved from the working directory. Each executable script provides `--help`.

| Script | Function | Main output |
| --- | --- | --- |
| [`scripts/train.py`](scripts/train.py) | Pretrain or jointly train the registration and clustering model from YAML. | Model checkpoint and, with validation, a best checkpoint. |
| [`scripts/infer_registration.py`](scripts/infer_registration.py) | Register one moving tractogram to one fixed tractogram. | Aligned tractogram and fitted transform. |
| [`scripts/infer_clustering.py`](scripts/infer_clustering.py) | Assign and filter clusters in one tractogram. | Per-cluster VTP files and rejected streamlines. |
| [`scripts/group_registration.py`](scripts/group_registration.py) | Register multiple subjects to an iteratively estimated mean-keypoint template. | Mean keypoints, aligned subjects, and transforms. |
| [`scripts/re_clustering.py`](scripts/re_clustering.py) | Re-estimate clustering centroids from a dataset using a trained representation. | A new model checkpoint with updated centroids. |
| [`scripts/atlas_creation.py`](scripts/atlas_creation.py) | Pool model-defined clusters across already aligned subjects. | Atlas clusters and rejected streamlines as VTP files. |
| [`scripts/__init__.py`](scripts/__init__.py) | Mark `scripts` as a Python package; contains only a package docstring. | No command-line entrypoint. |

### `train.py`: pretraining and joint training

The [pretraining example](configs/pretrain.yaml) and [formal-training example](configs/train.yaml) explain every parameter in English comments, including all nested loss weights. Comments describe purpose, units, choices, and interactions with checkpoints and training phases.

Copy the example configs and set your dataset paths and training settings in your own copies, then pass those paths with `--config`. The commands below show the shipped example filenames. Training writes checkpoints to `outputs/pretrain/pretrain.pth` and `outputs/train/TractoRC.pth`.

```bash
python scripts/train.py --config configs/pretrain.yaml
python scripts/train.py --config configs/train.yaml
```

The formal-training config loads the pretraining checkpoint at `outputs/pretrain/pretrain.pth`, so run pretraining first. For distributed training, launch with `torchrun`; `batch_size` in YAML is the global batch across all processes and must divide evenly by the process count. These example values use global batches 4 and 20 respectively, matching four processes (per-process batches 1 and 5):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/pretrain.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/train.yaml
```

DDP training requires CUDA. The YAML `gpu` value selects the device for a single process; `torchrun` selects each worker's local device from `LOCAL_RANK`. Formal DDP training uses `pairing: all`; chain pairing is single-GPU only.

The examples use 500 pretraining epochs, followed by 490 registration/embedding warm-up epochs and 10 joint-training epochs. K-means initializes the 800 centroids at formal epoch 490 (zero-based). If shortening a run, also adjust `precluster_epochs` and `freeze_clustering_after_epoch` so that `precluster_epochs < epochs <= freeze_clustering_after_epoch`. Provide at least four subjects for the four-process examples, and at least `num_samples` streamlines per subject after length filtering to keep batches equally sized. Centroid initialization needs at least `num_clusters` embeddings after subsampling and outlier filtering.

`--config` is required and selects the YAML file. `--local-rank` / `--local_rank` is reserved for distributed launchers. In YAML, set `train_path` and `output_path`; `val_path` is optional. `pretrain: true` selects pretraining, while `pretrain: false` selects formal training. `use_checkpoint` loads an existing model; `resume_training: false` starts a new optimization run from those weights, while `true` also restores the saved training state and epoch. Checkpoint architecture fields take precedence over corresponding YAML fields when weights are loaded.

The primary checkpoint is written to `output_path/model_name` and updated during training. With validation, the best model is also saved with a `_best.pth` suffix. Formal training saves a `_kmeans_init.pth` snapshot when centroids are initialized; pretraining can optionally produce a `_with_clusters.pth` checkpoint. Keep generated checkpoints outside the source files you commit.

### `infer_registration.py`: pairwise registration

Fit a moving-to-fixed transform from sampled keypoints, then apply it to the retained moving streamlines. The example first fits a coarse affine, re-extracts the moving keypoints, and fits a residual TPS with regularization **0.05**:

```bash
python scripts/infer_registration.py \
  --checkpoint checkpoints/TractoRC_pretrained_k256_c800.pth \
  --moving data/test/subject-02.trk --fixed data/test/subject-01.trk \
  --output outputs/registration/subject-02_aligned.vtp \
  --registration_pipeline affine_refine_tps --refine_feature_stats fixed \
  --regis_type tps_0.05 --nb_points 14 --num_samples 3200 \
  --seed 42 --device 0
```

Required arguments are `--checkpoint`, `--moving`, `--fixed`, and `--output`.

| Option | Purpose |
| --- | --- |
| `--registration_pipeline` | `affine_refine_tps` performs coarse affine alignment and moving-keypoint re-extraction; `single_tps` fits the requested stages on the original keypoints. |
| `--refine_feature_stats` | `fixed` re-encodes coarse-aligned moving fibers with fixed-subject statistics; `per_side` uses the original per-subject statistics. |
| `--regis_type tps_0.05` | Set the TPS stage explicitly. Ordered stages can be space- or comma-separated; in the coarse-to-fine pipeline these are the residual stages. |
| `--nb_points`, `--num_samples` | Points per streamline and sampled streamlines for keypoint extraction. When `--num_samples` is left at the script default, the checkpoint's training value is preferred. |
| `--min_length`, `--max_length` | Retain streamlines in the specified length range, in millimetres; defaults are 20 and 260. |
| `--transform_output` | Fitted transform path; defaults to `<output_stem>_transform.pt` next to `--output`. |
| `--output_keypoints` | Also write moving, fixed, aligned, and, where applicable, refined keypoints as VTP files. |
| `--apply_dir`, `--apply_output_dir` | Optional paired arguments: apply the same transform to anatomical `.vtk`/`.vtp` files recursively, preserving their relative paths and original point counts. |
| `--apply_batch_streamlines` | Complete streamlines transformed per batch; default 10,000. |
| `--seed` | Seed streamline sampling for repeatable runs. |

The main output contains the moving streamlines after length filtering and resampling. The optional `--apply_dir` path instead transforms the supplied anatomical tracts without that filtering or resampling. The script warns when the selected pipeline or feature statistics differ from the checkpoint's training settings; it does not automatically change those arguments.

### `infer_clustering.py`: cluster one tractogram

Assign each resampled streamline to a model cluster, then reject low-confidence streamlines within each cluster using **MAD z = 0.80**:

```bash
python scripts/infer_clustering.py \
  --checkpoint checkpoints/TractoRC_pretrained_k256_c800.pth \
  --input data/test/subject-01.trk --output outputs/clustering/subject-01 \
  --nb_points 14 --batchsize 5000 \
  --outlier-mode mad --mad-z 0.80 --min-cluster-size 20 --min-keep 5 \
  --device 0
```

Required arguments are `--checkpoint`, `--input`, and `--output`. `--nb_points` defaults to 14 and `--batchsize` to 50,000; the example uses a smaller batch to reduce memory demand. This script resamples the input but does not perform length filtering or registration.

Outputs use one-based cluster filenames: `clusters/cluster_00001.vtp`, etc. Rejected streamlines are saved separately as `clusters/outliers/cluster_00001_outliers.vtp`, etc. There are 800 cluster files with the released checkpoint, including files for empty clusters. Coordinates remain in the input frame.

The shared filtering options for this script and `atlas_creation.py` are:

| Option | Meaning |
| --- | --- |
| `--outlier-mode mad` | Use the per-cluster lower confidence tail. Other supported modes are `none`, `absolute`, and `hybrid`. |
| `--mad-z 0.80` | Reject confidence below `median - 0.80 * 1.4826 * MAD`, where `MAD = median(abs(confidence - median))`. Confidence is the maximum predicted cluster probability. This is a dimensionless confidence threshold, not a geometric distance. |
| `--min-cluster-size 20` | Apply MAD filtering only when at least 20 valid, non-rejected streamlines are available. A zero MAD skips the MAD step. |
| `--min-keep 5` | Retain at least five streamlines when applying additional MAD rejections, if the cluster initially has that many. |
| `--min-confidence` | Required for `absolute` or `hybrid`; rejects confidence below the supplied value in `[0, 1]`. These explicit rejections are not restored by `--min-keep`. |

### `group_registration.py`: register a group

Estimate a mean-keypoint reference from at least two subjects and iteratively align the subjects to it. The template is estimated from the input group; an external atlas is not required.

```bash
python scripts/group_registration.py \
  --checkpoint checkpoints/TractoRC_pretrained_k256_c800.pth \
  --input_dir data/test --output_dir outputs/group_registration \
  --regis_type affine tps_0.05 --nb_points 14 --num_samples 3200 \
  --num_iters 5 --seed 42 --device 0
```

Required arguments are `--checkpoint`, `--input_dir`, and `--output_dir`. Files are discovered recursively. `--regis_type` sets ordered transform stages; the example uses affine followed by TPS regularization **0.05**. Unlike pairwise `affine_refine_tps`, this script extracts keypoints once per subject and iterates their alignment to the group mean.

`--nb_points` defaults to 14, `--num_iters` to 5, and `--min_length` / `--max_length` to 20 / 260 mm. Leaving `--num_samples` at its default prefers the checkpoint's training value. `--transform_batch_streamlines` controls transform application batches (default 10,000), and `--seed` controls streamline sampling.

The output contains `mean_keypoints.vtp`, `<subject>_aligned.vtp`, and `<subject>_transform.pt`. Use distinct subject filenames to avoid output-name collisions. By default, only the final results are written directly under `--output_dir`. With `--save_each_iter`, intermediate results go into `iter_01/`, etc., and the final iteration goes into `final/`. Aligned tractograms contain the length-filtered, resampled streamlines.

### `re_clustering.py`: re-estimate model centroids

Extract embeddings from a dataset, initialize and refine cluster centroids, and save a new checkpoint. This is useful when deliberately building a new clustering model; it changes cluster assignments and identities and is not needed to use the released checkpoint.

```bash
python scripts/re_clustering.py \
  --checkpoint checkpoints/TractoRC_pretrained_k256_c800.pth \
  --inputdir data/train --output_path outputs/reclustering/TractoRC_reclustered.pth \
  --clusters 800 --nb_points 14 --batchsize 5000 \
  --clustering_fiber_interval 4 --cluster_kmeans_n_init 10 \
  --init kmeans --device 0
```

Required arguments are `--checkpoint`, `--inputdir`, and `--output_path`.

| Option | Purpose / default |
| --- | --- |
| `--clusters` | Number of new centroids; default 800. Can differ from the input checkpoint's cluster count. |
| `--nb_points` | Resampling points per streamline; default 14. |
| `--batchsize` | Embedding extraction batch size; default 50,000. |
| `--clustering_fiber_interval` | Use every Nth extracted embedding for initialization; default 4. |
| `--cluster_kmeans_n_init` | K-means restarts; default 10. |
| `--init` | `kmeans` (default) or `gmm`; the GMM path uses its own initialization settings. |

Input tractograms are length-filtered to 20–260 mm. Initialization includes global outlier filtering and centroid refinement, so it may take substantial CPU time after GPU embedding extraction. At least `--clusters` embeddings must remain after subsampling and filtering. The output stores model weights and architecture metadata, without optimizer state or the original training configuration; it is intended for inference or a new training run, not exact training resumption. This script does not export tractograms or register subjects.

### `atlas_creation.py`: pool clusters into an atlas

Cluster already aligned subjects and pool their streamlines by cluster ID. First register the subjects into a common space, for example using `group_registration.py`.

```bash
mkdir -p data/aligned
cp outputs/group_registration/*_aligned.vtp data/aligned/
python scripts/atlas_creation.py \
  --checkpoint checkpoints/TractoRC_pretrained_k256_c800.pth \
  --inputdir data/aligned --outdir outputs/atlas \
  --nb_points 14 --batchsize 5000 \
  --outlier-mode mad --mad-z 0.80 --min-cluster-size 20 --min-keep 5 \
  --device 0
```

Required arguments are `--checkpoint`, `--inputdir`, and `--outdir`. `--batchsize` defaults to 30,000. Set `--nb_points 14` explicitly for the released checkpoint, because this script defaults to 30 points. Input tractograms are length-filtered to 20–260 mm and resampled. The filtering options are the same as for single-subject clustering above, but the MAD statistics are calculated on the pooled streamlines within each atlas cluster.

Outputs are `clusters/cluster_XXXXX.vtp` and `clusters/outliers/cluster_XXXXX_outliers.vtp` under `--outdir`. This script preserves the supplied input positions; it does not estimate registration transforms or write a voxel label map. Copy only the aligned tractograms into the atlas input directory: group-registration output also contains a `mean_keypoints.vtp` point cloud, which is not a tractogram. If group registration used `--save_each_iter`, take the `*_aligned.vtp` files from its `final/` directory.

### Shared device and memory options

The inference, group-registration, reclustering, and atlas scripts accept `--device auto`, `cpu`, a CUDA index such as `0`, or a device string such as `cuda:0`. `auto` selects CUDA when available and otherwise CPU. Single-subject clustering defaults to GPU `0`; the other scripts default to `auto`. Use `--device cpu` explicitly when CUDA is unavailable. GPU inference uses AMP. Reduce `--batchsize` for clustering, atlas creation, or reclustering if memory is limited; this controls model inference batches rather than a CPU process count. Registration transform-application batches have their own options described above.

## Code layout

`tractorc/` contains the model, data loading, training losses, alignment, and I/O helpers. `scripts/` provides the command-line entrypoints, and `configs/` holds the pretraining and training examples. Keep downloaded weights, datasets, and generated results out of source commits; pretrained weights are distributed as Release assets.
