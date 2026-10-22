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

## Train

The [pretraining example](configs/pretrain.yaml) and [formal-training example](configs/train.yaml) explain every parameter in English comments, including all nested loss weights. Comments describe purpose, units, choices, and interactions with checkpoints and training phases.

Run commands from the repository root. Edit the example configs to set your dataset paths and training settings. Training writes checkpoints to `outputs/pretrain/pretrain.pth` and `outputs/train/TractoRC.pth`.

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

## Inference

Registration fits an affine coarse alignment, re-extracts moving keypoints, then fits a residual TPS by default. Specify a single moving and fixed tractogram:

```bash
python scripts/infer_registration.py \
  --checkpoint outputs/train/TractoRC.pth \
  --moving data/train/subject-02.trk --fixed data/train/subject-01.trk \
  --output outputs/registration/subject-02_aligned.vtp \
  --registration_pipeline affine_refine_tps --refine_feature_stats fixed \
  --regis_type tps_0.05 --device 0
```

Cluster one tractogram and write per-cluster tract files:

```bash
python scripts/infer_clustering.py \
  --checkpoint outputs/train/TractoRC.pth --input data/train/subject-01.trk \
  --output outputs/clustering/subject-01 --device 0
```

Optional group-wise registration and reclustering commands:

```bash
python scripts/group_registration.py --checkpoint outputs/train/TractoRC.pth \
  --input_dir data/train --output_dir outputs/group_registration --device 0
python scripts/re_clustering.py --checkpoint outputs/train/TractoRC.pth \
  --inputdir data/train --output_path outputs/train/TractoRC.reclustered.pth --device 0
```

Atlas creation pools clustered streamlines across subjects; it does not register them. First place the registered tractograms in `data/aligned/`, then run:

```bash
python scripts/atlas_creation.py --checkpoint outputs/train/TractoRC.pth \
  --inputdir data/aligned --outdir outputs/atlas --nb_points 14 --device 0
```

The inference scripts accept `--device auto`, `cpu`, a CUDA index such as `0`, or a device string such as `cuda:0`. Check `--help` for sampling, registration, and outlier-filter options.

## Checkpoints and weights

Model checkpoints must include the current TractoRC architecture metadata and compatible model state dictionary. Older or otherwise incompatible checkpoints are rejected; retrain from the affine pretraining stage with this release. The repository does not distribute pretrained weights. The YAML files are examples, not pretrained models.

## Code layout and checks

`tractorc/` contains the model, data loading, training losses, alignment, and I/O helpers. `scripts/` provides the command-line entrypoints, and `configs/` holds the pretraining and training examples. Data, weights, and generated outputs are ignored by Git.

Run the CPU/Gloo distributed and checkpoint tests with:

```bash
python tests/test_distributed_training.py
```
