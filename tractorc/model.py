import torch
import torch.nn as nn

import math
from . import keypoint_aligners
from . import module
from . import utilis


# "single_tps" fits one keypoint transform per forward pass; "affine_refine_tps"
# fits an affine on the shared keypoints, re-extracts moving features and
# keypoints from the coarse-aligned fibers, then fits a residual TPS.
REGISTRATION_PIPELINES = ("single_tps", "affine_refine_tps")
# Statistics used to re-encode coarse-aligned moving fibers: "per_side" keeps
# each side's original complete-tractogram statistics, "fixed" re-encodes the
# moving side with the fixed-side statistics.
REFINE_FEATURE_STATS = ("per_side", "fixed")


class TractoRC(nn.Module):
    MODEL_TYPE = "TractoRC"

    def __init__(
        self,
        num_keypoints,
        num_clusters,
        dim=3,
        feat_len=1024,
        use_amp=False,
        cluster_temperature=1,
        keypoint_temperature=1,
        keypoint_head_append_coords=False,
        ):
        super().__init__()
        if num_keypoints < 1 or num_clusters < 1:
            raise ValueError("num_keypoints and num_clusters must both be positive")
        if cluster_temperature <= 0 or keypoint_temperature <= 0:
            raise ValueError("cluster_temperature and keypoint_temperature must be positive")
        self.num_keypoints = num_keypoints
        self.num_clusters = num_clusters
        self.feat_len = feat_len
        self.dim = dim
        self.cluster_temperature = cluster_temperature
        self.keypoint_temperature = keypoint_temperature
        self.append_coords = keypoint_head_append_coords

        self.use_amp = use_amp

        self.backbone = module.DGCNN(features_len=self.feat_len)
        # Optionally set to a torch.compile wrapper of ``backbone`` (see
        # ``enable_compiled_backbone``). Stored outside the module tree so
        # checkpoints keep plain parameter names.
        self._compiled_backbone = None

        self.cluster_head = module.ClusteringLayer(
            self.feat_len,
            self.num_clusters,
            temperature=self.cluster_temperature,
        )
        self.keypoint_head = module.KeypointLayer(
            out_dim=self.num_keypoints,
            indim=self.feat_len,
            temperature=self.keypoint_temperature,
            append_coords=self.append_coords,
        )

    def enable_compiled_backbone(self):
        """Best-effort torch.compile of the backbone for training speed.

        The compiled wrapper shares the backbone's parameters but is kept
        outside the module tree (state_dict, train/eval, and inference paths
        are unchanged). Dynamo errors fall back to eager execution instead of
        failing the run.
        """
        try:
            compiled = torch.compile(self.backbone)
            # Any graph the compiler cannot handle runs eagerly with a
            # warning rather than aborting training.
            from torch._dynamo import config as dynamo_config

            dynamo_config.suppress_errors = True
        except Exception as error:  # pragma: no cover - environment dependent
            print(f"[Compile] torch.compile unavailable, running eager: {error}")
            return
        object.__setattr__(self, "_compiled_backbone", compiled)

    def extract_features(self, points, idx=None):
        """Extract DGCNN features from already normalized coordinates."""
        # The compiled graph is only used for training-mode batches: switching
        # it between eval (validation under inference_mode) and train mode has
        # produced a faulty re-specialized kernel at the first train step after
        # validation. Eval/inference passes run the eager backbone.
        use_compiled = self._compiled_backbone is not None and self.training
        backbone = self._compiled_backbone if use_compiled else self.backbone
        return backbone(points, idx_seq=idx)

    def _autocast(self, device):
        cast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        return torch.autocast(device_type=device.type, enabled=self.use_amp, dtype=cast_dtype)

    @staticmethod
    def _normalize_feature_coords(points, feature_stats):
        center, scale = feature_stats
        return utilis.norm_coords(points, center=center, scale=scale)

    def encode_keypoints(self, points, feature_stats, idx=None, use_weights=False):
        """Encode features and extract global keypoints in a single pass.

        ``points`` stays in registration (world) coordinates; the backbone
        encodes their complete-tractogram normalized copy. Returns
        ``(keypoints, features, normalized_coords)``.
        """
        points_norm = self._normalize_feature_coords(points, feature_stats)
        with self._autocast(points.device):
            features = self.extract_features(points_norm, idx=idx)
            keypoints = self.get_keypoints_and_confidence(
                points,
                features,
                feature_coords=points_norm,
                return_confidence=use_weights,
            )
        return keypoints, features, points_norm

    def get_keypoints_and_confidence(
        self, points, feat, feature_coords=None, return_confidence=True
    ):
        # feat: (B*N, feat_len, P)
        # points: (B, N, P, D) in registration/world coordinates.
        B, N, P, _ = points.shape
        point_count = N * P

        points_flat = points.reshape(B, point_count, self.dim) # (B,NP,D)

        feat = (
            feat.reshape(B, N, self.feat_len, P)   # (B, N, F, P)
                .transpose(1, 2)                  # (B, F, N, P)
                .reshape(B, self.feat_len, N * P) # (B, F, NP)
        )

        if self.append_coords:
            # feature_coords are the normalized coordinates the backbone saw,
            # so the keypoint head conditions on the same frame.
            points_flat_norm = feature_coords.reshape(B, point_count, self.dim)
            feat = torch.cat([feat, points_flat_norm.transpose(1, 2)], dim=1)

        # KeypointLayer owns the temperature-scaled softmax. Passing P keeps
        # its local convolution confined to each streamline after flattening.
        w = self.keypoint_head(feat, num_per_line=P)  # (B, C, NP)

        keypoints = torch.bmm(w, points_flat)  # (B,C,D)

        if return_confidence:
            entropy = -(w * (w.clamp_min(utilis.EPS)).log()).sum(dim=2) # (B, C)
            if point_count == 1:
                confidence = torch.ones_like(entropy)
            else:
                confidence = 1.0 - entropy / math.log(point_count)  # (B, C)

            return keypoints, confidence.clamp(min=0.0, max=1.0)
        else:
            return keypoints

    def get_streamline_label(self, feat, num_lines, return_emb=False):
        """
        Get streamline labels (and optional embeddings) from feature maps.

        Args:
            feat (torch.Tensor): Feature maps of shape (B*N, feat_len, P)
            num_lines (int): Number of streamlines per batch.

        Returns:
            torch.Tensor: Streamline labels of shape (B, N)
        """
        cluster_out = self.cluster_head(feat, return_emb=return_emb)
        probs, emb = cluster_out if return_emb else (cluster_out, None)
        probs = probs.reshape(-1, num_lines, self.num_clusters)  # (B, N, C)
        labels = probs.argmax(dim=-1)  # (B, N)
        if return_emb:
            return labels, probs, emb
        return labels, probs

    def _fit_keypoint_aligner(
        self, points_m, points_f, weights, transform_type, pair_center, pair_scale
    ):
        """Fit one keypoint aligner in the pair's shared coordinate frame."""
        return keypoint_aligners.build_keypoint_aligner(
            points_m=utilis.normalize_pairwise_points(points_m.float(), pair_center, pair_scale),
            points_f=utilis.normalize_pairwise_points(points_f.float(), pair_center, pair_scale),
            weights=weights,
            transform_type=transform_type,
        )

    def _apply_keypoint_aligner(self, aligner, points, pair_center, pair_scale):
        """Apply a fitted aligner to world-coordinate points, keeping gradients."""
        aligned_norm = aligner.get_forward_transformed_points(
            utilis.normalize_pairwise_points(points.float(), pair_center, pair_scale)
        )
        return utilis.denormalize_pairwise_points(aligned_norm, pair_center, pair_scale)

    def forward(
        self,
        fixed_coord,
        moving_coord,
        transform_type="affine",
        clustering=False,
        registration=True,
        return_emb=False,
        use_weights=False,
        idx=None,
        fixed_feature_stats=None,
        moving_feature_stats=None,
        registration_pipeline="single_tps",
        refine_feature_stats="per_side",
        return_pretraining_outputs=False,
        return_pretraining_embeddings=True,
    ):
        """
        Forward pass for one mini-batch step.

        :param moving_coord: Coordinates of the moving tractography
        :param fixed_coord: Coordinates of the fixed tractography
        :param transform_type: str of keypoint alignment type. In the
            affine_refine_tps pipeline it specifies the fine (residual) stage;
            the coarse stage is always affine.

        :return res: Dictionary of results
        """
        batch_size, num_lines, num_points, _ = moving_coord.shape
        device = moving_coord.device

        with self._autocast(device):
            output_f, feat_f, fixed_norm = self.encode_keypoints(
                fixed_coord, fixed_feature_stats, idx=idx, use_weights=use_weights
            )
            output_m, feat_m, moving_norm = self.encode_keypoints(
                moving_coord, moving_feature_stats, idx=idx, use_weights=use_weights
            )

            if return_pretraining_outputs and return_pretraining_embeddings:
                emb_f = self.cluster_head._extract_embedding(feat_f)
                emb_m = self.cluster_head._extract_embedding(feat_m)

            if clustering:
                clu_f = self.get_streamline_label(feat_f, num_lines, return_emb=return_emb) # (B, N)
                clu_m = self.get_streamline_label(feat_m, num_lines, return_emb=return_emb) # (B, N)

                if return_emb:
                    label_f, prob_f, emb_f = clu_f
                    label_m, prob_m, emb_m = clu_m
                else:
                    label_f, prob_f = clu_f
                    label_m, prob_m = clu_m

            if use_weights:
                points_f, confidence_f = output_f
                points_m, confidence_m = output_m

                weights = confidence_f * confidence_m  # (B, cluster_num)
                weights = weights / (weights.sum(dim=1, keepdim=True) + utilis.EPS)
            else:
                points_f = output_f
                points_m = output_m

                weights = None

        if registration:
            moving_center, moving_scale = moving_feature_stats
            fixed_center, fixed_scale = fixed_feature_stats
            pair_center, pair_scale = utilis.shared_tps_normalization_stats(
                moving_center,
                moving_scale,
                fixed_center,
                fixed_scale,
                device=device,
                dtype=torch.float32,
            )

            moving_flat = moving_coord.reshape(batch_size, num_lines * num_points, self.dim)
            # The pipelines share one backbone and one keypoint head; stages
            # differ only in the transform fitted on the keypoints.
            coarse_type = (
                "affine" if registration_pipeline == "affine_refine_tps" else transform_type
            )
            coarse_aligner = self._fit_keypoint_aligner(
                points_m, points_f, weights, coarse_type, pair_center, pair_scale
            )
            points_aligned = self._apply_keypoint_aligner(
                coarse_aligner, moving_flat, pair_center, pair_scale
            )

            if registration_pipeline == "affine_refine_tps":
                points_aligned_coarse = points_aligned
                # Re-encode the coarse-aligned moving fibers so the fine stage
                # sees features at their target location. Keypoint confidence
                # weights apply to the coarse affine fit; the residual stage
                # is unweighted.
                refine_stats = (
                    fixed_feature_stats if refine_feature_stats == "fixed" else moving_feature_stats
                )
                points_m_refined, _, _ = self.encode_keypoints(
                    points_aligned_coarse.reshape(batch_size, num_lines, num_points, self.dim),
                    refine_stats,
                    idx=idx,
                )
                fine_aligner = self._fit_keypoint_aligner(
                    points_m_refined, points_f, None, transform_type, pair_center, pair_scale
                )
                points_aligned = self._apply_keypoint_aligner(
                    fine_aligner, points_aligned_coarse, pair_center, pair_scale
                )

        res = {}
        if return_pretraining_outputs:
            if use_weights:
                raise ValueError("Pretraining outputs do not support keypoint weights.")
            res["points_f"] = output_f
            res["points_m"] = output_m
            if return_pretraining_embeddings:
                res["emb_f"] = emb_f.reshape(batch_size, num_lines, -1)
                res["emb_m"] = emb_m.reshape(batch_size, num_lines, -1)
        if clustering:
            res['label_f'] = label_f
            res['label_m'] = label_m
            res['prob_f'] = prob_f
            res['prob_m'] = prob_m
            if return_emb:
                res['emb_f'] = emb_f.reshape(batch_size, num_lines, -1)
                res['emb_m'] = emb_m.reshape(batch_size, num_lines, -1)
        if registration:
            res['points_aligned'] = points_aligned.reshape(
                batch_size, num_lines, num_points, self.dim
            )
            res['points_m'] = points_m
            res['points_f'] = points_f
            if registration_pipeline == "affine_refine_tps":
                res['points_aligned_coarse'] = points_aligned_coarse.reshape(
                    batch_size, num_lines, num_points, self.dim
                )
                res['points_m_refined'] = points_m_refined

        return res
