import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utilis


class KeypointLayer(nn.Module):
    def __init__(self, out_dim, indim, temperature=0.6, append_coords=True):
        super(KeypointLayer, self).__init__()
        self.append_coords = bool(append_coords)
        self.num_keypoints = int(out_dim)
        self.feat = 64
        self.indim = indim + 3 if self.append_coords else indim
        self.temperature = float(temperature)

        self.conv = nn.Sequential(
            nn.Conv1d(self.indim, 256, 1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(256, 128, 1),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv1d(128, self.feat, 1),
            nn.InstanceNorm1d(self.feat, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.interior_conv1 = nn.Conv2d(
            self.feat,
            self.feat,
            (1, 3),
            padding=(0, 1),
        )
        self.keypoints_head = nn.Conv2d(
            self.feat,
            self.num_keypoints,
            (1, 3),
            padding=(0, 1),
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, num_per_line: int):
        # x: (B, F, N)
        weight = self.conv(x)  # (B, feat, N)
        num_lines = x.shape[-1] // num_per_line
        weight_line = weight.reshape(x.shape[0], self.feat, num_lines, num_per_line)
        feat = self.leaky_relu(weight_line + self.interior_conv1(weight_line))

        logits = self.keypoints_head(feat).reshape(
            x.shape[0], self.num_keypoints, -1
        )  # (B, num_keypoints, N)
        weight_interior = F.softmax(logits / self.temperature, dim=-1)

        return weight_interior


class ClusteringLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        emb_dim=10,
        temperature=1,
    ):
        super(ClusteringLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.emb_dim = emb_dim
        self.temperature = temperature
        self.linear1 = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear3 = nn.Linear(256, emb_dim, bias=True)

        self.weight = nn.Parameter(torch.Tensor(out_features, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def _extract_embedding(self, x):
        # x: (B, F, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1) # (B, F)
        x2 = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1) # (B, F)

        x = torch.cat((x1, x2), dim=1) # (B, 2F)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)  # (B, emb_dim)
        return x

    def forward(self, x, return_emb=False):
        emb = self._extract_embedding(x)  # (B, emb_dim)

        norm_squared = ((emb.unsqueeze(1) - self.weight) ** 2).sum(dim=2)  # (B, out_features)
        # Scale-invariant student-t kernel. The embedding metric loss
        # constrains relative geometry but does not fix a global embedding
        # norm. Normalizing by each sample's own nearest-centroid distance
        # keeps the assignment kernel in the same operating range regardless
        # of that norm while making probabilities independent of batch
        # composition, so per-streamline confidence is reproducible no matter
        # how inference chunks the batch; temperature only controls the
        # kernel shape.
        scale = norm_squared.min(dim=1, keepdim=True).values.clamp_min(utilis.EPS)  # (B, 1)
        x = 1.0 / (1.0 + (norm_squared / (self.temperature * scale)))
        power = float(self.temperature + 1.0) / 2.0
        x = x ** power
        x = x / (x.sum(dim=1, keepdim=True)).clamp_min(utilis.EPS)  # (B, out_features)

        if return_emb:
            return x, emb
        return x

    @torch.no_grad()
    def set_weight(self, tensor):
        """Update centroids without replacing the optimizer-owned parameter."""
        if tensor.shape != self.weight.shape:
            raise ValueError(
                f"Expected centroid shape {tuple(self.weight.shape)}, got {tuple(tensor.shape)}."
            )
        self.weight.copy_(tensor.to(device=self.weight.device, dtype=self.weight.dtype))


class DGCNN(nn.Module):
    # from DGCNN's repo
    def __init__(self, k=5, input_channel=3, features_len=1024, bias=False):
        super(DGCNN, self).__init__()
        self.k = k

        # WIDE backbone with BatchNorm on every edge-conv layer including
        # conv1 (64/64/128/256, concat 512). 20dcbeb's wide variant had the
        # conv1 BatchNorm commented out; the 2026-8-25-era narrow backbone was
        # 32/32/64/128, also fully BatchNormalized.
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=bias),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=bias),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=bias),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=bias),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, features_len, kernel_size=1, bias=bias),
                                    nn.BatchNorm1d(features_len),
                                    nn.LeakyReLU(negative_slope=0.2))

    def forward_single(self, x, idx_seq=None):
        x = utilis.get_graph_feature(x, k=self.k, idx=idx_seq) # (B, 2D, P, k)
        x = self.conv1(x) # (B, 64, P, k)
        x1 = x.max(dim=-1)[0] # (B, 64, P)

        x = utilis.get_graph_feature(x1, k=self.k, idx=idx_seq) # (B, 128, P, k)
        x = self.conv2(x) # (B, 64, P, k)
        x2 = x.max(dim=-1)[0] # (B, 64, P)

        x = utilis.get_graph_feature(x2, k=self.k, idx=idx_seq) # (B, 128, P, k)
        x = self.conv3(x) # (B, 128, P, k)
        x3 = x.max(dim=-1)[0] # (B, 128, P)

        x = utilis.get_graph_feature(x3, k=self.k, idx=idx_seq) # (B, 128, P, k)
        x = self.conv4(x) # (B, 256, P, k)
        x4 = x.max(dim=-1)[0] # (B, 256, P)

        x = torch.cat((x1, x2, x3, x4), dim=1) # (B, 512, P)
        x = self.conv5(x) # (B, features_len, P)

        return x  # (B, features_len, P)

    def forward(self, x, idx_seq=None):
        if x.dim() == 4:
            B, N, P, D = x.size()
            x = x.reshape(B * N, P, D)  # (B*N, P, D)

        x = x.permute(0, 2, 1)  # (B*N, D, P)

        x_feat = self.forward_single(x, idx_seq=idx_seq)  # (B*N, feat_len, P)

        return x_feat  # (B*N, feat_len, P)
