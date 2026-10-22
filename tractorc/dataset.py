from torch.utils.data import Dataset
import numpy as np
import itertools

from . import utilis

from dipy.tracking.streamline import set_number_of_points, length
import concurrent.futures
import os


class TractographyData(object):
    """
    Represents tractography data.

    Attributes:
        input_path (str): Path to the tractography file.
        streamlines (list): List of streamlines loaded from the file.
        point_per_streamline (int): Number of points to resample each streamline to.
        min_length (int): Minimum length threshold for streamlines.
        max_length (int): Maximum length threshold for streamlines.

    Methods:
        resample_streamlines(nb_points, min_l, max_l): Resamples streamlines to a fixed number of points and filters by length.
    """
    def __init__(self, input_path, point_per_streamline=11, min_length=20, max_length=260, recenter=False):
        super().__init__()
        self.input_path = input_path

        self.min_length = min_length
        self.max_length = max_length
        self.point_per_streamline = point_per_streamline

        self.streamlines = utilis.load_tract(input_path)
        self.resample_streamlines(point_per_streamline, min_length, max_length)
        self.length = len(self.streamlines)
        if self.length == 0:
            raise ValueError(
                f"No streamlines remain in {input_path} after length filtering "
                f"[{min_length}, {max_length}]."
            )

        self.center_point, _ = utilis.streamline_feature_stats(self.streamlines)
        if recenter:
            self.streamlines -= self.center_point
            self.feature_center = np.zeros(3, dtype=np.float32)
        else:
            self.feature_center = self.center_point.copy()

        _, self.feature_scale = utilis.streamline_feature_stats(
            self.streamlines, center=self.feature_center
        )

    def resample_streamlines(self, nb_points, min_l, max_l):
        streamline_lengths = length(self.streamlines)
        kept = [
            streamline
            for streamline, streamline_length in zip(self.streamlines, streamline_lengths)
            if min_l <= streamline_length <= max_l
        ]
        self.streamlines = np.asarray(
            set_number_of_points(kept, nb_points), dtype=np.float32
        )

    def random_sample(self, num_samples, rng=None):
        rng = np.random if rng is None else rng
        if num_samples >= self.length:
            indices = np.arange(self.length)
            return self.streamlines.copy(), indices
        indices = rng.choice(self.length, num_samples, replace=False)
        return self.streamlines[indices].copy(), indices

    def feature_stats(self):
        """Return fixed full-tractogram normalization statistics."""
        return self.feature_center.copy(), self.feature_scale.copy()

class PairedTractographyDataset(Dataset):
    """Paired subjects for joint registration and clustering training.

    ``pairing="chain"`` (training) keeps one subject pair per batch: each
    epoch shuffles the subjects once, and batch ``b`` draws its fixed sample
    from ``perm[b]`` and its moving sample from ``perm[(b + 1) % n]``, so the
    last batch wraps around to the first subject and every subject appears
    once on each side per epoch. Streamlines are re-sampled at random for
    every batch item. ``pairing="all"`` enumerates every unordered subject
    pair (deterministic validation).
    """

    def __init__(
        self,
        path,
        point_per_streamline=11,
        min_length=20,
        max_length=260,
        num_samples=3000,
        pretrain=False,
        seed=None,
        deterministic=False,
        random_flip=True,
        random_swap=True,
        load_workers=None,
        recenter=False,
        pairing="chain",
        batch_size=1,
    ):
        super().__init__()
        self.tract_pathlist = self.search_tract_paths(path)
        self.tract_nums = len(self.tract_pathlist)
        self.tract_pairs = list(itertools.combinations(range(self.tract_nums), 2))
        if self.tract_nums == 0:
            raise ValueError(f"No tractography files found under {path}.")

        self.pretrain = pretrain
        self.nb_points = point_per_streamline
        self.min_length = min_length
        self.max_length = max_length
        self.num_samples = num_samples
        self.seed = 0 if seed is None else int(seed)
        self.deterministic = deterministic
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.recenter = recenter

        if pairing not in ("chain", "all"):
            raise ValueError("pairing must be 'chain' or 'all'")
        if not pretrain and pairing == "chain":
            if self.tract_nums < 2:
                raise ValueError("Chain pairing needs at least two subjects.")
            if int(batch_size) < 1:
                raise ValueError("batch_size must be at least 1")
        self.pairing = pairing
        self.batch_size = int(batch_size)
        self.epoch = 0
        self.chain_order = self._chain_permutation(0)

        # Loading each tractogram in a separate process is useful for I/O, but
        # creating one worker per file can briefly multiply the memory needed
        # for large tractograms. Four workers keeps loading parallel without
        # exhausting a shared training node.
        if load_workers is None:
            load_workers = min(4, self.tract_nums, os.cpu_count() or 1)
        if load_workers < 1:
            raise ValueError("load_workers must be at least 1")

        with concurrent.futures.ProcessPoolExecutor(max_workers=load_workers) as executor:
            futures = [
                executor.submit(
                    TractographyData,
                    input_path=str(tract_path),
                    point_per_streamline=self.nb_points,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    recenter=self.recenter,
                )
                for tract_path in self.tract_pathlist
            ]
            self.tract_data = [future.result() for future in futures]

    @staticmethod
    def search_tract_paths(input_dir):
        return utilis.search_tract_paths(input_dir)

    def __len__(self):
        if self.pretrain:
            return self.tract_nums
        if self.pairing == "chain":
            # One batch per shuffled subject; every batch holds ``batch_size``
            # items of that batch's single subject pair.
            return self.tract_nums * self.batch_size
        return len(self.tract_pairs)

    def _chain_permutation(self, epoch):
        """Deterministic subject order for ``epoch`` (seeded per epoch)."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        return rng.permutation(self.tract_nums)

    def set_epoch(self, epoch):
        """Reshuffle the cyclic subject chain; call once per training epoch.

        Must run in the process that owns the DataLoader iterators: workers
        fork from this copy when a new epoch iterator is created, which is
        why chain pairing uses non-persistent workers.
        """
        self.epoch = int(epoch)
        if self.pretrain or self.pairing != "chain":
            return
        self.chain_order = self._chain_permutation(epoch)

    def _rng(self, index, stream):
        if self.deterministic:
            return np.random.default_rng(
                np.random.SeedSequence([self.seed, int(index), stream])
            )
        return np.random

    def _sample_streamlines(self, tract_data, rng):
        coords, _ = tract_data.random_sample(self.num_samples, rng=rng)
        if self.random_flip:
            mask = rng.random(len(coords)) > 0.5
            coords[mask] = coords[mask, ::-1]
        center, scale = tract_data.feature_stats()
        return coords, center, scale

    def __getitem__(self, idx):
        if self.pretrain:
            return self._sample_streamlines(self.tract_data[idx], self._rng(idx, 0))

        if self.pairing == "chain":
            pair_index = (idx // self.batch_size) % self.tract_nums
            fixed_idx = int(self.chain_order[pair_index])
            moving_idx = int(self.chain_order[(pair_index + 1) % self.tract_nums])
        else:
            moving_idx, fixed_idx = self.tract_pairs[idx]
            if self.random_swap and self._rng(idx, 0).random() < 0.5:
                moving_idx, fixed_idx = fixed_idx, moving_idx

        mcoords, mcenter, mscale = self._sample_streamlines(
            self.tract_data[moving_idx], self._rng(idx, 1)
        )
        fcoords, fcenter, fscale = self._sample_streamlines(
            self.tract_data[fixed_idx], self._rng(idx, 2)
        )
        return fcoords, mcoords, fcenter, fscale, mcenter, mscale
