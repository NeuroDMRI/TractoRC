"""CPU/Gloo checks for the DDP helpers; run from the repository root."""

import pathlib
import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractorc import distributed, train_utilis
from tractorc.model import TractoRC
from tractorc.utilis import get_dgcnn_idx


def _gloo_worker(rank, init_method, checkpoint_path, result_dir):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch_dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=2
    )
    context = distributed.DistributedContext(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=torch.device("cpu"),
    )
    values = torch.tensor([[float(rank + 1)]], requires_grad=True)
    gathered = distributed.all_gather_cat(values, context)
    gathered.square().mean().backward()
    reduced = distributed.sum_across_ranks([rank + 1], torch.device("cpu"), context)

    logits = torch.tensor([[[float(rank), 1.0 - rank]]], requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)
    cluster_terms = train_utilis._compute_loss_terms(
        {"prob_f": probabilities, "prob_m": probabilities},
        None,
        None,
        {"assignment", "balance"},
        {"assignment": 1.0, "balance": 1.0},
        "RMS",
        distributed_context=context,
        gather_cluster_stats=True,
    )
    sum(cluster_terms.values()).backward()

    wrapped = DistributedDataParallel(torch.nn.Linear(1, 1))
    if context.is_master:
        train_utilis.save_checkpoint(wrapped, checkpoint_path)
    torch_dist.barrier()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretraining_model = DistributedDataParallel(
        TractoRC(num_keypoints=2, num_clusters=2, feat_len=8),
        find_unused_parameters=True,
    )
    coords = torch.randn(1, 2, 14, 3) + rank
    centers = torch.zeros(1, 3)
    scales = torch.ones(1)
    outputs = train_utilis.extract_pretraining_outputs(
        pretraining_model,
        coords,
        coords,
        (centers, scales),
        (centers, scales),
        idx=get_dgcnn_idx(5, 14, 2),
    )
    sum(value.square().mean() for value in outputs.values()).backward()
    pretraining_gradient = next(pretraining_model.parameters()).grad.detach().clone()
    torch.save(
        {
            "gathered": gathered.detach(),
            "gradient": values.grad,
            "cluster_gradient": logits.grad,
            "cluster_loss": sum(term.detach() for term in cluster_terms.values()),
            "pretraining_gradient": pretraining_gradient,
            "reduced": reduced,
            "state_keys": tuple(checkpoint["model_state_dict"]),
        },
        pathlib.Path(result_dir) / f"rank_{rank}.pt",
    )
    torch_dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def test_single_process_context_and_eval_sharding(self):
        context = distributed.DistributedContext(0, 0, 1, torch.device("cpu"))
        self.assertFalse(context.enabled)
        self.assertTrue(context.is_master)

        dataset = TensorDataset(torch.arange(5))
        shards = [
            list(distributed.DistributedEvalSampler(dataset, rank, 2))
            for rank in range(2)
        ]
        self.assertEqual(sorted(shards[0] + shards[1]), list(range(len(dataset))))
        self.assertFalse(set(shards[0]) & set(shards[1]))

        train_dataset = TensorDataset(torch.arange(8))
        train_shards = [
            list(
                DistributedSampler(
                    train_dataset,
                    num_replicas=2,
                    rank=rank,
                    shuffle=False,
                    drop_last=True,
                )
            )
            for rank in range(2)
        ]
        self.assertEqual(sorted(train_shards[0] + train_shards[1]), list(range(8)))
        self.assertFalse(set(train_shards[0]) & set(train_shards[1]))

    def test_formal_resume_preserves_optimizer_and_scheduler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = pathlib.Path(temp_dir) / "formal.pth"
            model = TractoRC(num_keypoints=2, num_clusters=2, feat_len=8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = train_utilis.build_scheduler(optimizer, epochs=8)
            model.cluster_head.weight.sum().backward()
            optimizer.step()
            scheduler.step()
            train_utilis.save_checkpoint(
                model,
                checkpoint_path,
                optimizer,
                scheduler,
                epoch=3,
                val_loss=0.25,
            )

            resumed_model = TractoRC(num_keypoints=2, num_clusters=2, feat_len=8)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
            resumed_scheduler = train_utilis.build_scheduler(resumed_optimizer, epochs=8)
            _, resumed_optimizer, resumed_scheduler, next_epoch, best_val = (
                train_utilis.load_checkpoint(
                    resumed_model,
                    checkpoint_path,
                    resumed_optimizer,
                    resumed_scheduler,
                )
            )

            self.assertEqual(next_epoch, 4)
            self.assertEqual(best_val, 0.25)
            self.assertIsInstance(
                scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            )
            source_state = optimizer.state[model.cluster_head.weight]
            resumed_state = resumed_optimizer.state[resumed_model.cluster_head.weight]
            self.assertEqual(source_state["step"], resumed_state["step"])
            self.assertTrue(torch.equal(source_state["exp_avg"], resumed_state["exp_avg"]))
            self.assertTrue(
                torch.equal(source_state["exp_avg_sq"], resumed_state["exp_avg_sq"])
            )
            self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_ddp_conv_gradient_uses_parameter_layout(self):
        model = TractoRC(num_keypoints=2, num_clusters=2, feat_len=8)
        context = distributed.DistributedContext(0, 0, 2, torch.device("cpu"))
        weights = [
            module.weight
            for module in model.backbone.modules()
            if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1)
        ]

        train_utilis._prepare_ddp_gradient_layout(model, context)
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(weight.grad.stride() == weight.stride() for weight in weights))
        weight = weights[0]
        model.backbone.conv1(torch.randn(2, 6, 14, 5)).sum().backward()
        self.assertEqual(weight.grad.stride(), weight.stride())

    def test_gloo_gather_and_unwrapped_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            init_file = root / "gloo_init"
            checkpoint_path = root / "checkpoint.pth"
            mp.spawn(
                _gloo_worker,
                args=(f"file://{init_file}", str(checkpoint_path), str(root)),
                nprocs=2,
                join=True,
            )

            results = [
                torch.load(root / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
                for rank in range(2)
            ]
            for result in results:
                self.assertTrue(torch.equal(result["gathered"], torch.tensor([[1.0], [2.0]])))
                self.assertTrue(torch.isfinite(result["gradient"]).all())
                self.assertTrue(torch.isfinite(result["cluster_gradient"]).all())
                self.assertTrue(torch.isfinite(result["pretraining_gradient"]).all())
                self.assertEqual(result["reduced"], [3.0])
                self.assertEqual(set(result["state_keys"]), {"weight", "bias"})
            self.assertTrue(torch.equal(results[0]["cluster_loss"], results[1]["cluster_loss"]))
            self.assertTrue(
                torch.equal(
                    results[0]["pretraining_gradient"], results[1]["pretraining_gradient"]
                )
            )


if __name__ == "__main__":
    unittest.main()
