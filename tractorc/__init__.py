"""TractoRC core library.

Modules:
    utilis            -- tractography/geometry IO and helpers
    module            -- nn building blocks (DGCNN, ClusteringLayer, KeypointLayer)
    model             -- TractoRC model
    keypoint_aligners -- Affine/Rigid/TPS keypoint aligners
    dataset           -- tractography torch datasets
    augmentation      -- affine augmentation
    runtime           -- device/checkpoint/CLI helpers shared by entry points
    train_utilis      -- training loops, losses, checkpoint IO
"""
