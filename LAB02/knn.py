"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    for i in range(num_train):
        for j in range(num_test):
            diff = x_train[i].reshape(-1) - x_test[j].reshape(-1)
            dists[i, j] = (diff * diff).sum()
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set using one Python loop.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    train_flat = x_train.flatten(start_dim=1)
    test_flat = x_test.flatten(start_dim=1)
    for i in range(num_train):
        diff = test_flat - train_flat[i]
        dists[i] = (diff * diff).sum(dim=1)
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set without Python loops.
    """
    train_flat = x_train.flatten(start_dim=1)
    test_flat = x_test.flatten(start_dim=1)
    train_sq = (train_flat * train_flat).sum(dim=1, keepdim=True)
    test_sq = (test_flat * test_flat).sum(dim=1).unsqueeze(0)
    # Reuse the output matrix to avoid additional pairwise-sized temporaries.
    dists = train_flat.mm(test_flat.t())
    dists.mul_(-2)
    dists.add_(train_sq)
    dists.add_(test_sq)
    dists.clamp_(min=0)
    return dists
