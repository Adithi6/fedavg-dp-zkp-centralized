import logging
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


def _partition_to_tensordataset(
    partition,
    normalize_mean: list[float],
    normalize_std: list[float],
) -> TensorDataset:
    """
    Convert one Flower/Hugging Face partition into a PyTorch TensorDataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(normalize_mean), tuple(normalize_std)),
    ])

    images = []
    labels = []

    for item in partition:
        img = transform(item["image"])
        label = int(item["label"])

        images.append(img)
        labels.append(label)

    x_tensor = torch.stack(images)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)


def make_client_loaders(
    n_clients: int,
    batch_size: int,
    alpha: float,
    dataset_name: str,
    partition_by: str,
    min_partition_size: int,
    self_balancing: bool,
    seed: int,
    test_batch_size: int,
    normalize_mean: list[float],
    normalize_std: list[float],
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create non-IID client DataLoaders using Flower DirichletPartitioner.

    Args:
        n_clients: number of federated clients
        batch_size: training DataLoader batch size
        alpha: Dirichlet concentration parameter
        dataset_name: dataset identifier
        partition_by: column used for partitioning
        min_partition_size: minimum samples per partition
        self_balancing: whether to self-balance partitions
        seed: random seed for partitioner
        test_batch_size: test DataLoader batch size
        normalize_mean: normalization mean
        normalize_std: normalization std

    Returns:
        client_loaders: list of client train DataLoaders
        test_loader: DataLoader for full test split
    """
    partitioner = DirichletPartitioner(
        num_partitions=n_clients,
        partition_by=partition_by,
        alpha=alpha,
        min_partition_size=min_partition_size,
        self_balancing=self_balancing,
        seed=seed,
    )

    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner},
    )

    client_loaders: List[DataLoader] = []

    logging.info(
        f"Creating {n_clients} Dirichlet-based client partitions | "
        f"dataset={dataset_name} alpha={alpha} partition_by={partition_by} "
        f"min_partition_size={min_partition_size} self_balancing={self_balancing} seed={seed}"
    )

    for client_id in range(n_clients):
        partition = fds.load_partition(client_id, "train")
        dataset = _partition_to_tensordataset(
            partition,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

        labels = [int(item["label"]) for item in partition]
        unique_labels = sorted(set(labels))
        logging.info(
            f"Client {client_id} | samples={len(labels)} | labels={unique_labels}"
        )

    test_partition = fds.load_split("test")
    test_dataset = _partition_to_tensordataset(
        test_partition,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    logging.info(
        f"Created {n_clients} non-IID client loaders using DirichletPartitioner"
    )

    return client_loaders, test_loader