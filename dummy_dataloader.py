import torch
from torch.utils.data import DataLoader, Subset

from dummy_dataset import (
    KRadarGTDetectionDataset,
    KRadarMultiSequenceGTDetectionDataset,
    KRadarRADRAEDataset,
    detection_collate,
)
from zxy_data_path import get_gt_txt_path, get_rad_rae_npy_root_dir


def get_config_sequences(cfg):
    sequences = getattr(cfg, "sequences", None)
    if sequences is None:
        sequences = (cfg.sequence,)

    if isinstance(sequences, int):
        sequences = (sequences,)

    sequences = tuple(sequences)
    if len(sequences) == 0:
        raise ValueError("cfg.sequences must not be empty")

    return sequences


def build_detection_dataset_for_sequence(cfg, sequence):
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(),
        sequence,
    )

    return KRadarGTDetectionDataset(
        radar_dataset=radar_dataset,
        gt_txt_path=get_gt_txt_path(cfg, sequence=sequence),
        sequence=sequence,
    )


def build_train_val_dataloaders(
    cfg,
    batch_size,
    train_ratio,
    seed,
    num_workers,
    limit_samples,
):
    sequence_datasets = [
        build_detection_dataset_for_sequence(cfg, sequence)
        for sequence in get_config_sequences(cfg)
    ]
    full_dataset = KRadarMultiSequenceGTDetectionDataset(
        sequence_datasets=sequence_datasets
    )

    train_indices = []
    val_indices = []
    remaining_limit = limit_samples
    split_generator = torch.Generator()
    split_generator.manual_seed(seed)

    for sequence_range in full_dataset.get_sequence_ranges():
        start = sequence_range["start"]
        end = sequence_range["end"]
        sequence_indices = list(range(start, end))

        if remaining_limit is not None:
            if remaining_limit <= 0:
                break
            sequence_indices = sequence_indices[:remaining_limit]
            remaining_limit -= len(sequence_indices)

        if len(sequence_indices) == 0:
            continue

        random_order = torch.randperm(
            len(sequence_indices),
            generator=split_generator
        ).tolist()
        sequence_indices = [sequence_indices[idx] for idx in random_order]

        train_size = int(len(sequence_indices) * train_ratio)
        train_indices.extend(sequence_indices[:train_size])
        val_indices.extend(sequence_indices[train_size:])

    if len(train_indices) == 0:
        raise ValueError("Training split is empty. Increase --limit-samples or train_ratio.")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=num_workers,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate,
        num_workers=num_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def prepare_model_inputs(batch, device):
    rad = batch["rad"].to(device, dtype=torch.float32)
    rae = batch["rae"].to(device, dtype=torch.float32)

    if rad.ndim != 4 or rae.ndim != 4:
        raise ValueError(
            f"Expected batched RAD/RAE tensors, got rad={rad.shape}, rae={rae.shape}"
        )

    # Dataset tensors are [B, R, A, D/E]. The model expects [B, D/E, R, A].
    rad = rad.permute(0, 3, 1, 2).contiguous()
    rae = rae.permute(0, 3, 1, 2).contiguous()

    return rad, rae
