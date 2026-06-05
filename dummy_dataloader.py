import os

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


def build_detection_dataset_for_sequence(
        cfg,
        sequence,
        class_to_idx=None,
        ignore_unmapped_classes=True
    ):
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(),
        sequence,
    )

    return KRadarGTDetectionDataset(
        radar_dataset=radar_dataset,
        gt_txt_path=get_gt_txt_path(cfg, sequence=sequence),
        class_to_idx=class_to_idx,
        sequence=sequence,
        ignore_unmapped_classes=ignore_unmapped_classes,
    )


def build_train_val_dataloaders(
    cfg,
    batch_size,
    train_ratio,
    seed,
    num_workers,
    limit_samples,
    class_to_idx=None,
    ignore_unmapped_classes=True,
    split_mode="random",
    split_dir="split",
):
    sequence_datasets = [
        build_detection_dataset_for_sequence(
            cfg=cfg,
            sequence=sequence,
            class_to_idx=class_to_idx,
            ignore_unmapped_classes=ignore_unmapped_classes,
        )
        for sequence in get_config_sequences(cfg)
    ]
    full_dataset = KRadarMultiSequenceGTDetectionDataset(
        sequence_datasets=sequence_datasets
    )

    if split_mode == "random":
        train_indices, val_indices = build_random_split_indices(
            full_dataset=full_dataset,
            train_ratio=train_ratio,
            seed=seed,
            limit_samples=limit_samples,
        )
    elif split_mode == "file":
        train_indices, val_indices = build_file_split_indices(
            full_dataset=full_dataset,
            split_dir=split_dir,
            allowed_sequences=get_config_sequences(cfg),
            limit_samples=limit_samples,
        )
    elif split_mode == "order":
        train_indices = list(range(0, int(len(full_dataset) * train_ratio)))
        val_indices = list(range(int(len(full_dataset) * train_ratio), len(full_dataset)))
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

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


def build_random_split_indices(full_dataset, train_ratio, seed, limit_samples):
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

    return train_indices, val_indices


def split_line_to_sequence_and_frame_names(line):
    line = line.strip()
    if line == "" or line.startswith("#"):
        return None

    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid split line: {line!r}")

    sequence = int(parts[0])
    frame_token = os.path.splitext(parts[1])[0]
    frame_name = frame_token.split("_")[0]
    if frame_name == "":
        raise ValueError(f"Invalid split frame token: {line!r}")

    return sequence, [frame_name]


def read_split_file(split_path, allowed_sequences):
    allowed_sequences = set(int(sequence) for sequence in allowed_sequences)
    split_by_sequence = {sequence: [] for sequence in allowed_sequences}

    with open(split_path, "r") as split_file:
        for line in split_file:
            parsed = split_line_to_sequence_and_frame_names(line)
            if parsed is None:
                continue

            sequence, frame_names = parsed
            if sequence not in allowed_sequences:
                continue

            split_by_sequence.setdefault(sequence, []).append(frame_names)

    return split_by_sequence


def build_sequence_index_lookup(full_dataset):
    lookup = {}
    ranges = full_dataset.get_sequence_ranges()

    for dataset, sequence_range in zip(full_dataset.sequence_datasets, ranges):
        sequence = int(sequence_range["sequence"])
        start = sequence_range["start"]
        frame_name_to_local_idx = {
            frame_name: local_idx
            for local_idx, frame_name in enumerate(dataset.radar_dataset.frame_names)
        }
        lookup[sequence] = {
            "start": start,
            "frame_name_to_local_idx": frame_name_to_local_idx,
        }

    return lookup


def split_entries_to_indices(split_by_sequence, sequence_lookup, split_name):
    indices = []
    seen = set()
    missing = []

    for sequence, frame_name_candidates_list in split_by_sequence.items():
        if sequence not in sequence_lookup:
            continue

        start = sequence_lookup[sequence]["start"]
        frame_name_to_local_idx = sequence_lookup[sequence]["frame_name_to_local_idx"]

        for frame_name_candidates in frame_name_candidates_list:
            local_idx = None
            for frame_name in frame_name_candidates:
                local_idx = frame_name_to_local_idx.get(frame_name)
                if local_idx is not None:
                    break

            if local_idx is None:
                missing.append(f"{sequence},{'/'.join(frame_name_candidates)}")
                continue

            index = start + local_idx
            if index in seen:
                continue

            indices.append(index)
            seen.add(index)

    if len(missing) > 0:
        preview = ", ".join(missing[:10])
        print(
            f"Warning: skipped {len(missing)} samples from {split_name} because they were "
            f"not found in rad/rae files. First missing entries: {preview}"
        )

    return indices


def build_file_split_indices(full_dataset, split_dir, allowed_sequences, limit_samples):
    train_split_path = os.path.join(split_dir, "train.txt")
    val_split_path = os.path.join(split_dir, "test.txt")

    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Training split file not found: {train_split_path}")
    if not os.path.exists(val_split_path):
        raise FileNotFoundError(f"Validation split file not found: {val_split_path}")

    sequence_lookup = build_sequence_index_lookup(full_dataset)
    train_by_sequence = read_split_file(train_split_path, allowed_sequences)
    val_by_sequence = read_split_file(val_split_path, allowed_sequences)

    train_indices = split_entries_to_indices(
        split_by_sequence=train_by_sequence,
        sequence_lookup=sequence_lookup,
        split_name="train.txt",
    )
    val_indices = split_entries_to_indices(
        split_by_sequence=val_by_sequence,
        sequence_lookup=sequence_lookup,
        split_name="test.txt",
    )

    if limit_samples is not None:
        train_indices = train_indices[:limit_samples]
        val_indices = val_indices[:limit_samples]

    return train_indices, val_indices


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
