import os
import shutil
from datetime import datetime

import torch


def _create_unique_checkpoint_dir(checkpoint_dir):
    suffix = 1
    unique_checkpoint_dir = checkpoint_dir
    while os.path.exists(unique_checkpoint_dir):
        unique_checkpoint_dir = f"{checkpoint_dir}_{suffix}"
        suffix += 1

    os.makedirs(unique_checkpoint_dir, exist_ok=False)
    return unique_checkpoint_dir


def format_sequence_run_name(sequences):
    if isinstance(sequences, int):
        return f"seq{sequences}"

    sequences = tuple(sequences)
    if len(sequences) == 1:
        return f"seq{sequences[0]}"

    ranges = []
    start = sequences[0]
    previous = sequences[0]
    for sequence in sequences[1:]:
        if sequence == previous + 1:
            previous = sequence
            continue

        ranges.append((start, previous))
        start = sequence
        previous = sequence
    ranges.append((start, previous))

    range_texts = [
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    ]
    return 'seq' + '_'.join(range_texts)


def create_checkpoint_run_dir(base_dir, experiment_name, sequence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{format_sequence_run_name(sequence)}_{timestamp}"
    checkpoint_dir = os.path.join(base_dir, experiment_name, run_name)
    return _create_unique_checkpoint_dir(checkpoint_dir)


def create_checkpoint_run_dirs(base_dir, experiment_name, sequences):
    sequences = tuple(sequences)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{format_sequence_run_name(sequences)}_{timestamp}"
    checkpoint_dir = os.path.join(base_dir, experiment_name, run_name)
    return {sequences: _create_unique_checkpoint_dir(checkpoint_dir)}


def metric_for_filename(value):
    return f"{value:.4f}".replace(".", "p")


def build_checkpoint_payload(
        model,
        optimizer,
        args,
        cfg,
        epoch,
        train_metrics,
        val_metrics,
        f1,
        learning_rate,
        saved_at,
        is_best
    ):
    return {
        "epoch": epoch,
        "saved_at": saved_at,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "f1": f1,
        "mAP": val_metrics["mAP"],
        "eval_iou_thresh": val_metrics["iou_thresh"],
        "learning_rate": learning_rate,
        "is_best": is_best,
        "config": {
            "sequence": cfg.sequence,
            "sequences": getattr(cfg, "sequences", None),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_boxes": args.num_boxes,
            "num_classes": args.num_classes,
            "background_weight": args.background_weight,
            "score_thresh": args.score_thresh,
            "eval_iou_thresh": args.eval_iou_thresh,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "limit_samples": args.limit_samples,
        },
    }


def save_epoch_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        args,
        cfg,
        epoch,
        train_metrics,
        val_metrics,
        f1,
        learning_rate,
        is_best
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_text = metric_for_filename(val_metrics["mAP"])
    filename = (
        f"candidate_epoch_{epoch:03d}_{saved_at}_mAP_{map_text}.pth"
    )
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        args=args,
        cfg=cfg,
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        f1=f1,
        learning_rate=learning_rate,
        saved_at=saved_at,
        is_best=is_best
    )

    torch.save(payload, checkpoint_path)

    return checkpoint_path


def save_named_checkpoint_copy(
        checkpoint_dir,
        source_checkpoint_path,
        best_epoch,
        best_map,
        name_prefix
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_text = metric_for_filename(best_map)
    best_filename = (
        f"{name_prefix}_epoch_{best_epoch:03d}_{saved_at}_mAP_{map_text}.pth"
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    shutil.copy2(source_checkpoint_path, best_checkpoint_path)
    return best_checkpoint_path


def save_best_checkpoint_copy(checkpoint_dir, source_checkpoint_path, best_epoch, best_map):
    return save_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=source_checkpoint_path,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix="best"
    )
