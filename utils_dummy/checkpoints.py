import os
import shutil
import copy
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
        is_best,
        clone_for_memory=False
    ):
    model_for_state_dict = model.module if isinstance(model, torch.nn.DataParallel) else model
    payload = {
        "epoch": epoch,
        "saved_at": saved_at,
        "model_state_dict": model_for_state_dict.state_dict(),
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
            "class_names": getattr(args, "class_names", None),
            "class_to_idx": getattr(args, "class_to_idx", None),
            "background_weight": args.background_weight,
            "match_iou_thresh": getattr(args, "match_iou_thresh", None),
            "score_thresh": args.score_thresh,
            "eval_iou_thresh": args.eval_iou_thresh,
            "train_ratio": args.train_ratio,
            "split_mode": getattr(args, "split_mode", None),
            "split_dir": getattr(args, "split_dir", None),
            "seed": args.seed,
            "limit_samples": args.limit_samples,
        },
    }

    if clone_for_memory:
        payload = clone_checkpoint_payload_for_memory(payload)

    return payload


def clone_checkpoint_payload_for_memory(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: clone_checkpoint_payload_for_memory(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, list):
        return [clone_checkpoint_payload_for_memory(child_value) for child_value in value]
    if isinstance(value, tuple):
        return tuple(clone_checkpoint_payload_for_memory(child_value) for child_value in value)
    return copy.deepcopy(value)


def get_model_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_num_classes_from_state_dict(state_dict, num_boxes):
    cls_weight = state_dict["decoder.cls_head.weight"]
    total_output_dim = cls_weight.shape[0]
    if total_output_dim % num_boxes != 0:
        raise ValueError(
            f"Cannot infer num_classes: cls_head output dim {total_output_dim} "
            f"is not divisible by num_boxes {num_boxes}"
        )

    return total_output_dim // num_boxes - 1


def get_num_classes_from_checkpoint(checkpoint, num_boxes):
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config", {})
        if config.get("num_classes") is not None:
            return int(config["num_classes"])
        if checkpoint.get("num_classes") is not None:
            return int(checkpoint["num_classes"])

    state_dict = get_model_state_dict_from_checkpoint(checkpoint)
    return infer_num_classes_from_state_dict(state_dict, num_boxes)


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


def save_named_checkpoint_payload(
        checkpoint_dir,
        payload,
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
    torch.save(payload, best_checkpoint_path)
    return best_checkpoint_path


def remove_named_checkpoints(checkpoint_dir, name_prefix):
    if not os.path.isdir(checkpoint_dir):
        return

    filename_prefix = f"{name_prefix}_epoch_"
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(filename_prefix) and filename.endswith(".pth"):
            os.remove(os.path.join(checkpoint_dir, filename))


def save_replacing_named_checkpoint_copy(
        checkpoint_dir,
        source_checkpoint_path,
        best_epoch,
        best_map,
        name_prefix
    ):
    remove_named_checkpoints(checkpoint_dir, name_prefix)
    return save_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=source_checkpoint_path,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix=name_prefix
    )


def save_replacing_named_checkpoint_payload(
        checkpoint_dir,
        payload,
        best_epoch,
        best_map,
        name_prefix
    ):
    remove_named_checkpoints(checkpoint_dir, name_prefix)
    return save_named_checkpoint_payload(
        checkpoint_dir=checkpoint_dir,
        payload=payload,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix=name_prefix
    )


def save_best_checkpoint_copy(checkpoint_dir, source_checkpoint_path, best_epoch, best_map):
    return save_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=source_checkpoint_path,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix="best"
    )
