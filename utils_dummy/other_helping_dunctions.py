from dataclasses import dataclass
import random

import numpy as np
import torch

from utils_dummy.checkpoints import (
    build_checkpoint_payload,
    save_best_checkpoint_copy,
    save_epoch_checkpoint,
    save_named_checkpoint_copy,
    save_named_checkpoint_payload,
    save_replacing_named_checkpoint_copy,
    save_replacing_named_checkpoint_payload,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_training_history(history, epoch, train_metrics, val_metrics, f1):
    history.append({
        "epoch": epoch,
        "train_loss": train_metrics["train_loss"],
        "train_box_loss": train_metrics["train_box_loss"],
        "train_cls_loss": train_metrics["train_cls_loss"],
        "train_iou": train_metrics["train_iou"],
        "val_loss": val_metrics["val_loss"],
        "val_box_loss": val_metrics["val_box_loss"],
        "val_cls_loss": val_metrics["val_cls_loss"],
        "val_mAP": val_metrics["mAP"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_iou": val_metrics["val_iou"],
        "val_f1": f1,
        "iou": val_metrics["iou_thresh"],
        "tp": val_metrics["tp"],
        "fp": val_metrics["fp"],
        "fn": val_metrics["fn"],
    })


def build_epoch_eval_metrics(train_metrics, eval_metrics, val_loss_metrics):
    train_metrics["train_iou"] = eval_metrics["train_eval_metrics"]["mean_iou"]

    val_metrics = eval_metrics["val_eval_metrics"].copy()
    val_metrics["val_iou"] = val_metrics["mean_iou"]
    val_metrics.update(val_loss_metrics)

    precision = val_metrics["precision"]
    recall = val_metrics["recall"]
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return val_metrics, f1


@dataclass
class BestCheckpointState:
    map_score: float = -1.0
    epoch: int = -1
    train_metrics: object = None
    metrics: object = None
    f1: float = 0.0
    checkpoint_path: object = None
    checkpoint_payload: object = None
    global_best_path: object = None

    def reset(self):
        self.map_score = -1.0
        self.epoch = -1
        self.train_metrics = None
        self.metrics = None
        self.f1 = 0.0
        self.checkpoint_path = None
        self.checkpoint_payload = None
        self.global_best_path = None

    def update(
            self,
            epoch,
            train_metrics,
            val_metrics,
            f1,
            checkpoint_path=None,
            checkpoint_payload=None,
            global_best_path=None
        ):
        self.map_score = val_metrics["mAP"]
        self.epoch = epoch
        self.train_metrics = train_metrics.copy()
        self.metrics = val_metrics.copy()
        self.f1 = f1
        self.checkpoint_path = checkpoint_path
        self.checkpoint_payload = checkpoint_payload
        self.global_best_path = global_best_path

    def is_better(self, val_metrics):
        return val_metrics["mAP"] > self.map_score


def save_epoch_and_update_best_checkpoint(
        best_state,
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
        total_epochs,
        checkpoint_epoch_step
    ):
    is_best = best_state.is_better(val_metrics)
    should_save_checkpoint = (
        epoch % checkpoint_epoch_step == 0
        or epoch == total_epochs
    )

    if not should_save_checkpoint and not is_best:
        return None

    if should_save_checkpoint:
        checkpoint_path = save_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            args=args,
            cfg=cfg,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
            is_best=is_best
        )
        checkpoint_payload = None
    else:
        checkpoint_path = None
        checkpoint_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            cfg=cfg,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
            saved_at="memory",
            is_best=is_best,
            clone_for_memory=True
        )

    if is_best:
        if checkpoint_path is not None:
            global_best_path = save_replacing_named_checkpoint_copy(
                checkpoint_dir=checkpoint_dir,
                source_checkpoint_path=checkpoint_path,
                best_epoch=epoch,
                best_map=val_metrics["mAP"],
                name_prefix="global_best"
            )
        else:
            global_best_path = save_replacing_named_checkpoint_payload(
                checkpoint_dir=checkpoint_dir,
                payload=checkpoint_payload,
                best_epoch=epoch,
                best_map=val_metrics["mAP"],
                name_prefix="global_best"
            )

        best_state.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
            global_best_path=global_best_path
        )

    return checkpoint_path


def save_window_best_checkpoint_if_ready(
        window_best_state,
        checkpoint_dirs,
        checkpoint_key,
        checkpoint_path,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        f1,
        window_size
    ):
    if checkpoint_path is None:
        return None, None

    if window_best_state.is_better(val_metrics):
        window_best_state.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            checkpoint_path=checkpoint_path
        )

    should_save_window_best = (
        window_best_state.checkpoint_path is not None
        and (epoch % window_size == 0 or epoch == total_epochs)
    )
    if not should_save_window_best:
        return None, None

    best_checkpoint_paths = {}
    for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
        best_checkpoint_paths[sequence] = save_best_checkpoint_copy(
            checkpoint_dir=sequence_checkpoint_dir,
            source_checkpoint_path=window_best_state.checkpoint_path,
            best_epoch=window_best_state.epoch,
            best_map=window_best_state.map_score
        )
    best_checkpoint_path = best_checkpoint_paths[checkpoint_key]
    window_best_state.reset()

    return best_checkpoint_path, best_checkpoint_paths


def save_global_best_checkpoint(best_state, checkpoint_dirs, checkpoint_key):
    if best_state.global_best_path is not None:
        return best_state.global_best_path, {checkpoint_key: best_state.global_best_path}

    if best_state.checkpoint_path is None and best_state.checkpoint_payload is None:
        return None, None

    global_best_checkpoint_paths = {}
    for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
        if best_state.checkpoint_path is not None:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_copy(
                checkpoint_dir=sequence_checkpoint_dir,
                source_checkpoint_path=best_state.checkpoint_path,
                best_epoch=best_state.epoch,
                best_map=best_state.map_score,
                name_prefix="global_best"
            )
        else:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_payload(
                checkpoint_dir=sequence_checkpoint_dir,
                payload=best_state.checkpoint_payload,
                best_epoch=best_state.epoch,
                best_map=best_state.map_score,
                name_prefix="global_best"
            )
    global_best_checkpoint_path = global_best_checkpoint_paths[checkpoint_key]

    return global_best_checkpoint_path, global_best_checkpoint_paths
