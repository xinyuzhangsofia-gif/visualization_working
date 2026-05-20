import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def print_training_history(history):
    if len(history) == 0:
        return

    print("\nTraining history")
    print(
        f"{'epoch':>5} "
        f"{'train_loss':>11} "
        f"{'train_box':>10} "
        f"{'train_cls':>10} "
        f"{'val_loss':>9} "
        f"{'val_box':>9} "
        f"{'val_cls':>9} "
        f"{'val_mAP':>8} "
        f"{'val_precision':>14} "
        f"{'val_recall':>11} "
        f"{'val_iou':>9} "
        f"{'val_f1':>8} "
        f"{'IoU_thr':>8} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6}"
    )
    print("-" * 158)

    for row in history:
        print(
            f"{row['epoch']:5d} "
            f"{row['train_loss']:11.4f} "
            f"{row['train_box_loss']:10.4f} "
            f"{row['train_cls_loss']:10.4f} "
            f"{row['val_loss']:9.4f} "
            f"{row['val_box_loss']:9.4f} "
            f"{row['val_cls_loss']:9.4f} "
            f"{row['val_mAP']:8.4f} "
            f"{row['val_precision']:14.4f} "
            f"{row['val_recall']:11.4f} "
            f"{row['val_iou']:9.4f} "
            f"{row['val_f1']:8.4f} "
            f"{row['iou']:8.4f} "
            f"{row['tp']:6d} "
            f"{row['fp']:6d} "
            f"{row['fn']:6d}"
        )


def create_tensorboard_run_dir(base_dir, experiment_name, sequence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"seq{sequence}_{timestamp}"
    log_dir = os.path.join(base_dir, experiment_name, run_name)

    suffix = 1
    unique_log_dir = log_dir
    while os.path.exists(unique_log_dir):
        unique_log_dir = f"{log_dir}_{suffix}"
        suffix += 1

    os.makedirs(unique_log_dir, exist_ok=False)
    return unique_log_dir


def create_tensorboard_writer(base_dir, experiment_name, sequence):
    log_dir = create_tensorboard_run_dir(
        base_dir=base_dir,
        experiment_name=experiment_name,
        sequence=sequence
    )
    return SummaryWriter(log_dir=log_dir), log_dir


def write_tensorboard_run_config(
        writer,
        cfg,
        num_epochs,
        batch_size,
        train_size,
        val_size,
        learning_rate,
        num_boxes,
        background_weight,
        eval_iou_thresh
    ):
    config_text = "\n".join([
        f"sequence: {cfg.sequence}",
        f"sequences: {getattr(cfg, 'sequences', None)}",
        f"num_epochs: {num_epochs}",
        f"batch_size: {batch_size}",
        f"train_size: {train_size}",
        f"val_size: {val_size}",
        f"learning_rate: {learning_rate}",
        f"num_boxes: {num_boxes}",
        f"background_weight: {background_weight}",
        f"eval_iou_thresh: {eval_iou_thresh}",
    ])
    writer.add_text("run/config", config_text, 0)
    writer.flush()


def write_tensorboard_metrics(writer, epoch, train_metrics, val_metrics, f1, learning_rate):
    writer.add_scalar("training_metrics/train_loss", train_metrics["train_loss"], epoch)
    writer.add_scalar("training_metrics/train_box_loss", train_metrics["train_box_loss"], epoch)
    writer.add_scalar("training_metrics/train_cls_loss", train_metrics["train_cls_loss"], epoch)

    writer.add_scalar("validation_metrics/val_loss", val_metrics["val_loss"], epoch)
    writer.add_scalar("validation_metrics/val_box_loss", val_metrics["val_box_loss"], epoch)
    writer.add_scalar("validation_metrics/val_cls_loss", val_metrics["val_cls_loss"], epoch)
    writer.add_scalar("validation_metrics/val_mAP", val_metrics["mAP"], epoch)
    writer.add_scalar("validation_metrics/val_precision", val_metrics["precision"], epoch)
    writer.add_scalar("validation_metrics/val_recall", val_metrics["recall"], epoch)
    writer.add_scalar("validation_metrics/val_iou", val_metrics["val_iou"], epoch)
    writer.add_scalar("validation_metrics/val_f1", f1, epoch)

    writer.add_scalar("parameters/learning_rate", learning_rate, epoch)
    writer.add_scalar("parameters/eval_iou_thresh", val_metrics["iou_thresh"], epoch)
    writer.flush()
