import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
from dummy_dataloader import (build_train_val_dataloaders,get_config_sequences,prepare_model_inputs)
from dummy_evaluation import *
from dummy_module import MVRSS3DModel
from utils_dummy.checkpoints import *
from utils_dummy.logging_utils import *
from utils_dummy.other_helping_dunctions import *
from zxy_config import DataConfig

@torch.no_grad()
def hungarian_cost_match(
        pred_boxes,
        gt_boxes,
        cost_bbox=1.0,
        cost_iou=1.0
    ):
    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    bbox_cost = torch.cdist(
        pred_boxes[:, :6],
        gt_boxes[:, :6],
        p=1
    )

    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)

    ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)
    iou_cost = 1.0 - ious

    total_cost = cost_bbox * bbox_cost + cost_iou * iou_cost

    cost_matrix = total_cost.detach().cpu().numpy()

    from scipy.optimize import linear_sum_assignment
    matched_pred, matched_gt = linear_sum_assignment(cost_matrix)

    matched_pred_indices = torch.as_tensor(
        matched_pred,
        dtype=torch.long,
        device=device
    )

    matched_gt_indices = torch.as_tensor(
        matched_gt,
        dtype=torch.long,
        device=device
    )
    return matched_pred_indices, matched_gt_indices

def detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        num_classes,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.1
    ):
    if isinstance(outputs, dict):
        pred_boxes = outputs["box_pred"].sigmoid()
        pred_logits = outputs["cls_pred"]
    else:
        box_dim = 7
        expected_output_dim = box_dim + num_classes + 1

        if outputs.shape[-1] != expected_output_dim:
            raise ValueError(
                f"Expected output dim {expected_output_dim}, got {outputs.shape[-1]}"
            )

        pred_boxes = outputs[:, :, :box_dim].sigmoid()
        pred_logits = outputs[:, :, box_dim:]

    device = pred_boxes.device
    batch_size = pred_boxes.shape[0]
    num_queries = pred_boxes.shape[1]

    target_classes = torch.full(
        (batch_size, num_queries),
        fill_value=num_classes,
        dtype=torch.long,
        device=device
    )

    matched_pred_boxes_all = []
    matched_gt_boxes_all = []

    for b in range(batch_size):
        gt_boxes = gt_boxes_list[b].to(device)
        gt_labels = gt_labels_list[b].to(device)

        if gt_boxes.shape[0] == 0:
            continue

        pred_boxes_b = pred_boxes[b]

        matched_pred_indices, matched_gt_indices = hungarian_cost_match(
            pred_boxes=pred_boxes_b,
            gt_boxes=gt_boxes,
            cost_bbox=1.0,
            cost_iou=1.0
        )

        if matched_pred_indices.numel() == 0:
            continue

        
        target_classes[b, matched_pred_indices] = gt_labels[matched_gt_indices]

        
        matched_pred_boxes_all.append(
            pred_boxes_b[matched_pred_indices]
        )
        matched_gt_boxes_all.append(
            gt_boxes[matched_gt_indices]
        )

    class_weights = torch.ones(num_classes + 1, device=device)
    class_weights[-1] = background_weight

    cls_loss = F.cross_entropy(
        pred_logits.reshape(-1, num_classes + 1),
        target_classes.reshape(-1),
        weight=class_weights
    )

    if len(matched_pred_boxes_all) > 0:
        matched_pred_boxes_all = torch.cat(matched_pred_boxes_all, dim=0)
        matched_gt_boxes_all = torch.cat(matched_gt_boxes_all, dim=0)

        box_loss = F.smooth_l1_loss(
            matched_pred_boxes_all,
            matched_gt_boxes_all
        )
    else:
        box_loss = torch.tensor(0.0, device=device)

    total_loss = (
        box_loss_weight * box_loss
        + cls_loss_weight * cls_loss
    )

    loss_dict = {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item()
    }

    return total_loss, loss_dict

def train_one_epoch(
        model,
        dataloader,
        optimizer,
        device,
        num_classes,
        epoch=None,
        num_epochs=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.1
    ):
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    num_batches = 0

    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch + 1}/{num_epochs}"
    else:
        desc = "Training"

    pbar = tqdm(dataloader, desc=desc, ncols=120)

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        loss, loss_dict = detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            num_classes=num_classes,
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            background_weight=background_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        num_batches += 1

        avg_loss = total_loss_sum / num_batches
        avg_box = box_loss_sum / num_batches
        avg_cls = cls_loss_sum / num_batches

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "box": f"{avg_box:.4f}",
            "cls": f"{avg_cls:.4f}",
        })

    avg_total_loss = total_loss_sum / max(num_batches, 1)
    avg_box_loss = box_loss_sum / max(num_batches, 1)
    avg_cls_loss = cls_loss_sum / max(num_batches, 1)

    return {
        "train_loss": avg_total_loss,
        "train_box_loss": avg_box_loss,
        "train_cls_loss": avg_cls_loss,
    }


@torch.no_grad()
def validate_loss(
        model,
        dataloader,
        device,
        num_classes,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.1
    ):
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation loss", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        _, loss_dict = detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            num_classes=num_classes,
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            background_weight=background_weight
        )

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        num_batches += 1

    return {
        "val_loss": total_loss_sum / max(num_batches, 1),
        "val_box_loss": box_loss_sum / max(num_batches, 1),
        "val_cls_loss": cls_loss_sum / max(num_batches, 1),
    }

def argparse_args():
    parser = argparse.ArgumentParser(description="Train the dummy MVRSS detection module.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--background-weight", type=float, default=0.6)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--best-window-size", type=int, default=5)
    parser.add_argument("--checkpoint-base-dir", default="checkpoints")
    parser.add_argument("--log-base-dir", default="runs")
    args = parser.parse_args()
    return args


def main():

    args = argparse_args()
    if args.best_window_size <= 0:
        raise ValueError(f"--best-window-size must be greater than 0, got {args.best_window_size}")

    set_seed(args.seed)
    cfg = DataConfig()
    device = torch.device("cuda")
    (train_dataset,val_dataset,train_loader,val_loader) = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples
        )
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty. Adjust --train-ratio or --limit-samples.")

    model = MVRSS3DModel(
        d_in=64,
        e_in=37,
        num_boxes=args.num_boxes,
        box_dim=7,
        num_classes=args.num_classes,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(8, 8)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_state = BestCheckpointState()
    window_best_state = BestCheckpointState()

    configured_sequences = get_config_sequences(cfg)
    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name="mvrss_detection",
        sequences=configured_sequences
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name="mvrss_detection",
        sequence=configured_sequences
    )
    write_tensorboard_run_config(
        writer=writer,
        cfg=cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        learning_rate=args.lr,
        num_boxes=args.num_boxes,
        background_weight=args.background_weight,
        eval_iou_thresh=args.eval_iou_thresh
    )
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=args.num_classes,
            epoch=epoch,
            num_epochs=args.epochs,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            background_weight=args.background_weight
        )
        val_loss_metrics = validate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=args.num_classes,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            background_weight=args.background_weight
        )
        eval_metrics = evaluate_train_val_iou(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            num_classes=args.num_classes,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=min(args.num_boxes, 20)
        )
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate
        )

        checkpoint_path = save_epoch_and_update_best_checkpoint(
            best_state=best_state,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            args=args,
            cfg=cfg,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate
        )

        save_window_best_checkpoint_if_ready(
            window_best_state=window_best_state,
            checkpoint_dirs=checkpoint_dirs,
            checkpoint_key=checkpoint_key,
            checkpoint_path=checkpoint_path,
            epoch=epoch + 1,
            total_epochs=args.epochs,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            window_size=args.best_window_size
        )
        append_training_history(
            history=history,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1
        )
    writer.close()

    print_training_history(history)
    save_global_best_checkpoint(
        best_state=best_state,
        checkpoint_dirs=checkpoint_dirs,
        checkpoint_key=checkpoint_key
    )

if __name__ == "__main__":
    main()
