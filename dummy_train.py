import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dummy_dataloader import (build_train_val_dataloaders, get_config_sequences, prepare_model_inputs)
from dummy_dataset import CLASS_NAMES, CLASS_TO_IDX
from dummy_evaluation import boxes_3d_to_ra_xyxy, evaluate_train_val_iou
from dummy_module import (
    MVRSS3DModel,
    MVRSS3DModelDeform,
    MVRSS3DModelDeformDepthwiseSeparable,
)
from dummy_module_multiscale import MVRSS3DModel2
from utils_dummy.checkpoints import *
from utils_dummy.logging_utils import *
from utils_dummy.other_helping_dunctions import *
from zxy_config import DataConfig


NUM_CLASSES = 2
BACKGROUND_CLASS_ID = 2
NUM_OUTPUT_CLASSES = 3


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def pairwise_box_giou_2d(boxes1, boxes2):
    """
    Computes Generalized IoU (GIoU) between two sets of boxes.
    Boxes should be in [x_min, y_min, x_max, y_max] format.
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # Standard IoU Intersections
    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    iou = inter / union

    # Enclosing Box
    enclose_left_top = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_right_bottom = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_right_bottom - enclose_left_top).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1] + 1e-6

    # GIoU Calculation
    giou = iou - (enclose_area - union) / enclose_area
    return giou


def focal_loss_fn(logits, targets, class_weights, gamma=2.0):
    """
    Applies Focal Loss to combat the massive background class imbalance.
    """
    ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
    pt = torch.exp(-ce_loss) # Get the predicted probability for the target class
    focal_term = (1.0 - pt) ** gamma
    return (focal_term * ce_loss).mean()


@torch.no_grad()
def hungarian_cost_match(
        pred_boxes,
        gt_boxes,
        cost_bbox=1.0,
        cost_giou=2.0  # Increased priority for IoU overlap
    ):
    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    bbox_cost = torch.cdist(pred_boxes[:, :6], gt_boxes[:, :6], p=1)

    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)

    # Use GIoU for matching cost instead of standard IoU
    gious = pairwise_box_giou_2d(pred_ra_boxes, gt_ra_boxes)
    giou_cost = 1.0 - gious

    total_cost = cost_bbox * bbox_cost + cost_giou * giou_cost

    cost_matrix = total_cost.detach().cpu().numpy()

    from scipy.optimize import linear_sum_assignment
    matched_pred, matched_gt = linear_sum_assignment(cost_matrix)

    matched_pred_indices = torch.as_tensor(matched_pred, dtype=torch.long, device=device)
    matched_gt_indices = torch.as_tensor(matched_gt, dtype=torch.long, device=device)
    
    return matched_pred_indices, matched_gt_indices

@torch.no_grad()
def greedy_cost_match(
        pred_boxes,
        gt_boxes,
        cost_bbox=1.0,
        cost_giou=2.0
    ):
    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    # 1. L1 bbox cost
    bbox_cost = torch.cdist(pred_boxes[:, :6], gt_boxes[:, :6], p=1)

    # 2. RA-plane GIoU cost
    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)

    gious = pairwise_box_giou_2d(pred_ra_boxes, gt_ra_boxes)
    giou_cost = 1.0 - gious

    # 3. Total cost
    total_cost = cost_bbox * bbox_cost + cost_giou * giou_cost

    num_pred, num_gt = total_cost.shape
    max_matches = min(num_pred, num_gt)

    # 4. Flatten and sort all candidate pairs by cost
    flat_cost = total_cost.flatten()
    sorted_indices = torch.argsort(flat_cost)

    matched_pred = []
    matched_gt = []

    used_pred = torch.zeros(num_pred, dtype=torch.bool, device=device)
    used_gt = torch.zeros(num_gt, dtype=torch.bool, device=device)

    # 5. Greedy matching
    for idx in sorted_indices:
        pred_idx = idx // num_gt
        gt_idx = idx % num_gt

        if used_pred[pred_idx] or used_gt[gt_idx]:
            continue

        matched_pred.append(pred_idx)
        matched_gt.append(gt_idx)

        used_pred[pred_idx] = True
        used_gt[gt_idx] = True

        if len(matched_pred) >= max_matches:
            break

    if len(matched_pred) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    matched_pred_indices = torch.stack(matched_pred).long()
    matched_gt_indices = torch.stack(matched_gt).long()

    return matched_pred_indices, matched_gt_indices

def detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.5
    ):
    if isinstance(outputs, dict):
        pred_boxes = outputs["box_pred"].sigmoid()
        pred_logits = outputs["cls_pred"]
    else:
        box_dim = 7
        pred_boxes = outputs[:, :, :box_dim].sigmoid()
        pred_logits = outputs[:, :, box_dim:]

    device = pred_boxes.device
    batch_size = pred_boxes.shape[0]
    num_queries = pred_boxes.shape[1]

    target_classes = torch.full(
        (batch_size, num_queries),
        fill_value=BACKGROUND_CLASS_ID,
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

        # matched_pred_indices, matched_gt_indices = hungarian_cost_match(
        #     pred_boxes=pred_boxes_b,
        #     gt_boxes=gt_boxes,
        #     cost_bbox=1.0,
        #     cost_giou=2.0 
        # )
        matched_pred_indices, matched_gt_indices = greedy_cost_match(
            pred_boxes=pred_boxes_b,
            gt_boxes=gt_boxes,
            cost_bbox=1.0,
            cost_giou=2.0
        )

        if matched_pred_indices.numel() == 0:
            continue

        target_classes[b, matched_pred_indices] = gt_labels[matched_gt_indices]
        
        matched_pred_boxes_all.append(pred_boxes_b[matched_pred_indices])
        matched_gt_boxes_all.append(gt_boxes[matched_gt_indices])

    class_weights = torch.ones(NUM_OUTPUT_CLASSES, device=device)
    class_weights[BACKGROUND_CLASS_ID] = background_weight

    # 1. Classification Loss (Upgraded to Focal Loss)
    cls_loss = focal_loss_fn(
        pred_logits.reshape(-1, NUM_OUTPUT_CLASSES),
        target_classes.reshape(-1),
        class_weights=class_weights,
        gamma=2.0
    )

    if len(matched_pred_boxes_all) > 0:
        matched_pred_boxes_all = torch.cat(matched_pred_boxes_all, dim=0)
        matched_gt_boxes_all = torch.cat(matched_gt_boxes_all, dim=0)

        # 2. Coordinate distance loss (L1)
        pred_boxes_dim = matched_pred_boxes_all[:, :6]
        gt_boxes_dim = matched_gt_boxes_all[:, :6]
        loss_dim = F.l1_loss(pred_boxes_dim, gt_boxes_dim)

        # 3. Circular angle loss (Dim 6)
        pred_angles = matched_pred_boxes_all[:, 6]
        gt_angles = matched_gt_boxes_all[:, 6]
        angle_diff = torch.abs(pred_angles - gt_angles)
        angle_diff = torch.min(angle_diff, 1.0 - angle_diff) 
        loss_angle = angle_diff.mean()

        # 4. GIoU loss integration (Upgraded from standard IoU)
        pred_ra = boxes_3d_to_ra_xyxy(matched_pred_boxes_all)
        gt_ra = boxes_3d_to_ra_xyxy(matched_gt_boxes_all)
        gious = pairwise_box_giou_2d(pred_ra, gt_ra).diag()
        loss_giou = (1.0 - gious).mean()

        box_loss = loss_dim + loss_angle + (2.0 * loss_giou)
        box_loss = 1*(loss_dim + loss_angle) + (2.0 * loss_giou)
    else:
        box_loss = torch.tensor(0.0, device=device)

    total_loss = (box_loss_weight * box_loss) + (cls_loss_weight * cls_loss)

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
        epoch=None,
        num_epochs=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.5
    ):
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    num_batches = 0

    desc = f"Epoch {epoch + 1}/{num_epochs}" if epoch is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, ncols=120)

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        loss, loss_dict = detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
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

        pbar.set_postfix({
            "loss": f"{(total_loss_sum / num_batches):.4f}",
            "box": f"{(box_loss_sum / num_batches):.4f}",
            "cls": f"{(cls_loss_sum / num_batches):.4f}",
        })

    return {
        "train_loss": total_loss_sum / max(num_batches, 1),
        "train_box_loss": box_loss_sum / max(num_batches, 1),
        "train_cls_loss": cls_loss_sum / max(num_batches, 1),
    }


@torch.no_grad()
def validate_loss(
        model,
        dataloader,
        device,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.5
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
    # Training timeline increased to allow the Hungarian Matcher to stabilize
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=50)#50
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--background-weight", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="file", choices=["random", "file", "order"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--checkpoint-epoch-step", type=int, default=10)
    parser.add_argument("--checkpoint-base-dir", default="checkpoints")
    parser.add_argument("--log-base-dir", default="runs")
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--model-type", type=str, default="model3", choices=["model1", "model2", "model3", "model4", "model5"])
    args = parser.parse_args()
    return args


def main():
    args = argparse_args()
    if args.checkpoint_epoch_step <= 0:
        raise ValueError(f"--checkpoint-epoch-step must be greater than 0")

    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_config_sequences(cfg)
    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES.copy()
    args.class_to_idx = CLASS_TO_IDX.copy()

    print(f"Training classes: {CLASS_NAMES}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpu_count = torch.cuda.device_count()
        unavailable_gpu_ids = [
            gpu_id for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if len(unavailable_gpu_ids) > 0:
            raise ValueError(
                f"Requested GPU ids {unavailable_gpu_ids}, "
                f"but only {available_gpu_count} CUDA device(s) are available."
            )
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = []
        device = torch.device("cpu")

    (train_dataset, val_dataset, train_loader, val_loader) = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=CLASS_TO_IDX,
        split_mode=args.split_mode,
        split_dir=args.split_dir,
    )
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    if args.model_type == "model1":
        model = MVRSS3DModel(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(16, 16)
        ).to(device)
    elif args.model_type == "model2":
        model = MVRSS3DModel2(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(8, 8)
        ).to(device)
    elif args.model_type == "model3":
        model = MVRSS3DModelDeform(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(16, 16)
        ).to(device)
    # elif args.model_type == "model4":
    #     model = MVRSS3DModelDeform(
    #         d_in=64,
    #         e_in=37,
    #         num_boxes=args.num_boxes,
    #         box_dim=7,
    #         num_classes=NUM_CLASSES,
    #         feature_channels=64,
    #         fusion_hidden_channels=64,
    #         decoder_hidden_channels=128,
    #         pooled_size=(4, 4)
    #     ).to(device)
    # elif args.model_type == "model5":
    #     model = MVRSS3DModelDeformDepthwiseSeparable(
    #         d_in=64,
    #         e_in=37,
    #         num_boxes=args.num_boxes,
    #         box_dim=7,
    #         num_classes=NUM_CLASSES,
    #         feature_channels=64,
    #         fusion_hidden_channels=64,
    #         decoder_hidden_channels=128,
    #         pooled_size=(8,8)
    #     ).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0]
        )
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_state = BestCheckpointState()

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
        writer=writer, cfg=cfg, num_epochs=args.epochs, batch_size=args.batch_size,
        train_size=len(train_dataset), val_size=len(val_dataset), learning_rate=args.lr,
        num_boxes=args.num_boxes, num_classes=NUM_CLASSES, class_names=CLASS_NAMES,
        background_weight=args.background_weight, eval_iou_thresh=args.eval_iou_thresh
    )

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
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
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            background_weight=args.background_weight
        )


        eval_metrics = evaluate_train_val_iou(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            num_classes=NUM_CLASSES,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=args.num_boxes
        )
        
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1, learning_rate=learning_rate
        )

        checkpoint_path = save_epoch_and_update_best_checkpoint(
            best_state=best_state, checkpoint_dir=checkpoint_dir, model=model,
            optimizer=optimizer, args=args, cfg=cfg, epoch=epoch + 1,
            train_metrics=train_metrics, val_metrics=val_metrics, f1=f1,
            learning_rate=learning_rate,
            total_epochs=args.epochs,
            checkpoint_epoch_step=args.checkpoint_epoch_step
        )
        
        append_training_history(
            history=history, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1
        )
        
    writer.close()
    save_global_best_checkpoint(best_state=best_state, checkpoint_dirs=checkpoint_dirs, checkpoint_key=checkpoint_key)

if __name__ == "__main__":
    main()
