import argparse
import os
import re
import torch
import tqdm
from dummy_dataloader import build_train_val_dataloaders, prepare_model_inputs
from dummy_dataset import (
    class_to_idx_from_class_names,
    fallback_class_names_for_num_classes,
    normalize_class_names,
    normalize_class_to_idx,
)
from dummy_visualize import build_model, load_checkpoint, select_device
from utils_dummy.checkpoints import get_num_classes_from_checkpoint
from zxy_config import DataConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dummy MVRSS checkpoints.")
    parser.add_argument("--checkpoint-root", default=
                        "./checkpoints/mvrss_detection/seq1-11_20260522_183125_346849/global_best_epoch_021_20260522_200603_mAP_0p0913.pth")
    parser.add_argument("--epoch-step", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=45)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--num-boxes", type=int, default=64)
    return parser.parse_args()


def get_checkpoint_class_info(checkpoint, num_boxes):
    num_classes = get_num_classes_from_checkpoint(
        checkpoint=checkpoint,
        num_boxes=num_boxes
    )
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    class_names = normalize_class_names(config.get("class_names"))
    if class_names is None:
        class_names = fallback_class_names_for_num_classes(num_classes)

    class_to_idx = normalize_class_to_idx(config.get("class_to_idx"))
    if class_to_idx is None:
        class_to_idx = class_to_idx_from_class_names(class_names)

    return num_classes, class_names, class_to_idx


def boxes_3d_to_ra_xyxy(boxes):
    r = boxes[:, 0]
    a = boxes[:, 1]
    r_w = boxes[:, 3]
    a_w = boxes[:, 4]

    r_min = r - r_w / 2.0
    r_max = r + r_w / 2.0
    a_min = a - a_w / 2.0
    a_max = a + a_w / 2.0

    return torch.stack([r_min, a_min, r_max, a_max], dim=-1)


def box_iou_2d(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device
        )

    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )
    union = area1[:, None] + area2[None, :] - inter + 1e-6

    return inter / union


def average_precision(tp_flags, fp_flags, num_gt):
    if num_gt == 0 or len(tp_flags) == 0:
        return 0.0

    tp = torch.tensor(tp_flags, dtype=torch.float32)
    fp = torch.tensor(fp_flags, dtype=torch.float32)

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for idx in range(precision.numel() - 1, 0, -1):
        precision[idx - 1] = torch.maximum(precision[idx - 1], precision[idx])

    changed = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])

    return float(ap.item())


def compute_map(predictions_by_class, gt_by_class, num_classes, iou_thresh):
    ap_per_class = {}

    for class_id in range(num_classes):
        predictions = sorted(
            predictions_by_class[class_id],
            key=lambda item: item["score"],
            reverse=True
        )
        gt_for_class = gt_by_class[class_id]
        num_gt = sum(data["boxes"].shape[0] for data in gt_for_class.values())

        matched_gt = {
            image_id: torch.zeros(data["boxes"].shape[0], dtype=torch.bool)
            for image_id, data in gt_for_class.items()
        }

        tp_flags = []
        fp_flags = []

        for pred in predictions:
            image_id = pred["image_id"]
            pred_box = pred["box"]

            if image_id not in gt_for_class or gt_for_class[image_id]["boxes"].shape[0] == 0:
                tp_flags.append(0)
                fp_flags.append(1)
                continue

            gt_boxes = gt_for_class[image_id]["boxes"].to(pred_box.device)
            ious = box_iou_2d(
                boxes_3d_to_ra_xyxy(pred_box.unsqueeze(0)),
                boxes_3d_to_ra_xyxy(gt_boxes)
            ).squeeze(0)

            best_iou, best_gt_idx = ious.max(dim=0)
            best_gt_idx = int(best_gt_idx.item())

            if best_iou.item() >= iou_thresh and not matched_gt[image_id][best_gt_idx]:
                tp_flags.append(1)
                fp_flags.append(0)
                matched_gt[image_id][best_gt_idx] = True
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        ap_per_class[class_id] = average_precision(tp_flags, fp_flags, num_gt)

    classes_with_gt = [
        class_id
        for class_id in range(num_classes)
        if sum(data["boxes"].shape[0] for data in gt_by_class[class_id].values()) > 0
    ]

    if len(classes_with_gt) == 0:
        mean_ap = 0.0
    else:
        mean_ap = sum(ap_per_class[class_id] for class_id in classes_with_gt) / len(classes_with_gt)

    return mean_ap, ap_per_class


@torch.no_grad()
def evaluate_precision_recall(
        model,
        dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        score_thresh=0.2, 
        iou_thresh=0.5,
        max_detections=64
    ):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    total_iou_count = 0
    
    predictions_by_class = {class_id: [] for class_id in range(num_classes)}
    gt_by_class = {class_id: {} for class_id in range(num_classes)}
    image_counter = 0

    for batch in tqdm.tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        pred_boxes = outputs["box_pred"].sigmoid()
        pred_logits = outputs["cls_pred"]
        pred_probs = pred_logits.softmax(dim=-1)

        foreground_probs = pred_probs[:, :, :num_classes]
        background_probs = pred_probs[:, :, num_classes]
        pred_scores, pred_labels = foreground_probs.max(dim=-1)

        batch_size = pred_boxes.shape[0]

        for b in range(batch_size):
            # 1. Setup ID
            if "image_id" in batch:
                image_id = batch["image_id"][b]
            elif "file_idx" in batch:
                image_id = batch["file_idx"][b]
            else:
                image_id = image_counter
                image_counter += 1

            scores_b = pred_scores[b]
            labels_b = pred_labels[b]
            boxes_b = pred_boxes[b]
            background_scores_b = background_probs[b]

            # 2. Extract Ground Truth
            gt_boxes_all = batch["gt_boxes"][b].to(device)
            gt_labels_all = batch["gt_labels"][b].to(device)
            valid_gt = gt_labels_all < num_classes
            gt_boxes = gt_boxes_all[valid_gt]
            gt_labels = gt_labels_all[valid_gt]

            # Register GT for AP calculation
            for class_id in range(num_classes):
                class_gt_boxes = gt_boxes[gt_labels == class_id].detach().cpu()
                gt_by_class[class_id][image_id] = {"boxes": class_gt_boxes}

            # 3. Filter for mAP Calculation (NO SCORE THRESHOLD HERE)
            map_keep = scores_b > background_scores_b
            map_boxes = boxes_b[map_keep]
            map_labels = labels_b[map_keep]
            map_scores = scores_b[map_keep]

            # Top-K for mAP
            if map_scores.shape[0] > max_detections:
                topk_scores, topk_indices = map_scores.topk(max_detections)
                map_boxes = map_boxes[topk_indices]
                map_labels = map_labels[topk_indices]
                map_scores = topk_scores

            # Populate predictions for rigorous mAP calculation
            for pred_box, pred_label, pred_score in zip(map_boxes, map_labels, map_scores):
                predictions_by_class[int(pred_label.item())].append({
                    "image_id": image_id,
                    "score": float(pred_score.item()),
                    "box": pred_box.detach().cpu(),
                })

            # 4. Filter for Fixed-Point Metrics (TP/FP/FN/F1 at specific threshold)
            point_keep = (map_scores > score_thresh)
            point_boxes = map_boxes[point_keep]
            point_labels = map_labels[point_keep]
            point_scores = map_scores[point_keep]

            if point_boxes.shape[0] == 0:
                total_fn += gt_boxes.shape[0]
                continue

            if gt_boxes.shape[0] == 0:
                total_fp += point_boxes.shape[0]
                continue

            # Calculate strict-point TP/FP
            pred_ra_boxes = boxes_3d_to_ra_xyxy(point_boxes)
            gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
            ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)

            matched_gt = set()
            order = point_scores.argsort(descending=True)

            for pred_idx_tensor in order:
                pred_idx = pred_idx_tensor.item()
                best_iou = -1.0
                best_gt_idx = -1

                for gt_idx in range(gt_boxes.shape[0]):
                    if gt_idx in matched_gt:
                        continue
                    if point_labels[pred_idx].item() != gt_labels[gt_idx].item():
                        continue

                    iou_value = ious[pred_idx, gt_idx].item()
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou >= iou_thresh:
                    total_tp += 1
                    total_iou += best_iou
                    total_iou_count += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_fn += gt_boxes.shape[0] - len(matched_gt)

    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = total_iou / max(total_iou_count, 1)
    
    mean_ap, ap_per_class = compute_map(
        predictions_by_class=predictions_by_class,
        gt_by_class=gt_by_class,
        num_classes=num_classes,
        iou_thresh=iou_thresh
    )

    return {
        "precision": precision,
        "recall": recall,
        "mAP": mean_ap,
        "ap_per_class": ap_per_class,
        "mean_iou": mean_iou,
        "iou_thresh": iou_thresh,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"epoch_(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


def find_epoch_checkpoints(checkpoint_root, epoch_step):
    if epoch_step <= 0:
        raise ValueError(f"--epoch-step must be greater than 0, got {epoch_step}")

    if os.path.isfile(checkpoint_root):
        epoch = checkpoint_epoch(checkpoint_root)
        if epoch is None:
            checkpoint = torch.load(checkpoint_root, map_location="cpu")
            epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
        return [(epoch, checkpoint_root)]

    checkpoint_paths = []
    for filename in os.listdir(checkpoint_root):
        if not filename.endswith(".pth"):
            continue
        if not (filename.startswith("epoch_") or filename.startswith("candidate_epoch_")):
            continue

        checkpoint_path = os.path.join(checkpoint_root, filename)
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is None or epoch % epoch_step != 0:
            continue

        checkpoint_paths.append((epoch, checkpoint_path))

    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths


def class_ap_from_name(ap_per_class, class_names, target_name):
    for class_id, class_name in class_names.items():
        if class_name == target_name:
            return ap_per_class.get(class_id, 0.0)
    return 0.0


def metrics_for_graph(metrics, class_names):
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ap_per_class = metrics["ap_per_class"]
    graph_metrics = {
        "mAP": metrics["mAP"],
        "bus_or_truck_ap": class_ap_from_name(ap_per_class, class_names, "Bus or Truck"),
        "sedan_ap": class_ap_from_name(ap_per_class, class_names, "Sedan"),
        "two_wheeler_ap": class_ap_from_name(ap_per_class, class_names, "Two-wheeler"),
        "pedestrian_ap": class_ap_from_name(ap_per_class, class_names, "Pedestrian"),
        "iou": metrics["mean_iou"],
        "TP": metrics["tp"],
        "FP": metrics["fp"],
        "f1": f1,
    }
    return graph_metrics


def print_evaluation_result(epoch, graph_metrics):
    print(
        f"epoch={epoch}",
        f"mAP={graph_metrics['mAP']:.4f}",
        f"bus_or_truck_ap={graph_metrics['bus_or_truck_ap']:.4f}",
        f"sedan_ap={graph_metrics['sedan_ap']:.4f}",
        f"two_wheeler_ap={graph_metrics['two_wheeler_ap']:.4f}",
        f"pedestrian_ap={graph_metrics['pedestrian_ap']:.4f}",
        f"iou={graph_metrics['iou']:.4f}",
        f"TP={graph_metrics['TP']}",
        f"FP={graph_metrics['FP']}",
        f"f1={graph_metrics['f1']:.4f}",
    )


def print_results_table(results):
    """Print evaluation results as a formatted table."""
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "="*110)
    print(
        f"{'Epoch':<8} {'mAP':<10} {'bus_or_truck_ap':<16} {'sedan_ap':<10} "
        f"{'two_wheeler_ap':<16} {'pedestrian_ap':<16} {'iou':<10} "
        f"{'f1':<10} {'TP':<8} {'FP':<8}"
    )
    print("="*110)
    
    for result in results:
        print(
            f"{result['epoch']:<8} "
            f"{result['mAP']:<10.4f} "
            f"{result['bus_or_truck_ap']:<16.4f} "
            f"{result['sedan_ap']:<10.4f} "
            f"{result['two_wheeler_ap']:<16.4f} "
            f"{result['pedestrian_ap']:<16.4f} "
            f"{result['iou']:<10.4f} "
            f"{result['f1']:<10.4f} "
            f"{result['TP']:<8} "
            f"{result['FP']:<8}"
        )
    
    print("="*110 + "\n")


def main():
    args = parse_args()
    device = select_device()
    cfg = DataConfig()
    checkpoint_paths = find_epoch_checkpoints(args.checkpoint_root, args.epoch_step)
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No epoch checkpoints found in {args.checkpoint_root}")

    first_checkpoint = torch.load(checkpoint_paths[0][1], map_location=device)
    num_classes, class_names, class_to_idx = get_checkpoint_class_info(
        checkpoint=first_checkpoint,
        num_boxes=args.num_boxes
    )
    print(f"Evaluation classes: {class_names}")

    _, validation_dataset, _, validation_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=class_to_idx,
        ignore_unmapped_classes=True,
    )

    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty. Adjust --train-ratio or --limit-samples.")

    model = build_model(device=device, num_boxes=args.num_boxes, num_classes=num_classes)

    def evaluate_checkpoint(checkpoint_path):
        load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device
        )
        metrics = evaluate_precision_recall(
            model=model,
            dataloader=validation_loader,
            device=device,
            num_classes=num_classes,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=args.num_boxes
        )
        return metrics

    results = []
    print(f"Evaluating {len(checkpoint_paths)} checkpoints from {args.checkpoint_root}")
    for epoch, checkpoint_path in tqdm.tqdm(checkpoint_paths, desc="Checkpoints", ncols=120):
        metrics = evaluate_checkpoint(checkpoint_path)
        graph_metrics = metrics_for_graph(metrics, class_names)
        graph_metrics["epoch"] = epoch
        results.append(graph_metrics)
        print_evaluation_result(epoch, graph_metrics)

    print_results_table(results)

if __name__ == "__main__":
    main()