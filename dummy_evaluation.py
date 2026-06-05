import argparse
import os
import re
import torch
import tqdm
from dummy_dataloader import build_train_val_dataloaders, prepare_model_inputs
from dummy_dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
)
from dummy_visualize import build_model, load_checkpoint
from zxy_config import DataConfig


NUM_CLASSES = 2


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def parse_cuda_choice(cuda_text, fallback_gpu_ids_text):
    if cuda_text is None:
        return parse_gpu_ids(fallback_gpu_ids_text)

    cuda_text = cuda_text.strip().lower()
    if cuda_text in ("", "cpu", "none"):
        return []

    gpu_ids = []
    for cuda_part in cuda_text.split(","):
        cuda_part = cuda_part.strip().lower()
        if cuda_part.startswith("cuda:"):
            cuda_part = cuda_part.removeprefix("cuda:")
        if cuda_part.startswith("gpu"):
            gpu_number = int(cuda_part.removeprefix("gpu"))
            if gpu_number <= 0:
                raise ValueError(f"GPU names start from gpu1, got {cuda_part!r}")
            gpu_ids.append(gpu_number - 1)
        else:
            gpu_ids.append(int(cuda_part))

    return gpu_ids


def select_evaluation_device(cuda_text, gpu_ids_text):
    gpu_ids = parse_cuda_choice(cuda_text, gpu_ids_text)
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
        if len(gpu_ids) > 1:
            print(f"Evaluation uses one GPU only; using cuda:{gpu_ids[0]} from {gpu_ids}.")
        return torch.device(f"cuda:{gpu_ids[0]}")

    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dummy MVRSS checkpoints.")
    parser.add_argument("--checkpoint-root", default=
                        "checkpoints/mvrss_detection/seq1-11_20260531_233835_389707/global_best_epoch_047_20260601_041256_mAP_0p0643.pth")
    parser.add_argument("--epoch-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="file", choices=["random", "file"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--model-type", default="model3", choices=["model1", "model2", "model3", "model4", "model5"])
    parser.add_argument("--gpu-ids", default="1,2" )
    parser.add_argument(
        "--cuda",
        default=None,
        help="Choose device: gpu1, gpu2, cuda:0, cuda:1, 0, 1, or cpu."
    )
    return parser.parse_args()


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
            sequence_id: torch.zeros(data["boxes"].shape[0], dtype=torch.bool)
            for sequence_id, data in gt_for_class.items()
        }

        tp_flags = []
        fp_flags = []

        for pred in predictions:
            sequence_id = pred["sequence_id"]
            pred_box = pred["box"]

            if sequence_id not in gt_for_class or gt_for_class[sequence_id]["boxes"].shape[0] == 0:
                tp_flags.append(0)
                fp_flags.append(1)
                continue

            gt_boxes = gt_for_class[sequence_id]["boxes"].to(pred_box.device)
            ious = box_iou_2d(
                boxes_3d_to_ra_xyxy(pred_box.unsqueeze(0)),
                boxes_3d_to_ra_xyxy(gt_boxes)
            ).squeeze(0)

            best_iou, best_gt_idx = ious.max(dim=0)
            best_gt_idx = int(best_gt_idx.item())

            if best_iou.item() >= iou_thresh and not matched_gt[sequence_id][best_gt_idx]:
                tp_flags.append(1)
                fp_flags.append(0)
                matched_gt[sequence_id][best_gt_idx] = True
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
    sequence_counter = 0

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
            if "sequence_id" in batch:
                sequence_id = batch["sequence_id"][b]
            elif "file_idx" in batch:
                sequence_id = batch["file_idx"][b]
            else:
                sequence_id = sequence_counter
                sequence_counter += 1

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
                gt_by_class[class_id][sequence_id] = {"boxes": class_gt_boxes}

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
                    "sequence_id": sequence_id,
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


@torch.no_grad()
def evaluate_train_val_iou(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        score_thresh=0.5,
        iou_thresh=0.5,
        max_detections=20
    ):
    train_eval_metrics = evaluate_precision_recall(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections
    )
    val_eval_metrics = evaluate_precision_recall(
        model=model,
        dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections
    )

    return {
        "train_eval_iou": train_eval_metrics["mean_iou"],
        "val_eval_iou": val_eval_metrics["mean_iou"],
        "train_eval_metrics": train_eval_metrics,
        "val_eval_metrics": val_eval_metrics,
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

    checkpoint_by_epoch = {}
    for filename in os.listdir(checkpoint_root):
        if not filename.endswith(".pth"):
            continue

        is_global_best = filename.startswith("global_best_epoch_")
        is_candidate = filename.startswith("candidate_epoch_")
        is_epoch = filename.startswith("epoch_")
        if not (is_global_best or is_candidate or is_epoch):
            continue

        checkpoint_path = os.path.join(checkpoint_root, filename)
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is None:
            continue

        if not is_global_best and epoch % epoch_step != 0:
            continue

        existing = checkpoint_by_epoch.get(epoch)
        if existing is None or is_global_best:
            checkpoint_by_epoch[epoch] = (epoch, checkpoint_path)

    checkpoint_paths = list(checkpoint_by_epoch.values())
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
        f"{'iou':<10} {'f1':<10} {'TP':<8} {'FP':<8}"
    )
    print("="*110)
    
    for result in results:
        print(
            f"{result['epoch']:<8} "
            f"{result['mAP']:<10.4f} "
            f"{result['bus_or_truck_ap']:<16.4f} "
            f"{result['sedan_ap']:<10.4f} "
            f"{result['iou']:<10.4f} "
            f"{result['f1']:<10.4f} "
            f"{result['TP']:<8} "
            f"{result['FP']:<8}"
        )
    
    print("="*110 + "\n")


def main():
    args = parse_args()
    device = select_evaluation_device(args.cuda, args.gpu_ids)
    cfg = DataConfig()
    checkpoint_paths = find_epoch_checkpoints(args.checkpoint_root, args.epoch_step)
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No epoch checkpoints found in {args.checkpoint_root}")

    print(f"Evaluation classes: {CLASS_NAMES}")
    print(f"Using evaluation device: {device}")

    _, validation_dataset, _, validation_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=CLASS_TO_IDX,
        ignore_unmapped_classes=True,
        split_mode=args.split_mode,
        split_dir=args.split_dir,
    )

    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty. Adjust --train-ratio or --limit-samples.")

    model = build_model(
        device=device,
        num_boxes=args.num_boxes,
        num_classes=NUM_CLASSES,
        model_type=args.model_type
    )

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
            num_classes=NUM_CLASSES,
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
        graph_metrics = metrics_for_graph(metrics, CLASS_NAMES)
        graph_metrics["epoch"] = epoch
        results.append(graph_metrics)
        print_evaluation_result(epoch, graph_metrics)

    print_results_table(results)

if __name__ == "__main__":
    main()
