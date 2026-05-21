import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from dummy_dataloader import *
from dummy_dataset import KRadarMultiSequenceGTDetectionDataset, detection_collate
from dummy_module import MVRSS3DModel
from zxy_config import DataConfig


CLASS_NAMES = {
    0: "Sedan",
    1: "Bus or Truck",
    2: "Bicycle",
    3: "Motorcycle",
    4: "Pedestrian",
    5: "Pedestrian Group",
}
NUM_CLASSES = len(CLASS_NAMES)
MAX_DETECTIONS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ground-truth and predicted boxes on RA maps."
    )
    parser.add_argument("--checkpoint-path", default="/home/local/xinyu/MVRSS/mvrss/checkpoints/mvrss_detection/"
    "seq11_20260520_001311_361632/best_epoch_010_20260520_015504_mAP_0p0008.pth")
    parser.add_argument("--sequence", type=int, default=11)
    parser.add_argument("--start-file-idx", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-dir", default="ra_vis")
    return parser.parse_args()


def build_model(device, num_boxes=64, num_classes=NUM_CLASSES):
    model = MVRSS3DModel(
        d_in=64,
        e_in=37,
        num_boxes=num_boxes,
        box_dim=7,
        num_classes=num_classes,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(8, 8),
    ).to(device)
    return model

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_dataset(cfg):
    sequence_datasets = [
        build_detection_dataset_for_sequence(cfg, sequence)
        for sequence in get_config_sequences(cfg)
    ]
    return KRadarMultiSequenceGTDetectionDataset(
        sequence_datasets=sequence_datasets
    )


def make_ra_map(rae):
    if torch.is_tensor(rae):
        rae = rae.detach().cpu().numpy()

    ra_map = np.mean(rae, axis=2)
    ra_map = np.abs(ra_map)
    ra_map = np.log1p(ra_map)
    return ra_map


def normalized_boxes_to_raw_rae(boxes, rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))

    r_size, a_size, e_size = rae_shape
    raw = boxes.clone()
    raw[:, 0] = raw[:, 0] * r_size
    raw[:, 1] = raw[:, 1] * a_size
    raw[:, 2] = raw[:, 2] * e_size
    raw[:, 3] = raw[:, 3] * r_size
    raw[:, 4] = raw[:, 4] * a_size
    raw[:, 5] = raw[:, 5] * e_size
    return raw


def filter_predictions(outputs, rae_shape, score_thresh, max_detections):
    pred_boxes_norm = outputs["box_pred"].squeeze(0).sigmoid()
    pred_logits = outputs["cls_pred"].squeeze(0)
    pred_probs = pred_logits.softmax(dim=-1)

    foreground_probs = pred_probs[:, :NUM_CLASSES]
    background_probs = pred_probs[:, NUM_CLASSES]
    pred_scores, pred_labels = foreground_probs.max(dim=-1)

    keep = (pred_scores > score_thresh) & (pred_scores > background_probs)
    pred_boxes_norm = pred_boxes_norm[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    if pred_scores.shape[0] > max_detections:
        pred_scores, topk_indices = pred_scores.topk(max_detections)
        pred_boxes_norm = pred_boxes_norm[topk_indices]
        pred_labels = pred_labels[topk_indices]

    pred_boxes_raw = normalized_boxes_to_raw_rae(pred_boxes_norm, rae_shape)
    return pred_boxes_raw.cpu(), pred_labels.cpu(), pred_scores.cpu()


def draw_boxes(ax, boxes, labels=None, scores=None, color="lime", prefix="GT"):
    for i, box in enumerate(boxes):
        r_idx = float(box[0])
        a_idx = float(box[1])
        r_width = float(box[3])
        a_width = float(box[4])

        a_min = a_idx - a_width / 2.0
        r_min = r_idx - r_width / 2.0

        rect = Rectangle(
            (a_min, r_min),
            a_width,
            r_width,
            linewidth=1.8,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        text = prefix
        if labels is not None:
            label_id = int(labels[i])
            text += f" {CLASS_NAMES.get(label_id, label_id)}"
        if scores is not None:
            text += f" {float(scores[i]):.2f}"

        ax.text(
            a_min,
            max(r_min - 2.0, 0.0),
            text,
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
        )


@torch.no_grad()
def get_frame_prediction(model, prepare_model_inputs, dataset, file_idx, device, score_thresh, max_detections):
    item = dataset[file_idx]
    batch = detection_collate([item])

    rad, rae = prepare_model_inputs(batch, device)
    outputs = model(rad, rae)

    rae_shape = tuple(item["rae"].shape)
    pred_boxes, pred_labels, pred_scores = filter_predictions(
        outputs=outputs,
        rae_shape=rae_shape,
        score_thresh=score_thresh,
        max_detections=max_detections,
    )

    return {
        "item": item,
        "rae_shape": rae_shape,
        "ra_map": make_ra_map(item["rae"]),
        "gt_boxes": item["gt_boxes_raw"].cpu(),
        "gt_labels": item["gt_labels"].cpu(),
        "pred_boxes": pred_boxes,
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
    }


def show_frame(ax, frame_data):
    item = frame_data["item"]
    r_size, a_size, _ = frame_data["rae_shape"]

    ax.clear()
    ax.imshow(frame_data["ra_map"], origin="lower", aspect="auto", cmap="viridis")

    draw_boxes(
        ax,
        frame_data["gt_boxes"],
        labels=frame_data["gt_labels"],
        color="lime",
        prefix="GT",
    )
    draw_boxes(
        ax,
        frame_data["pred_boxes"],
        labels=frame_data["pred_labels"],
        scores=frame_data["pred_scores"],
        color="red",
        prefix="Pred",
    )

    ax.set_title(
        f"RA map | sequence={item['sequence']} | file_idx={item['file_idx']} | "
        f"gt_frame_idx={item['gt_frame_idx']} | "
        f"GT={len(frame_data['gt_boxes'])} | Pred={len(frame_data['pred_boxes'])}"
    )
    ax.set_xlabel("Azimuth bin")
    ax.set_ylabel("Range bin")
    ax.set_xlim(0, a_size - 1)
    ax.set_ylim(0, r_size - 1)


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint_path

    cfg = DataConfig()
    cfg.sequence = args.sequence
    cfg.sequences = (args.sequence,)

    device = select_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = build_dataset(cfg)
    num_frames = len(dataset)
    print(
        f"dataset_frames={num_frames} sequences={get_config_sequences(cfg)} "
        f"device={device}"
    )

    model = build_model(device, num_classes=NUM_CLASSES)
    model = load_checkpoint(model, checkpoint_path, device)

    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)

    start_file_idx = args.start_file_idx
    if start_file_idx < 0 or start_file_idx >= num_frames:
        raise ValueError(
            f"--start-file-idx must be in [0, {num_frames - 1}], got {start_file_idx}"
        )
    if args.frame_step <= 0:
        raise ValueError(f"--frame-step must be greater than 0, got {args.frame_step}")
    if args.max_frames < 0:
        raise ValueError(f"--max-frames must be >= 0, got {args.max_frames}")

    rendered_count = 0
    for file_idx in range(start_file_idx, num_frames, args.frame_step):
        frame_data = get_frame_prediction(
            model=model,
            prepare_model_inputs=prepare_model_inputs,
            dataset=dataset,
            file_idx=file_idx,
            device=device,
            score_thresh=args.score_thresh,
            max_detections=MAX_DETECTIONS,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        show_frame(ax, frame_data)
        fig.tight_layout()

        item = frame_data["item"]
        print(
            f"sequence={item['sequence']} "
            f"file_idx={item['file_idx']} "
            f"gt_frame_idx={item['gt_frame_idx']} "
            f"GT={len(frame_data['gt_boxes'])} "
            f"Pred={len(frame_data['pred_boxes'])}"
        )

        if args.save_images:
            output_path = os.path.join(args.save_dir, f"ra_map_gt_pred_file_{file_idx:05d}.png")
            fig.savefig(output_path, dpi=160)
            print(f"saved={output_path}")

        print("Close the matplotlib window to continue.")
        plt.show()

        plt.close(fig)
        rendered_count += 1

        if args.max_frames > 0 and rendered_count >= args.max_frames:
            break

    print(f"rendered_frames={rendered_count}")


if __name__ == "__main__":
    main()
