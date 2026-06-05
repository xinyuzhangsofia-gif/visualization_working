import argparse
import os

import torch

from dummy_dataloader import build_train_val_dataloaders, get_config_sequences, prepare_model_inputs
from dummy_dataset import CLASS_NAMES, CLASS_TO_IDX
from dummy_evaluation import evaluate_train_val_iou
from dummy_module import (
    MVRSS3DModel,
    MVRSS3DModelDeform,
    MVRSS3DModelDeformDepthwiseSeparable,
)
from dummy_module_multiscale import MVRSS3DModel2
from dummy_train import NUM_CLASSES, parse_gpu_ids, train_one_epoch, validate_loss
from utils_dummy.checkpoints import (
    create_checkpoint_run_dirs,
    get_model_state_dict_from_checkpoint,
    save_replacing_named_checkpoint_copy,
)
from utils_dummy.logging_utils import (
    create_tensorboard_writer,
    write_tensorboard_metrics,
    write_tensorboard_run_config,
)
from utils_dummy.other_helping_dunctions import (
    BestCheckpointState,
    append_training_history,
    build_epoch_eval_metrics,
    save_epoch_and_update_best_checkpoint,
    save_global_best_checkpoint,
    set_seed,
)
from zxy_config import DataConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Resume dummy MVRSS training from a checkpoint.")
    parser.add_argument(
        "--resume-checkpoint",
        default=(
            "checkpoints/mvrss_detection/seq1-11_20260527_162350_686812/candidate_epoch_100_20260528_080639_mAP_0p4397.pth"
            
        ),
        help="Checkpoint to load model and optimizer state from."
    )
    parser.add_argument(
        "--initial-best-checkpoint",
        default=(
            "checkpoints/mvrss_detection/seq1-11_20260527_162350_686812/global_best_epoch_100_20260528_080639_mAP_0p4397.pth"
            
        ),
        help="Existing best checkpoint to keep until a resumed epoch gets better mAP."
    )
    parser.add_argument("--start-epoch", type=int, default=101)
    parser.add_argument("--end-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--background-weight", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="random", choices=["random", "file"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--checkpoint-epoch-step", type=int, default=4)
    parser.add_argument("--checkpoint-base-dir", default="checkpoints")
    parser.add_argument("--log-base-dir", default="runs")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--model-type", default="model3", choices=["model1", "model2", "model3", "model4", "model5"])
    parser.add_argument("--no-load-optimizer", action="store_true")
    return parser.parse_args()


def select_device_and_gpus(gpu_ids_text):
    gpu_ids = parse_gpu_ids(gpu_ids_text)
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
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    return torch.device("cpu"), []


def build_model(device, args):
    if args.model_type == "model1":
        return MVRSS3DModel(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(16, 16),
        ).to(device)

    if args.model_type == "model2":
        return MVRSS3DModel2(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(8, 8),
        ).to(device)

    if args.model_type == "model4":
        return MVRSS3DModelDeform(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(4, 4),
        ).to(device)

    if args.model_type == "model5":
        return MVRSS3DModelDeformDepthwiseSeparable(
            d_in=64,
            e_in=37,
            num_boxes=args.num_boxes,
            box_dim=7,
            num_classes=NUM_CLASSES,
            feature_channels=64,
            fusion_hidden_channels=64,
            decoder_hidden_channels=128,
            pooled_size=(4, 4),
        ).to(device)

    return MVRSS3DModelDeform(
        d_in=64,
        e_in=37,
        num_boxes=args.num_boxes,
        box_dim=7,
        num_classes=NUM_CLASSES,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(8, 8),
    ).to(device)


def load_resume_checkpoint(model, optimizer, checkpoint_path, device, load_optimizer=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_for_state_dict = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_for_state_dict.load_state_dict(get_model_state_dict_from_checkpoint(checkpoint))

    optimizer_loaded = False
    if (
        load_optimizer
        and isinstance(checkpoint, dict)
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        optimizer_loaded = True

    checkpoint_epoch = None
    if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None:
        checkpoint_epoch = int(checkpoint["epoch"])

    return checkpoint_epoch, optimizer_loaded


def initialize_best_state(best_state, initial_best_checkpoint, checkpoint_dir):
    if initial_best_checkpoint is None or initial_best_checkpoint == "":
        return None
    if not os.path.exists(initial_best_checkpoint):
        raise FileNotFoundError(f"Initial best checkpoint not found: {initial_best_checkpoint}")

    checkpoint = torch.load(initial_best_checkpoint, map_location="cpu")
    best_epoch = int(checkpoint.get("epoch", 0))
    best_map = float(checkpoint.get("mAP", checkpoint.get("val_metrics", {}).get("mAP", -1.0)))
    train_metrics = checkpoint.get("train_metrics", {})
    val_metrics = checkpoint.get("val_metrics", {"mAP": best_map})
    f1 = float(checkpoint.get("f1", 0.0))

    global_best_path = save_replacing_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=initial_best_checkpoint,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix="global_best",
    )
    best_state.update(
        epoch=best_epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        f1=f1,
        checkpoint_path=initial_best_checkpoint,
        global_best_path=global_best_path,
    )
    return global_best_path


def main():
    args = parse_args()
    if args.start_epoch > args.end_epoch:
        raise ValueError("--start-epoch must be <= --end-epoch")
    if args.checkpoint_epoch_step <= 0:
        raise ValueError("--checkpoint-epoch-step must be greater than 0")

    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_config_sequences(cfg)
    args.epochs = args.end_epoch
    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES.copy()
    args.class_to_idx = CLASS_TO_IDX.copy()

    print(f"Training classes: {CLASS_NAMES}")
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"Initial best checkpoint: {args.initial_best_checkpoint}")
    print(f"Resume training epochs: {args.start_epoch}-{args.end_epoch}")

    device, gpu_ids = select_device_and_gpus(args.gpu_ids)

    train_dataset, val_dataset, train_loader, val_loader = build_train_val_dataloaders(
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

    model = build_model(device=device, args=args)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0],
        )
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_epoch, optimizer_loaded = load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=args.resume_checkpoint,
        device=device,
        load_optimizer=not args.no_load_optimizer,
    )
    print(
        f"Loaded checkpoint epoch={checkpoint_epoch}, "
        f"optimizer_loaded={optimizer_loaded}"
    )

    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name="mvrss_detection_resume",
        sequences=configured_sequences,
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]
    print(f"Saving checkpoints to: {checkpoint_dir}")

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name="mvrss_detection_resume",
        sequence=configured_sequences,
    )
    write_tensorboard_run_config(
        writer=writer,
        cfg=cfg,
        num_epochs=args.end_epoch,
        batch_size=args.batch_size,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        learning_rate=args.lr,
        num_boxes=args.num_boxes,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        background_weight=args.background_weight,
        eval_iou_thresh=args.eval_iou_thresh,
    )

    history = []
    best_state = BestCheckpointState()
    initial_best_path = initialize_best_state(
        best_state=best_state,
        initial_best_checkpoint=args.initial_best_checkpoint,
        checkpoint_dir=checkpoint_dir,
    )
    if initial_best_path is not None:
        print(f"Copied initial global best to: {initial_best_path}")

    for epoch_number in range(args.start_epoch, args.end_epoch + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch_number - 1,
            num_epochs=args.end_epoch,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            background_weight=args.background_weight,
        )

        val_loss_metrics = validate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            background_weight=args.background_weight,
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
            max_detections=args.num_boxes,
        )
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics,
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer,
            epoch=epoch_number,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
        )

        checkpoint_path = save_epoch_and_update_best_checkpoint(
            best_state=best_state,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            args=args,
            cfg=cfg,
            epoch=epoch_number,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
            total_epochs=args.end_epoch,
            checkpoint_epoch_step=args.checkpoint_epoch_step,
        )
        if checkpoint_path is not None:
            print(f"Saved candidate checkpoint: {checkpoint_path}")

        append_training_history(
            history=history,
            epoch=epoch_number,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
        )

    writer.close()
    global_best_path, _ = save_global_best_checkpoint(
        best_state=best_state,
        checkpoint_dirs=checkpoint_dirs,
        checkpoint_key=checkpoint_key,
    )
    if global_best_path is not None:
        print(f"Current global best checkpoint: {global_best_path}")


if __name__ == "__main__":
    main()
