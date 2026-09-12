"""Reproducible mixed-precision training for binary segmentation."""

from __future__ import annotations

import json
import platform
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from fabric_inspection.data.dataset import make_dataloaders
from fabric_inspection.models.unet import ResNet18UNet
from fabric_inspection.training.losses import BCEDiceLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric_threshold: float,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    progress_bar: bool = True,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_positive = false_positive = false_negative = 0
    progress = tqdm(
        loader,
        leave=False,
        desc="train" if training else "validation",
        disable=not progress_bar,
    )
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * images.shape[0]
        predictions = torch.sigmoid(logits.detach()) >= metric_threshold
        target_mask = targets >= 0.5
        true_positive += int((predictions & target_mask).sum())
        false_positive += int((predictions & ~target_mask).sum())
        false_negative += int((~predictions & target_mask).sum())
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice = 2 * true_positive / dice_denominator if dice_denominator else 1.0
    iou_denominator = true_positive + false_positive + false_negative
    iou = true_positive / iou_denominator if iou_denominator else 1.0
    return {"loss": total_loss / len(loader.dataset), "dice": dice, "iou": iou}


def _environment_metadata(device: torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if device.type == "cuda" else None,
    }


def _save_training_curves(history: list[dict[str, float | int]], output_path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="Validation")
    axes[0].set(title="Combined BCE + Dice loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(frame["epoch"], frame["train_dice"], label="Train")
    axes[1].plot(frame["epoch"], frame["validation_dice"], label="Validation")
    axes[1].set(title="Dice at threshold 0.5", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(config: dict[str, object], config_path: Path) -> dict[str, object]:
    seed = int(config["seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured Phase 2 training run")
    device = torch.device("cuda")
    data_config = config["data"]
    training_config = config["training"]
    output_dir = Path(config["output_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, dataset_counts = make_dataloaders(
        Path(data_config["patch_manifest"]),
        int(data_config["image_size"]),
        int(training_config["batch_size"]),
        int(training_config["num_workers"]),
        float(data_config["positive_sampling_fraction"]),
        seed,
    )
    model = ResNet18UNet(pretrained=bool(config["model"]["pretrained"])).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(training_config["bce_weight"]),
        dice_weight=float(training_config["dice_weight"]),
        positive_pixel_weight=float(training_config["positive_pixel_weight"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    use_amp = bool(training_config["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    best_path = checkpoint_dir / "best.pt"
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])

    for epoch in range(1, epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            float(training_config["metric_threshold"]),
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            progress_bar=bool(training_config["progress_bar"]),
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            float(training_config["metric_threshold"]),
            use_amp=use_amp,
            progress_bar=bool(training_config["progress_bar"]),
        )
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate_used,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        scheduler.step(validation_metrics["dice"])
        improved = validation_metrics["dice"] > best_dice + 1e-6 or (
            abs(validation_metrics["dice"] - best_dice) <= 1e-6
            and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_dice = validation_metrics["dice"]
            best_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "metric_threshold": float(training_config["metric_threshold"]),
                    "config": config,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_dice={validation_metrics['dice']:.4f}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    shutil.copy2(config_path, output_dir / "config.yaml")
    _save_training_curves(history, output_dir / "training_curves.png")
    summary = {
        "best_validation_dice": best_dice,
        "best_validation_loss": best_loss,
        "best_checkpoint": str(best_path),
        "epochs_completed": len(history),
        "dataset": dataset_counts,
        "environment": _environment_metadata(device),
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
