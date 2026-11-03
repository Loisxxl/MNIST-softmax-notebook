#!/usr/bin/env python3
"""Train and evaluate a PyTorch softmax-regression baseline on MNIST.

Run from this directory in a terminal with:
    python mnist_softmax.py --epochs 10 --batch-size 256 --learning-rate 0.1

"""
from __future__ import annotations

import sys
import torch

print(sys.executable)
print(torch.__version__)

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SoftmaxRegression(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images.flatten(start_dim=1))


from types import SimpleNamespace

args = SimpleNamespace(
    data_dir=Path("data"),
    output_dir=Path("outputs"),
    epochs=10,
    batch_size=256,
    learning_rate=0.1,
    seed=42,
    threads=4,
    num_workers=0,
)
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(args.seed)
print(f"随机种子已设置为 {args.seed}")
def save_dataset_visualization(train_dataset, counts, output_path):
    targets = train_dataset.targets
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(3, 5, height_ratios=(1, 1, 1.05))

    # 每个数字显示一张图片
    for digit in range(10):
        first_index = int(torch.nonzero(targets == digit, as_tuple=False)[0, 0])
        image, label = train_dataset[first_index]
        ax = fig.add_subplot(grid[digit // 5, digit % 5])
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Label: {int(label)}")
        ax.axis("off")

    # 下方画出各数字的图片数量
    bar_ax = fig.add_subplot(grid[2, :])
    bar_ax.bar(np.arange(10), counts, color="#4C78A8")
    bar_ax.set_xticks(np.arange(10))
    bar_ax.set_xlabel("Digit label")
    bar_ax.set_ylabel("Number of images")
    bar_ax.set_title("Training-set class distribution")
    bar_ax.grid(axis="y", alpha=0.25)

    fig.suptitle("MNIST: sample 28 x 28 grayscale images and class counts")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=str(args.data_dir),
    train=True,
    transform=transform,
    download=True
)

train_counts = np.bincount(
    train_dataset.targets.numpy(),
    minlength=10
)

args.output_dir.mkdir(parents=True, exist_ok=True)
dataset_figure = args.output_dir / "mnist_dataset_visualization.png"

save_dataset_visualization(
    train_dataset,
    train_counts,
    dataset_figure
)
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_count = labels.size(0)
        loss_sum += loss.item() * batch_count
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_count

    return loss_sum / total, correct / total

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_confidences: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
    return (
        loss_sum / len(loader.dataset),
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_confidences),
    )

def save_loss_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", linewidth=2, color="#1f77b4",
            label="Training cross-entropy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss")
    ax.set_title("MNIST softmax-regression training loss")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

def save_prediction_visualization(test_dataset: datasets.MNIST,
                                  indices: np.ndarray,
                                  labels: np.ndarray,
                                  predictions: np.ndarray,
                                  confidences: np.ndarray,
                                  output_path: Path) -> list[dict[str, object]]:
    """Plot a reproducible sample; green titles are correct, red are errors."""
    fig, axes = plt.subplots(4, 5, figsize=(13, 10), constrained_layout=True)
    samples: list[dict[str, object]] = []
    for ax, index in zip(axes.flat, indices):
        image, _ = test_dataset[int(index)]
        true_label = int(labels[index])
        predicted_label = int(predictions[index])
        confidence = float(confidences[index])
        correct = true_label == predicted_label
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"True: {true_label} | Pred: {predicted_label}\nConfidence: {confidence:.1%}",
            color=("#16823B" if correct else "#C62828"), fontsize=9,
        )
        ax.axis("off")
        samples.append({
            "test_index": int(index),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": correct,
        })
    fig.suptitle("MNIST test-set predictions (green = correct, red = incorrect)",
                 fontsize=14)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return samples

def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Image count")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("MNIST test-set confusion matrix")
    threshold = matrix.max() * 0.55
    for row in range(10):
        for col in range(10):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

seed_everything(args.seed)
torch.set_num_threads(args.threads)

args.data_dir.mkdir(parents=True, exist_ok=True)
args.output_dir.mkdir(parents=True, exist_ok=True)

data_dir = args.data_dir.resolve()
output_dir = args.output_dir.resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=str(data_dir), train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=str(data_dir), train=False, transform=transform, download=True
)

print("训练图片数：", len(train_dataset))
print("测试图片数：", len(test_dataset))
print("运行设备：", device)

generator = torch.Generator()
generator.manual_seed(args.seed)

loader_options = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "pin_memory": device.type == "cuda",
}

train_loader = DataLoader(
    train_dataset, shuffle=True, generator=generator, **loader_options
)
test_loader = DataLoader(
    test_dataset, shuffle=False, **loader_options
)

model = SoftmaxRegression().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

history = []
output_dir = args.output_dir.resolve()
output_dir.mkdir(parents=True, exist_ok=True)

import torchvision
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("设备:", device)
print("训练图片:", len(train_dataset), "| 测试图片:", len(test_dataset))
print("图片形状:", tuple(train_dataset[0][0].shape))
print("模型:", model)
print("模型参数量:", sum(p.numel() for p in model.parameters()))
print("数据批次大小：", args.batch_size)
print("SGD 学习率:", args.learning_rate)

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()

    train_loss, train_accuracy = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    elapsed = time.perf_counter() - start_time
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "seconds": elapsed,
    })

    print(
        f"第 {epoch}/{args.epochs} 轮 | "
        f"训练损失={train_loss:.4f} | "
        f"训练准确率={train_accuracy:.2%} | "
        f"用时={elapsed:.1f} 秒"
    )

loss_figure = output_dir / "training_loss.png"
save_loss_curve(history, loss_figure)
test_loss, y_true, y_pred, confidences = evaluate(
    model, test_loader, criterion, device
)

test_correct = int(np.sum(y_true == y_pred))
test_accuracy = test_correct / len(y_true)

confusion = np.zeros((10, 10), dtype=np.int64)
np.add.at(confusion, (y_true, y_pred), 1)

class_totals = np.bincount(y_true, minlength=10)
class_correct = np.bincount(y_true[y_true == y_pred], minlength=10)
per_class_accuracy = class_correct / class_totals

print(f"测试损失：{test_loss:.4f}")
print(f"测试准确率：{test_accuracy:.2%} ({test_correct}/{len(y_true)})")

confusion_figure = output_dir / "confusion_matrix.png"
save_confusion_matrix(confusion, confusion_figure)

plt.show()

sample_indices = np.sort(
    np.random.default_rng(args.seed).choice(
        len(y_true), size=20, replace=False
    )
)

prediction_figure = output_dir / "sample_predictions.png"
samples = save_prediction_visualization(
    test_dataset, sample_indices, y_true, y_pred,
    confidences, prediction_figure
)

plt.show()  # 显示测试样例预测图

print("测试索引 | 真实数字 | 预测数字 | 置信度 | 结果")
for row in samples:
    result = "正确" if row["correct"] else "错误"
    print(
        f'{row["test_index"]:>4} | {row["true_label"]} | '
        f'{row["predicted_label"]} | '
        f'{row["confidence"]:.2%} | {result}'
    )

