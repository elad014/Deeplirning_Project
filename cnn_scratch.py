import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple

sys.path.insert(0, os.path.dirname(__file__))
import stl10_input
from results_logger import log_result

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR         = "./data/stl10_binary"
TRAIN_X_PATH     = os.path.join(DATA_DIR, "train_X.bin")
TRAIN_Y_PATH     = os.path.join(DATA_DIR, "train_y.bin")
TEST_X_PATH      = os.path.join(DATA_DIR, "test_X.bin")
TEST_Y_PATH      = os.path.join(DATA_DIR, "test_y.bin")

NUM_CLASSES  = 10
BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LR           = 1e-2          # SGD works better with a higher LR
WEIGHT_DECAY = 1e-4
PATIENCE     = 10            # early-stopping patience
MIN_DELTA    = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP      = DEVICE.type == "cuda"

# STL-10 channel statistics (computed over the training set)
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD  = [0.2603, 0.2565, 0.2712]

CLASS_NAMES = [
    "airplane", "bird", "car", "deer", "dog",
    "horse", "monkey", "ship", "truck", "frog"
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class STL10Dataset(Dataset):
    """STL-10 binary data as a PyTorch Dataset (96x96 native resolution)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None) -> None:
        self.images    = images
        self.labels    = labels.astype(np.int64) - 1  # 1-10 → 0-9
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_tensor = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, int(self.labels[idx])


def build_transforms(train: bool) -> transforms.Compose:
    """Augmentation pipeline for 96x96 STL-10 images (no resize — training from scratch)."""
    normalize = transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(96, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            normalize,
        ])
    return transforms.Compose([normalize])


def load_dataset(
    images_path: str,
    labels_path: str,
    train: bool,
    batch_size: int,
) -> DataLoader:
    images  = stl10_input.read_all_images(images_path)
    labels  = stl10_input.read_labels(labels_path)
    dataset = STL10Dataset(images, labels, transform=build_transforms(train))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=2, pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Model  (VGG_SmallSigmoid — best architecture from HW1, adapted for 96x96)
# ---------------------------------------------------------------------------

def build_vgg_backbone() -> nn.Sequential:
    """
    4-block VGG convolutional backbone with BatchNorm.
    AdaptiveAvgPool2d(1) at the end makes it input-size agnostic,
    so it works for both 32x32 (CIFAR-10) and 96x96 (STL-10).
    Output: (N, 512, 1, 1)
    """
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                      # 96 → 48

        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                                      # 48 → 24

        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(2),                                      # 24 → 12

        nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),                              # → 512×1×1
    )


class VGGSmallSigmoid(nn.Module):
    """
    VGG_SmallSigmoid — the best-performing model from HW1 (CIFAR-10),
    ported to STL-10 (96x96).
    Classifier head: 512 → 64 (Sigmoid) → 32 (ReLU+BN) → 10
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = build_vgg_backbone()

        # layer 1: Sigmoid, xavier init, no BN, dropout 0.1
        self.fc1     = nn.Linear(512, 64)
        self.drop1   = nn.Dropout(0.1)

        # layer 2: ReLU, he init, BN, dropout 0.2
        self.fc2     = nn.Linear(64, 32)
        self.bn2     = nn.BatchNorm1d(32)
        self.drop2   = nn.Dropout(0.2)

        # final classifier
        self.classifier = nn.Linear(32, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).view(x.size(0), -1)   # (N, 512)
        x = self.drop1(torch.sigmoid(self.fc1(x)))  # (N, 64)
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))  # (N, 32)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    stl10_input.download_and_extract()

    print(f"Using device: {DEVICE}  |  AMP: {USE_AMP}")
    print("Loading datasets...")
    train_loader = load_dataset(TRAIN_X_PATH, TRAIN_Y_PATH, train=True,  batch_size=BATCH_SIZE)
    test_loader  = load_dataset(TEST_X_PATH,  TEST_Y_PATH,  train=False, batch_size=BATCH_SIZE)
    print(
        f"Train: {len(train_loader.dataset)} labeled images  |  "
        f"Test: {len(test_loader.dataset)} labeled images"
    )

    model     = VGGSmallSigmoid(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9,
        nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler        = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    print(f"\n{'='*60}")
    print("  Stage A — CNN from Scratch  (VGG_SmallSigmoid)")
    print(f"  Epochs: {NUM_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    print(f"{'='*60}")

    best_test_acc  = 0.0
    final_train_acc = 0.0
    epochs_trained  = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        epochs_trained  = epoch
        final_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "cnn_scratch_best.pth")

        print(
            f"Epoch [{epoch:>2}/{NUM_EPOCHS}]  "
            f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%  |  "
            f"Test  loss: {test_loss:.4f}  acc: {test_acc*100:.2f}%"
        )

        if early_stop(test_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nBest test accuracy (Stage A): {best_test_acc*100:.2f}%")
    print("Model saved to cnn_scratch_best.pth")

    log_result(
        stage="A",
        architecture="VGG_SmallSigmoid",
        mode="scratch",
        optimizer="SGD",
        lr=LR,
        epochs_trained=epochs_trained,
        train_acc=final_train_acc,
        test_acc=best_test_acc,
        notes="4-block VGG backbone + Sigmoid head, CosineAnnealingLR, EarlyStopping",
    )


if __name__ == "__main__":
    main()
