import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from typing import Tuple

sys.path.insert(0, os.path.dirname(__file__))
import stl10_input
from results_logger import log_result

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR          = "./data/stl10_binary"
TRAIN_X_PATH      = os.path.join(DATA_DIR, "train_X.bin")
TRAIN_Y_PATH      = os.path.join(DATA_DIR, "train_y.bin")
TEST_X_PATH       = os.path.join(DATA_DIR, "test_X.bin")
TEST_Y_PATH       = os.path.join(DATA_DIR, "test_y.bin")
UNLABELED_X_PATH  = os.path.join(DATA_DIR, "unlabeled_X.bin")

NUM_CLASSES     = 10
BATCH_SIZE      = 64
NUM_EPOCHS      = 10
LEARNING_RATE   = 1e-3
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane", "bird", "car", "deer", "dog",
    "horse", "monkey", "ship", "truck", "frog"
]

# ImageNet normalization expected by pretrained torchvision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class STL10Dataset(Dataset):
    """
    Wraps STL-10 binary data loaded via stl10_input into a PyTorch Dataset.
    Images are read as (N, 96, 96, 3) uint8 HWC arrays and converted on-the-fly.
    Labels are 1-indexed in the binary file; we shift them to 0-indexed here.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None) -> None:
        # images: (N, H, W, C) uint8
        self.images    = images
        self.labels    = labels.astype(np.int64) - 1  # shift 1-10 → 0-9
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]           # (96, 96, 3) uint8

        # Convert to PIL-compatible tensor path: HWC uint8 → CHW float [0,1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, int(self.labels[idx])


def build_transforms(train: bool) -> transforms.Compose:
    """Return augmentation pipeline for train/val."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(224),
        normalize,
    ])


class STL10UnlabeledDataset(Dataset):
    """
    Wraps the STL-10 unlabeled binary (unlabeled_X.bin) which has no labels.
    Returns only image tensors.
    """

    def __init__(self, images: np.ndarray, transform=None) -> None:
        self.images    = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.images[idx]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor


def load_dataset(
    images_path: str,
    labels_path: str,
    train: bool,
    batch_size: int,
) -> DataLoader:
    images = stl10_input.read_all_images(images_path)   # (N, 96, 96, 3)
    labels = stl10_input.read_labels(labels_path)       # (N,)
    dataset = STL10Dataset(images, labels, transform=build_transforms(train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)


def load_unlabeled_dataset(images_path: str, batch_size: int) -> DataLoader:
    """Load the 100 000 unlabeled STL-10 images (no labels available)."""
    images = stl10_input.read_all_images(images_path)   # (100000, 96, 96, 3)
    dataset = STL10UnlabeledDataset(images, transform=build_transforms(train=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_resnet50(num_classes: int, feature_extract: bool = False) -> nn.Module:
    """ResNet-50 pretrained on ImageNet with replaced classifier head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vgg16(num_classes: int, feature_extract: bool = False) -> nn.Module:
    """VGG-16 pretrained on ImageNet with replaced classifier head."""
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if feature_extract:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def infer_unlabeled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference on unlabeled images and return predicted class indices."""
    model.eval()
    all_preds: list[int] = []
    for images in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
    return np.array(all_preds, dtype=np.int64)


def run_experiment(
    model: nn.Module,
    model_name: str,
    frozen: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train and evaluate one transfer-learning experiment. Returns final test accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    mode_label = "frozen backbone" if frozen else "full fine-tune"
    print(f"\n{'='*60}")
    print(f"  Model  : {model_name}  [{mode_label}]")
    print(f"  LR     : {lr}  |  Device: {device}")
    print(f"{'='*60}")

    test_acc    = 0.0
    train_acc   = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch [{epoch:>2}/{num_epochs}]  "
            f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%  |  "
            f"Test  loss: {test_loss:.4f}  acc: {test_acc*100:.2f}%"
        )

    print(f"\nFinal test accuracy — {model_name} [{mode_label}]: {test_acc*100:.2f}%")
    save_name = f"{model_name.lower().replace('-', '_')}_{'frozen' if frozen else 'full'}_stl10.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Model saved to {save_name}")

    log_result(
        stage="B",
        architecture=model_name,
        mode=mode_label,
        optimizer="Adam",
        lr=lr,
        epochs_trained=num_epochs,
        train_acc=train_acc,
        test_acc=test_acc,
        notes=f"Pretrained ImageNet weights, StepLR(step=5, gamma=0.1)",
    )
    return test_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    stl10_input.download_and_extract()

    print(f"Using device: {DEVICE}")
    print("Loading datasets...")
    train_loader = load_dataset(TRAIN_X_PATH, TRAIN_Y_PATH, train=True,  batch_size=BATCH_SIZE)
    test_loader  = load_dataset(TEST_X_PATH,  TEST_Y_PATH,  train=False, batch_size=BATCH_SIZE)
    print(
        f"Train: {len(train_loader.dataset)} labeled images  |  "
        f"Test:  {len(test_loader.dataset)} labeled images"
    )

    # Each entry: (display name, builder fn, frozen, lr)
    # Frozen backbone  → higher LR (only head is trained)
    # Full fine-tune   → lower LR (all weights updated)
    experiments = [
        ("ResNet-50", build_resnet50, True,  1e-3),
        ("ResNet-50", build_resnet50, False, 1e-4),
        ("VGG-16",    build_vgg16,    True,  1e-3),
        ("VGG-16",    build_vgg16,    False, 1e-4),
    ]

    results: list[tuple[str, str, float]] = []
    for model_name, builder, frozen, lr in experiments:
        model = builder(NUM_CLASSES, feature_extract=frozen)
        acc   = run_experiment(
            model, model_name, frozen,
            train_loader, test_loader,
            NUM_EPOCHS, lr, DEVICE,
        )
        results.append((model_name, "frozen" if frozen else "full fine-tune", acc))

    print(f"\n{'='*60}")
    print("  Stage B — Summary")
    print(f"{'='*60}")
    for name, mode, acc in results:
        print(f"  {name:<12} [{mode:<15}]  test acc: {acc*100:.2f}%")
    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
