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
UNLABELED_X_PATH = os.path.join(DATA_DIR, "unlabeled_X.bin")

NUM_CLASSES    = 10
BATCH_SIZE     = 64
AE_EPOCHS      = 30          # autoencoder pretraining epochs
CLS_EPOCHS     = 20          # classifier fine-tuning epochs
AE_LR          = 1e-3        # autoencoder Adam LR
CLS_LR_FROZEN  = 1e-3        # classifier LR — frozen encoder
CLS_LR_FULL    = 1e-4        # classifier LR — full fine-tune
WEIGHT_DECAY   = 1e-4
PATIENCE       = 7
MIN_DELTA      = 1e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP        = DEVICE.type == "cuda"

STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD  = [0.2603, 0.2565, 0.2712]

CLASS_NAMES = [
    "airplane", "bird", "car", "deer", "dog",
    "horse", "monkey", "ship", "truck", "frog"
]

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class STL10Dataset(Dataset):
    """Labeled STL-10 images (train / test)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None) -> None:
        self.images    = images
        self.labels    = labels.astype(np.int64) - 1   # 1-10 → 0-9
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


class STL10UnlabeledDataset(Dataset):
    """100 000 unlabeled STL-10 images — used for autoencoder pretraining."""

    def __init__(self, images: np.ndarray, transform=None) -> None:
        self.images    = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)


def build_labeled_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(96, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            _normalize(),
        ])
    return transforms.Compose([_normalize()])


def build_unlabeled_transforms() -> transforms.Compose:
    """Light augmentation for autoencoder pretraining."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        _normalize(),
    ])


def load_labeled(images_path: str, labels_path: str, train: bool, batch_size: int) -> DataLoader:
    images  = stl10_input.read_all_images(images_path)
    labels  = stl10_input.read_labels(labels_path)
    dataset = STL10Dataset(images, labels, transform=build_labeled_transforms(train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)


def load_unlabeled(images_path: str, batch_size: int) -> DataLoader:
    images  = stl10_input.read_all_images(images_path)
    dataset = STL10UnlabeledDataset(images, transform=build_unlabeled_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """
    Convolutional encoder: 96x96x3 → 256x6x6 latent feature map.
    stride-2 convolutions halve the spatial size at each block.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 96 → 48
            nn.Conv2d(3,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            # 48 → 24
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            # 24 → 12
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            # 12 → 6
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)      # (N, 256, 6, 6)


class ConvDecoder(nn.Module):
    """
    Mirror of ConvEncoder using ConvTranspose2d: 256x6x6 → 3x96x96.
    Sigmoid output keeps pixel values in [0, 1] before normalization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 6 → 12
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            # 12 → 24
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            # 24 → 48
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            # 48 → 96
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)      # (N, 3, 96, 96)


class ConvAutoencoder(nn.Module):
    """Full autoencoder: encoder + decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class EncoderClassifier(nn.Module):
    """
    Classification model built on top of a pretrained ConvEncoder.
    Head: GlobalAvgPool → Linear(256→128, ReLU) → Linear(128→num_classes)
    """

    def __init__(self, encoder: ConvEncoder, num_classes: int, freeze_encoder: bool) -> None:
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (N, 256, 6, 6) → (N, 256, 1, 1)
            nn.Flatten(),               # (N, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)      # (N, 256, 6, 6)
        return self.head(features)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
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
# Stage 1 — Autoencoder pretraining
# ---------------------------------------------------------------------------

def train_autoencoder_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for images in loader:
        images = images.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total      += images.size(0)
    return total_loss / total


@torch.no_grad()
def eval_autoencoder_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    for images in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            loss = criterion(model(images), images)
        total_loss += loss.item() * images.size(0)
        total      += images.size(0)
    return total_loss / total


def pretrain_autoencoder(
    autoencoder: ConvAutoencoder,
    unlabeled_loader: DataLoader,
    device: torch.device,
) -> None:
    autoencoder = autoencoder.to(device)
    criterion   = nn.MSELoss()
    optimizer   = optim.Adam(autoencoder.parameters(), lr=AE_LR, weight_decay=WEIGHT_DECAY)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AE_EPOCHS)
    scaler      = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stop  = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    print(f"\n{'='*60}")
    print("  Stage C — Step 1: Autoencoder pretraining")
    print(f"  Unlabeled images: {len(unlabeled_loader.dataset)}  |  Device: {device}")
    print(f"{'='*60}")

    final_loss     = 0.0
    epochs_trained = 0
    for epoch in range(1, AE_EPOCHS + 1):
        train_loss     = train_autoencoder_epoch(
            autoencoder, unlabeled_loader, criterion, optimizer, scaler, device
        )
        scheduler.step()
        final_loss     = train_loss
        epochs_trained = epoch
        print(f"Epoch [{epoch:>2}/{AE_EPOCHS}]  Recon loss: {train_loss:.6f}")

        if early_stop(train_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    torch.save(autoencoder.state_dict(), "autoencoder_pretrained.pth")
    print("Autoencoder saved to autoencoder_pretrained.pth")

    log_result(
        stage="C",
        architecture="ConvAutoencoder",
        mode="pretrain (unsupervised)",
        optimizer="Adam",
        lr=AE_LR,
        epochs_trained=epochs_trained,
        train_acc=None,
        test_acc=None,
        notes=f"MSE reconstruction on 100k unlabeled images. Final recon loss: {final_loss:.6f}",
    )


# ---------------------------------------------------------------------------
# Stage 2 — Classifier fine-tuning
# ---------------------------------------------------------------------------

def train_classifier_epoch(
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
def eval_classifier(
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


def finetune_classifier(
    encoder: ConvEncoder,
    freeze_encoder: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Fine-tune a classification head on top of the pretrained encoder."""
    lr    = CLS_LR_FROZEN if freeze_encoder else CLS_LR_FULL
    mode  = "frozen encoder" if freeze_encoder else "full fine-tune"
    model = EncoderClassifier(encoder, NUM_CLASSES, freeze_encoder).to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLS_EPOCHS)
    scaler     = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    print(f"\n{'='*60}")
    print(f"  Stage C — Step 2: Classifier fine-tuning  [{mode}]")
    print(f"  LR: {lr}  |  Device: {device}")
    print(f"{'='*60}")

    best_acc       = 0.0
    final_train_acc = 0.0
    epochs_trained  = 0
    for epoch in range(1, CLS_EPOCHS + 1):
        train_loss, train_acc = train_classifier_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        test_loss, test_acc = eval_classifier(model, test_loader, criterion, device)
        scheduler.step()

        epochs_trained  = epoch
        final_train_acc = train_acc
        if test_acc > best_acc:
            best_acc  = test_acc
            save_name = f"autoenc_cls_{'frozen' if freeze_encoder else 'full'}_best.pth"
            torch.save(model.state_dict(), save_name)

        print(
            f"Epoch [{epoch:>2}/{CLS_EPOCHS}]  "
            f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%  |  "
            f"Test  loss: {test_loss:.4f}  acc: {test_acc*100:.2f}%"
        )

        if early_stop(test_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest test accuracy [{mode}]: {best_acc*100:.2f}%")

    log_result(
        stage="C",
        architecture="ConvEncoder + ClassHead",
        mode=mode,
        optimizer="Adam",
        lr=lr,
        epochs_trained=epochs_trained,
        train_acc=final_train_acc,
        test_acc=best_acc,
        notes="Encoder pretrained on 100k unlabeled. Head: GAP→Linear(256→128)→Linear(128→10)",
    )
    return best_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    stl10_input.download_and_extract()

    print(f"Using device: {DEVICE}  |  AMP: {USE_AMP}")
    print("Loading datasets...")
    unlabeled_loader = load_unlabeled(UNLABELED_X_PATH, batch_size=BATCH_SIZE)
    train_loader     = load_labeled(TRAIN_X_PATH, TRAIN_Y_PATH, train=True,  batch_size=BATCH_SIZE)
    test_loader      = load_labeled(TEST_X_PATH,  TEST_Y_PATH,  train=False, batch_size=BATCH_SIZE)
    print(
        f"Unlabeled: {len(unlabeled_loader.dataset)} images  |  "
        f"Train: {len(train_loader.dataset)} labeled  |  "
        f"Test: {len(test_loader.dataset)} labeled"
    )

    # ------------------------------------------------------------------
    # Step 1 — Pretrain autoencoder on 100 000 unlabeled images
    # ------------------------------------------------------------------
    autoencoder = ConvAutoencoder()
    pretrain_autoencoder(autoencoder, unlabeled_loader, DEVICE)

    # ------------------------------------------------------------------
    # Step 2 — Fine-tune classifier using the pretrained encoder
    #          Run twice: frozen encoder and full fine-tune
    # ------------------------------------------------------------------
    results: list[tuple[str, float]] = []

    for freeze in (True, False):
        # Load a fresh copy of the pretrained encoder for each run
        fresh_encoder = ConvEncoder()
        fresh_encoder.load_state_dict(autoencoder.encoder.state_dict())

        acc = finetune_classifier(
            fresh_encoder, freeze, train_loader, test_loader, DEVICE
        )
        results.append(("frozen encoder" if freeze else "full fine-tune", acc))

        # Free GPU memory before the next run
        del fresh_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"  GPU memory cleared after classifier ({'frozen' if freeze else 'full'})")

    print(f"\n{'='*60}")
    print("  Stage C — Summary")
    print(f"{'='*60}")
    for mode, acc in results:
        print(f"  [{mode:<18}]  test acc: {acc*100:.2f}%")
    print("\nStage C complete.")


if __name__ == "__main__":
    main()
