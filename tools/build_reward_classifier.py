"""Train the frozen 14-label CXR classifier used by the GRPO reward."""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_labels(value) -> dict[str, float]:
    return value if isinstance(value, dict) else ast.literal_eval(str(value))


class CXRDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[dict[str, float]], label_names: list[str], transform):
        self.paths = paths
        self.labels = labels
        self.label_names = label_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = self.transform(Image.open(self.paths[index]).convert("RGB"))
        values = []
        for name in self.label_names:
            value = float(self.labels[index].get(name, 0.0))
            values.append(0.0 if value < 0 or np.isnan(value) else value)
        return image, torch.tensor(values, dtype=torch.float32)


class Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, images):
        return self.model(images)


def run_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for images, labels in tqdm(loader, leave=False, desc="train" if training else "val"):
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = criterion(model(images), labels)
            if training:
                loss.backward()
                optimizer.step()
        total += loss.item() * images.shape[0]
    return total / max(1, len(loader.dataset))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Training CSV with image paths and CheXpert labels.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="artifacts/best_classifier.pt")
    parser.add_argument("--image-column", default="path")
    parser.add_argument("--labels-column", default="chexpert_labels")
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.csv).dropna(subset=[args.image_column, args.labels_column]).copy()
    labels = [parse_labels(value) for value in frame[args.labels_column]]
    label_names = list(labels[0].keys())
    root = Path(args.image_root)
    paths = [root / str(value) for value in frame[args.image_column]]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; first missing path: {missing[0]}")

    indices = np.random.default_rng(args.seed).permutation(len(paths)).tolist()
    val_size = max(1, int(round(len(indices) * args.val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("The classifier training set must contain at least two images.")
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_set = CXRDataset(
        [paths[i] for i in train_indices], [labels[i] for i in train_indices], label_names, train_transform
    )
    val_set = CXRDataset(
        [paths[i] for i in val_indices], [labels[i] for i in val_indices], label_names, val_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(args.model_name, len(label_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "label_names": label_names,
            "args": {"model_name": args.model_name, **vars(args)},
        },
        output,
    )
    print(f"Saved reward classifier to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
