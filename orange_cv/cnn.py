"""Approach 2: a small CNN trained from scratch with PyTorch.

This is kept simple so each part can be explained in an interview.
The layer names must stay the same as the included saved CNN checkpoint.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OrangeImages(Dataset):
    """Loads orange images and converts their labels to numbers."""

    def __init__(self, dataframe, classes, training=False, size=96, architecture="cnn"):
        self.rows = dataframe.reset_index(drop=True)
        self.label_numbers = {name: index for index, name in enumerate(classes)}

        # CNN was trained with values normalized around 0.5.
        # Only the training images get random changes.
        steps = [transforms.Resize((size, size))]
        if training:
            steps.append(transforms.RandomHorizontalFlip())
            steps.append(transforms.RandomRotation(12))
            steps.append(transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06))
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_tensor = self.transform(image)
        label = self.label_numbers[row["label"]]
        return image_tensor, label


class SmallCNN(nn.Module):
    """Four convolution blocks followed by the class prediction layer."""

    def __init__(self, number_of_classes):
        super().__init__()
        layers = []
        channels = [(3, 16), (16, 32), (32, 64), (64, 96)]

        for input_channels, output_channels in channels:
            layers.append(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(96, number_of_classes),
        )

    def forward(self, images):
        extracted_features = self.features(images)
        return self.head(extracted_features)


def make_model(architecture, number_of_classes, pretrained=False):
    if architecture != "cnn":
        raise ValueError("This project uses the small CNN. Choose --architecture cnn.")
    return SmallCNN(number_of_classes)


def load_model(checkpoint_path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = make_model(saved["architecture"], len(saved["classes"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def fit(train, val, classes, out_dir, architecture="cnn", epochs=10,
        batch_size=32, learning_rate=0.001, patience=4, size=96):
    """Train the CNN, save the best validation-loss checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if size is None:
        size = 96
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(architecture, len(classes)).to(device)

    train_images = OrangeImages(train, classes, training=True, size=size)
    val_images = OrangeImages(val, classes, training=False, size=size)
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False, num_workers=0)

    # Give the less common orange classes a little more importance.
    class_counts = train["label"].value_counts()
    weight_values = [len(train) / (len(classes) * class_counts[name]) for name in classes]
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        # Step 1: Training
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(labels)

        # Step 2: Check the model on validation images.
        model.eval()
        total_val_loss = 0.0
        actual_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                total_val_loss += loss.item() * len(labels)
                actual_labels.extend(labels.cpu().tolist())
                predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())

        train_loss = total_train_loss / len(train_images)
        val_loss = total_val_loss / len(val_images)
        macro_f1 = f1_score(actual_labels, predicted_labels, average="macro", zero_division=0)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_macro_f1": round(float(macro_f1), 5),
        })
        print(f"Epoch {epoch + 1}/{epochs} | loss={train_loss:.4f} | val loss={val_loss:.4f} | F1={macro_f1:.4f}")

        # Step 3: Save only if validation loss is better than before.
        if val_loss < best_loss - 0.0001:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "architecture": architecture,
                "size": size,
            }, folder / "cnn_best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Stopped early because validation loss did not improve.")
                break

    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    return load_model(folder / "cnn_best.pt"), history


def predict(model, dataframe, classes, architecture="cnn", size=96, batch_size=32):
    dataset = OrangeImages(dataframe, classes, training=False, size=size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = next(model.parameters()).device
    all_scores = []

    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            output = model(images.to(device))
            scores = torch.softmax(output, dim=1)
            all_scores.append(scores.cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = [classes[i] for i in scores.argmax(axis=1)]
    return labels, scores
