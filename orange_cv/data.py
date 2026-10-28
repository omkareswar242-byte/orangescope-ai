"""Read the orange dataset, check image quality and create train/val/test sets."""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXPECTED_CLASSES = {"blackspot", "canker", "fresh", "grenning"}


def read_image(path, size=None):
    # imdecode works even for some paths that imread cannot read.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot open image: {path}")
    if size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def scan_dataset(folder):
    """Get the filenames and labels from the dataset folders."""
    folder = Path(folder).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    split_names = {"train", "test", "val", "valid", "validation"}
    split_folders = [p for p in folder.iterdir() if p.is_dir() and p.name.lower() in split_names]
    image_list = []

    # Dataset may have train/class/image.jpg or class/image.jpg.
    if split_folders:
        folders_to_check = []
        for split_folder in split_folders:
            split = split_folder.name.lower()
            if split in {"valid", "validation"}:
                split = "val"
            folders_to_check.append((split_folder, split))
    else:
        folders_to_check = [(folder, "")]

    for parent, split in folders_to_check:
        for class_folder in sorted(parent.iterdir()):
            if not class_folder.is_dir():
                continue
            for image in class_folder.rglob("*"):
                if image.is_file() and image.suffix.lower() in IMAGE_TYPES:
                    image_list.append({
                        "path": str(image),
                        "relative_path": image.relative_to(folder).as_posix(),
                        "label": class_folder.name,
                        "provided_split": split,
                    })

    if not image_list:
        raise ValueError("No images found. Check the data/raw folder structure.")
    return pd.DataFrame(image_list)


def validate_orange_labels(dataframe):
    labels = set(dataframe["label"].str.lower().unique())
    if labels != EXPECTED_CLASSES:
        raise ValueError(f"Expected classes {sorted(EXPECTED_CLASSES)}; found {sorted(labels)}")
    if len(dataframe) < 20:
        raise ValueError("Too few images to compare the models.")


def file_hash(path):
    """SHA-256 tells us if two image files contain exactly the same bytes."""
    value = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def difference_hash(gray_image):
    """A rough check for similar pictures. It is not proof of duplicates."""
    small = cv2.resize(gray_image, (9, 8))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    bit_text = "".join(str(int(bit)) for bit in bits)
    return "".join(f"{int(bit_text[i:i+4], 2):x}" for i in range(0, 64, 4))


def audit(dataframe, results_folder):
    """Make a CSV of image quality checks and count the four classes."""
    folder = Path(results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    checked = []
    errors = []

    for row in dataframe.to_dict("records"):
        try:
            image = read_image(row["path"])
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            checked.append({
                **row,
                "width": width,
                "height": height,
                "mean_brightness": round(brightness, 2),
                "blur_laplacian": round(sharpness, 2),
                "sha256": file_hash(row["path"]),
                "dhash": difference_hash(gray),
            })
        except (OSError, ValueError) as error:
            errors.append({"path": row["path"], "error": str(error)})

    image_info = pd.DataFrame(checked)
    if image_info.empty:
        raise ValueError("None of the images could be read.")

    # These are screening rules, not guaranteed image defects.
    image_info["flag_too_dark"] = image_info["mean_brightness"] < 40
    image_info["flag_too_bright"] = image_info["mean_brightness"] > 215
    image_info["flag_low_detail"] = image_info["blur_laplacian"] < 30
    image_info["flag_small_image"] = (image_info["width"] < 100) | (image_info["height"] < 100)
    image_info["flag_exact_duplicate"] = image_info.duplicated("sha256", keep=False)
    image_info["flag_possible_near_duplicate"] = image_info.duplicated("dhash", keep=False)
    image_info.to_csv(folder / "image_audit.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(folder / "unreadable_files.csv", index=False)

    counts = image_info["label"].value_counts().sort_index()
    details = {
        "image_count": int(len(image_info)),
        "classes": counts.to_dict(),
        "imbalance_ratio_largest_to_smallest": round(counts.max() / counts.min(), 3),
        "unreadable_count": len(errors),
        "dark_count": int(image_info["flag_too_dark"].sum()),
        "bright_count": int(image_info["flag_too_bright"].sum()),
        "low_detail_count": int(image_info["flag_low_detail"].sum()),
        "small_count": int(image_info["flag_small_image"].sum()),
        "exact_duplicate_images": int(image_info["flag_exact_duplicate"].sum()),
        "possible_near_duplicate_images": int(image_info["flag_possible_near_duplicate"].sum()),
        "resolution_min": f"{image_info['width'].min()}x{image_info['height'].min()}",
        "resolution_max": f"{image_info['width'].max()}x{image_info['height'].max()}",
    }
    return image_info, details


def split_dataset(image_info, seed=42):
    """Keep the supplied test set separate, and split training into train/val.

    Stratified splitting keeps roughly the same label proportions in each set.
    """
    # Removing exact repeated files prevents them appearing on both sides.
    data = image_info.drop_duplicates(subset="sha256", keep="first").copy()
    given_splits = set(data["provided_split"].unique()) - {""}

    if "test" in given_splits:
        test = data[data["provided_split"] == "test"].copy()
        train = data[data["provided_split"] == "train"].copy()
        if "val" in given_splits:
            val = data[data["provided_split"] == "val"].copy()
        else:
            # 17.647% of the supplied training pool gives ~15% overall validation.
            train, val = train_test_split(train, test_size=0.17647,
                                          random_state=seed, stratify=train["label"])
    elif "train" in given_splits and "val" in given_splits:
        train = data[data["provided_split"] == "train"].copy()
        val = data[data["provided_split"] == "val"].copy()
        train, test = train_test_split(train, test_size=0.15,
                                       random_state=seed, stratify=train["label"])
    else:
        train, remaining = train_test_split(data, test_size=0.30,
                                            random_state=seed, stratify=data["label"])
        val, test = train_test_split(remaining, test_size=0.50,
                                     random_state=seed, stratify=remaining["label"])

    pieces = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.copy()
        part["split"] = name
        pieces.append(part)
    final = pd.concat(pieces, ignore_index=True)

    if final.groupby("sha256")["split"].nunique().max() > 1:
        raise RuntimeError("Same image appears in more than one data split.")
    all_labels = set(final["label"])
    for name in ["train", "val", "test"]:
        if set(final.loc[final["split"] == name, "label"]) != all_labels:
            raise ValueError(f"Some classes are missing in the {name} split.")
    return final
