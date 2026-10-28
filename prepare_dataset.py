"""Unzip and verify the orange dataset in data/raw/.
Usage: python prepare_dataset.py "C:/Users/NAME/Downloads/Orange Dataset.zip"
"""
import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

EXPECTED = {
    ("train", "blackspot"): 184, ("train", "canker"): 179,
    ("train", "fresh"): 281, ("train", "grenning"): 347,
    ("test", "blackspot"): 22, ("test", "canker"): 22,
    ("test", "fresh"): 33, ("test", "grenning"): 22,
}

def main(zip_path):
    destination = Path("data/raw").resolve()
    with ZipFile(zip_path) as archive:
        images = [i for i in archive.infolist() if not i.is_dir()]
        counts = Counter()
        for item in images:
            parts = Path(item.filename).parts
            if len(parts) != 3 or (parts[0], parts[1]) not in EXPECTED:
                raise ValueError(f"Unexpected entry: {item.filename}")
            if Path(parts[2]).suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                raise ValueError(f"Not an image: {item.filename}")
            counts[(parts[0], parts[1])] += 1
        if counts != EXPECTED:
            raise ValueError(f"Expected original dataset counts, found: {counts}")
        rows = []
        for item in images:
            path = (destination / item.filename).resolve()
            if not path.is_relative_to(destination):
                raise ValueError(f"Unsafe archive filename: {item.filename}")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(item)
            path.write_bytes(data)
            split, label, _ = Path(item.filename).parts
            rows.append([item.filename, split, label, len(data), hashlib.sha256(data).hexdigest()])
    with Path("data/orange_dataset_manifest.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["relative_path_from_data_raw", "split", "class", "bytes", "sha256"])
        writer.writerows(sorted(rows))
    print(f"Verified and extracted {len(rows)} images to {destination}")
    for category, count in sorted(counts.items()):
        print("/".join(category), count)
    print("Next: commit and push the images using GitHub Desktop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    args = parser.parse_args()
    main(args.zip_path)
