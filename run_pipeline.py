"""Train and compare the two orange classification models.

Run this after extracting Orange Dataset.zip into data/raw:
    python run_pipeline.py --data-root data/raw --epochs 10 --image-size 96
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from orange_cv import classical, cnn
from orange_cv.data import scan_dataset, validate_orange_labels, audit, split_dataset
from orange_cv.evaluation import evaluate, plot_dataset


def train_and_compare(args):
    project = Path(args.output).resolve()
    results = project / "results"
    models = project / "models"
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    # 1. Load the actual dataset and check the four orange class folders.
    images = scan_dataset(args.data_root)
    validate_orange_labels(images)
    if args.max_images_per_class is not None:
        # Only useful for checking that the code runs, not for final scores.
        grouped = images.groupby(["provided_split", "label"], group_keys=False)
        sample_count = min(args.max_images_per_class, grouped.size().min())
        images = grouped.sample(n=sample_count, random_state=args.seed)

    print("Dataset image count:", len(images))

    # 2. Inspect data problems (brightness, repeated images, class counts).
    audited_images, dataset_summary = audit(images, results)
    plot_dataset(audited_images, results)

    # 3. Split the data. The original test images are used only at the end.
    split_images = split_dataset(audited_images, seed=args.seed)
    split_images[["relative_path", "label", "split", "sha256"]].to_csv(
        results / "split_manifest.csv", index=False
    )
    split_table = pd.crosstab(split_images["label"], split_images["split"])
    split_table.to_csv(results / "split_distribution.csv")
    print("\nDataset split:\n", split_table)

    classes = sorted(split_images["label"].unique().tolist())
    (models / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    train = split_images[split_images["split"] == "train"]
    val = split_images[split_images["split"] == "val"]
    test = split_images[split_images["split"] == "test"]

    # 4. Train Approach 1: color/texture features + SVM.
    print("\nTraining SVM...")
    svm = classical.fit(train, val, models)
    svm_val_labels, _ = classical.predict(svm, val)
    svm_val = evaluate("svm", "validation", val["label"].tolist(),
                       svm_val_labels, classes, results)

    # 5. Train Approach 2: the small CNN with PyTorch.
    print("\nTraining CNN...")
    size = args.image_size
    network, history = cnn.fit(
        train, val, classes, models, architecture="cnn", epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        patience=args.patience, size=size
    )
    cnn_val_labels, _ = cnn.predict(network, val, classes, size=size)
    cnn_val = evaluate("cnn", "validation", val["label"].tolist(),
                       cnn_val_labels, classes, results)

    # 6. Decide which model is better using validation macro F1.
    # Never choose the winner based on the held-out test results.
    if svm_val["macro_f1"] >= cnn_val["macro_f1"]:
        selected_model = "svm"
    else:
        selected_model = "cnn"
    print("\nSelected by validation F1:", selected_model)

    # 7. Only now check both models on the original test images.
    svm_test_labels, svm_test_scores = classical.predict(svm, test)
    cnn_test_labels, cnn_test_scores = cnn.predict(network, test, classes, size=size)
    svm_test = evaluate("svm", "test", test["label"].tolist(),
                        svm_test_labels, classes, results)
    cnn_test = evaluate("cnn", "test", test["label"].tolist(),
                        cnn_test_labels, classes, results)

    prediction_table = test[["path", "relative_path", "label"]].copy()
    prediction_table["svm_prediction"] = svm_test_labels
    prediction_table["cnn_prediction"] = cnn_test_labels
    prediction_table["svm_confidence"] = svm_test_scores.max(axis=1).round(5)
    prediction_table["cnn_confidence"] = cnn_test_scores.max(axis=1).round(5)
    prediction_table.to_csv(results / "heldout_predictions.csv", index=False)

    # 8. Save the actual results used in the web page and final report.
    summary = {
        "dataset_path": str(Path(args.data_root).resolve()),
        "dataset_checked_as_orange": True,
        "is_smoke_test": args.max_images_per_class is not None,
        "classes": classes,
        "audit": dataset_summary,
        "split_counts": split_table.to_dict(),
        "task": "orange_visual_condition_classification",
        "cnn_architecture": "cnn",
        "image_size": size,
        "epochs_completed": len(history),
        "selected_on_validation_macro_f1": selected_model,
        "validation": {"svm": svm_val, "cnn": cnn_val},
        "test": {"svm": svm_test, "cnn": cnn_test},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([svm_val, cnn_val, svm_test, cnn_test]).to_csv(
        results / "model_comparison.csv", index=False
    )
    print("\nHeld-out test results:")
    print(pd.DataFrame([svm_test, cnn_test]).to_string(index=False))

    # 9. Show one example and create the PDF with the measured values.
    from predict import predict_one
    example = test.iloc[0]
    predict_one(str(example["path"]), str(models), str(results), expected=str(example["label"]))

    if args.max_images_per_class is None:
        from generate_report import create_report
        create_report(project / "report" / "Orange_Condition_Classification_Report.pdf",
                      results / "summary.json")
    else:
        print("Smoke test only - not generating a final report from partial data.")

    print("All files saved to:", project)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM and CNN orange classifiers")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default=".")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-class", type=int, default=None,
                        help="Quick code check only; not for report results")
    parser.add_argument("--architecture", default="cnn", choices=["cnn"],
                        help="Included deep learning architecture")
    train_and_compare(parser.parse_args())
