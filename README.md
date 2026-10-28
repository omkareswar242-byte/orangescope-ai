# OrangeScope: Orange Image Classification (SVM vs CNN)

This is a computer vision project in **Python and Flask**. It predicts one of four visual orange conditions:

- `blackspot`
- `canker`
- `fresh`
- `grenning` (spelled this way in the provided dataset)

The project compares **two different approaches**: an SVM with OpenCV color/texture features and a small CNN trained with PyTorch. The Flask website makes it easy to test an image with both models.

**Assignment scope note:** The original Greenpod Labs question asked for avocado ripening stages. This project instead uses the supplied orange dataset. It is an **orange-condition classification demo**, not an avocado ripening classifier. Obtain permission before substituting it for the requested assignment.

## 1. How to start the frontend (Windows)

You need Python 3.10 or newer. Open the folder in VS Code, then open a terminal:

```bash
python -m pip install -r requirements-web.txt
python app.py
```

Go to **http://127.0.0.1:5000**. Choose an image, click **Predict condition**, or click **Use sample**. The website runs locally; uploaded images are not saved.

You can also double-click `start_app.bat` after installing the packages.

### What the webpage includes

- Image upload and image preview
- SVM and CNN results on the same image
- Scores for the four classes (uncalibrated)
- A saved comparison of accuracy, macro F1, precision and recall
- Both confusion matrix images, class counts, and a PDF report download

## 2. Dataset

The original **Orange Dataset.zip** contained 1,090 images:

| Class | Train folder | Test folder | Total |
|---|---:|---:|---:|
| blackspot | 184 | 22 | 206 |
| canker | 179 | 22 | 201 |
| fresh | 281 | 33 | 314 |
| grenning | 347 | 22 | 369 |
| **Total** | **991** | **99** | **1,090** |

I split the original training images further into train and validation groups (816 and 175). The original **99 test images** were kept separate until evaluation. The GitHub project is intended to track the **individual source images** under `data/raw/train/` and `data/raw/test/` so training is reproducible. The original 139 MB ZIP itself should **not** be committed. To import it and generate a per-image SHA-256 manifest, run:

```bash
python prepare_dataset.py "C:/Users/YOUR_NAME/Downloads/Orange Dataset.zip"
```

This verifies exactly 1,090 images and writes `data/orange_dataset_manifest.csv`. Use GitHub Desktop to commit and push the extracted photos after checking that you have permission to publish the provided dataset.

## 3. Results (real results from the saved original training run)

| Metric | SVM | CNN |
|---|---:|---:|
| Test accuracy | 94.95% | 94.95% |
| Test macro precision | 0.9436 | 0.9537 |
| Test macro recall | 0.9432 | 0.9432 |
| Test macro F1 | 0.9432 | 0.9424 |
| Validation macro F1 | 0.9450 | 0.9039 |

**I selected SVM based on validation macro F1.** Both models made 94 correct predictions out of 99 test images. These are **within-dataset results**, not production performance. The uploaded dataset includes different backgrounds and lighting; the results might be lower on new real-world photos.

The Python code is compatible with the **saved trained checkpoints**. These checkpoint metrics were not recomputed by pretending to retrain after rewriting the code; the originally measured results remain in `results/summary.json`.

## 4. How to retrain both models

Extract the separately provided `Orange Dataset.zip` into `data/raw`, keeping its `train` and `test` directories. It should look like this:

```text
data/raw/
    train/
        blackspot/
        canker/
        fresh/
        grenning/
    test/
        blackspot/
        canker/
        fresh/
        grenning/
```

Run:

```bash
python run_pipeline.py --data-root data/raw --epochs 10 --image-size 96 --batch-size 32
```

Training saves checkpoints into `models/`, computes validation and test metrics, and regenerates the PDF report. Training again may produce slightly different scores on another computer.

**Steps used:** inspect images → create split → extract SVM features → train SVM → train CNN → evaluate on validation → select model → evaluate on test → save results and example prediction.

### Understanding the model code

| File | What it does |
|---|---|
| `orange_cv/data.py` | Reads dataset, counts images, screens dark/blurred files, splits data |
| `orange_cv/features.py` | Converts each image into color and texture numbers |
| `orange_cv/classical.py` | Trains an SVM on those numerical features |
| `orange_cv/cnn.py` | Defines and trains the CNN using PyTorch |
| `orange_cv/evaluation.py` | Calculates metrics and confusion matrices |
| `orange_cv/inference.py` | Loads saved models and predicts a new image |
| `run_pipeline.py` | Runs the complete experiment |
| `app.py` | Flask server, image-upload prediction endpoint, report routes |
| `web/` | HTML, CSS and JavaScript frontend |
| `generate_report.py` | Generates the assignment PDF from saved metrics |

## 5. Predict an image from the terminal

```bash
python predict.py --image examples/orange_blackspot_example.jpg
```

## 6. Test the project

```bash
python -m unittest discover -s tests -v
```

The Flask API tests require the Flask package from `requirements-web.txt`.

## 7. Model limitations

The CNN is a compact four-convolution-block network, **trained from scratch**, not pretrained. The SVM may be sensitive to the lighting and color of the photo. Both models classify the image as a whole; they do not perform verified bounding-box object detection. Single-view photos, unclear class boundaries, backgrounds, and the lack of independent real-world field photos are important limitations. Read `report/Orange_Condition_Classification_Report.pdf` for the four required company questions.
