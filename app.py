"""Flask website for the orange image classification project.

Start the website with: python app.py
The saved SVM and CNN models are loaded only when predictions are needed.
"""
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort
from werkzeug.exceptions import RequestEntityTooLarge

from orange_cv.inference import ImageValidationError, predict_bytes, load_models

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder="web", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB per image


@app.route("/")
def home():
    return send_file(BASE_DIR / "web" / "index.html")


@app.route("/api/health")
def health():
    try:
        load_models()
        return jsonify({"status": "ready", "models_loaded": True})
    except (OSError, RuntimeError, ValueError):
        return jsonify({"detail": "Unable to load saved models."}), 503


@app.route("/api/meta")
def model_details():
    # These are the results measured during training, not new predictions.
    result_file = BASE_DIR / "results" / "summary.json"
    with result_file.open(encoding="utf-8") as file:
        summary = json.load(file)

    return jsonify({
        "classes": summary["classes"],
        "counts": summary["audit"]["classes"],
        "total_images": summary["audit"]["image_count"],
        "validation": summary["validation"],
        "test": summary["test"],
        "selected_model": summary["selected_on_validation_macro_f1"],
    })


@app.route("/api/example")
def example_image():
    return send_file(BASE_DIR / "examples" / "orange_blackspot_example.jpg")


@app.route("/api/report")
def download_report():
    report = BASE_DIR / "report" / "Orange_Condition_Classification_Report.pdf"
    return send_file(report, as_attachment=True, download_name=report.name)


@app.route("/api/plot/<name>")
def plot_image(name):
    # Only allow the listed images. Never accept arbitrary file paths.
    plots = {
        "class_balance": "class_balance.png",
        "svm_confusion": "svm_test_confusion_matrix.png",
        "cnn_confusion": "cnn_test_confusion_matrix.png",
        "image_quality": "image_quality.png",
    }
    if name not in plots:
        abort(404)
    return send_file(BASE_DIR / "results" / plots[name])


@app.route("/api/predict", methods=["POST"])
def predict():
    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
    image_type = (request.mimetype or "").lower()

    if image_type not in allowed_types:
        return jsonify({"detail": "Please upload a JPG, PNG or WebP image."}), 415

    try:
        image_data = request.get_data(cache=False)
        prediction = predict_bytes(image_data)
        return jsonify(prediction)
    except ImageValidationError as error:
        return jsonify({"detail": str(error)}), 422
    except (OSError, RuntimeError) as error:
        app.logger.error("Prediction error: %s", error)
        return jsonify({"detail": "Could not load the model files."}), 503


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({"detail": "Maximum image size is 8 MB."}), 413


if __name__ == "__main__":
    # Use a development server for the local assignment demonstration.
    app.run(host="127.0.0.1", port=5000, debug=False)
