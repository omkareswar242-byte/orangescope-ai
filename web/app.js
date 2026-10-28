// The frontend sends an image to Flask and displays the two model results.
const imageInput = document.getElementById("image-input");
const dropArea = document.getElementById("drop-area");
const preview = document.getElementById("image-preview");
const predictButton = document.getElementById("predict-button");
let currentFile = null;
let previewLink = null;

function showError(message) {
    const errorBox = document.getElementById("error-message");
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function clearError() {
    document.getElementById("error-message").classList.add("hidden");
}

function chooseImage(file) {
    if (!file) return;
    clearError();

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
        showError("Please select a JPG, PNG or WebP image.");
        return;
    }
    if (file.size > 8 * 1024 * 1024) {
        showError("Image must be smaller than 8 MB.");
        return;
    }

    currentFile = file;
    if (previewLink) URL.revokeObjectURL(previewLink);
    previewLink = URL.createObjectURL(file);
    preview.src = previewLink;
    preview.classList.remove("hidden");
    document.getElementById("upload-placeholder").classList.add("hidden");
    document.getElementById("file-name").textContent = file.name;
    document.getElementById("empty-state").classList.remove("hidden");
    document.getElementById("prediction-results").classList.add("hidden");
    predictButton.disabled = false;
}

function clearImage() {
    currentFile = null;
    imageInput.value = "";
    if (previewLink) URL.revokeObjectURL(previewLink);
    previewLink = null;
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    document.getElementById("upload-placeholder").classList.remove("hidden");
    document.getElementById("file-name").textContent = "No image selected";
    document.getElementById("prediction-results").classList.add("hidden");
    document.getElementById("empty-state").classList.remove("hidden");
    predictButton.disabled = true;
    clearError();
}

function percent(value, decimals = 1) {
    return (Number(value) * 100).toFixed(decimals) + "%";
}

function showScoreBars(containerId, scores) {
    const container = document.getElementById(containerId);
    container.replaceChildren();

    for (const [label, value] of Object.entries(scores)) {
        const line = document.createElement("div");
        line.className = "score-line";

        const name = document.createElement("span");
        name.textContent = label;
        const track = document.createElement("div");
        track.className = "track";
        const fill = document.createElement("div");
        fill.className = "fill";
        fill.style.width = percent(value);
        track.appendChild(fill);
        const number = document.createElement("span");
        number.textContent = percent(value);

        line.append(name, track, number);
        container.appendChild(line);
    }
}

function showPrediction(result) {
    document.getElementById("empty-state").classList.add("hidden");
    document.getElementById("prediction-results").classList.remove("hidden");
    document.getElementById("svm-label").textContent = result.svm.label;
    document.getElementById("cnn-label").textContent = result.cnn.label;
    document.getElementById("svm-score").textContent = percent(result.svm.score) + " score";
    document.getElementById("cnn-score").textContent = percent(result.cnn.score) + " score";

    const agreement = document.getElementById("agreement");
    agreement.textContent = result.agreement
        ? "✓ Both models predicted the same condition"
        : "! The models predicted different conditions";
    agreement.classList.toggle("disagree", !result.agreement);

    showScoreBars("svm-bars", result.svm.scores);
    showScoreBars("cnn-bars", result.cnn.scores);
}

async function predictImage() {
    if (!currentFile) return;
    clearError();
    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": currentFile.type },
            body: currentFile
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Prediction failed");
        showPrediction(result);
    } catch (error) {
        showError(error.message);
    } finally {
        predictButton.textContent = "Predict condition";
        predictButton.disabled = false;
    }
}

async function useSample() {
    try {
        const response = await fetch("/api/example");
        if (!response.ok) throw new Error("Sample image not found");
        const imageBlob = await response.blob();
        chooseImage(new File([imageBlob], "orange_sample.jpg", { type: "image/jpeg" }));
        await predictImage();
    } catch (error) {
        showError(error.message);
    }
}

async function loadStatistics() {
    try {
        const response = await fetch("/api/meta");
        if (!response.ok) throw new Error("Results could not be loaded");
        const stats = await response.json();

        document.getElementById("image-count").textContent = stats.total_images.toLocaleString();
        for (const model of ["svm", "cnn"]) {
            const result = stats.test[model];
            document.getElementById("top-" + model).textContent = percent(result.accuracy, 2);
            document.getElementById(model + "-accuracy").textContent = percent(result.accuracy, 2);
            document.getElementById(model + "-f1").textContent = result.macro_f1.toFixed(4);
            document.getElementById(model + "-precision").textContent = result.macro_precision.toFixed(4);
            document.getElementById(model + "-recall").textContent = result.macro_recall.toFixed(4);
        }

        const maximum = Math.max(...Object.values(stats.counts));
        const container = document.getElementById("class-counts");
        for (const [name, count] of Object.entries(stats.counts)) {
            const title = document.createElement("div");
            title.className = "count-label";
            title.textContent = name + " — " + count + " images";
            const track = document.createElement("div");
            track.className = "count-line";
            const bar = document.createElement("div");
            bar.className = "count-fill";
            bar.style.width = (count / maximum * 100).toFixed(1) + "%";
            track.appendChild(bar);
            container.append(title, track);
        }

        const health = await fetch("/api/health");
        const status = document.getElementById("server-status");
        status.textContent = health.ok ? "● Models ready" : "Models unavailable";
    } catch (error) {
        document.getElementById("server-status").textContent = "Connection error";
        showError(error.message);
    }
}

// Connect the page buttons after the HTML is ready.
dropArea.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", () => chooseImage(imageInput.files[0]));
document.getElementById("clear-button").addEventListener("click", clearImage);
document.getElementById("sample-button").addEventListener("click", useSample);
predictButton.addEventListener("click", predictImage);

for (const eventName of ["dragenter", "dragover"]) {
    dropArea.addEventListener(eventName, event => {
        event.preventDefault();
        dropArea.classList.add("dragging");
    });
}
for (const eventName of ["dragleave", "drop"]) {
    dropArea.addEventListener(eventName, event => {
        event.preventDefault();
        dropArea.classList.remove("dragging");
    });
}
dropArea.addEventListener("drop", event => chooseImage(event.dataTransfer.files[0]));
loadStatistics();
