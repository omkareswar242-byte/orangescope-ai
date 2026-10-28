# Start OrangeScope with Flask

1. Extract the ZIP file.
2. Open **orange_disease_classification** in VS Code.
3. Install dependencies: `python -m pip install -r requirements-web.txt`
4. Start: `python app.py` (or double-click `start_app.bat`).
5. Open **http://127.0.0.1:5000** in your browser.
6. Click **Use sample**, or upload a new orange image and click **Predict condition**.
7. Scroll down to show the confusion matrices and compare the models.
8. Click **Download assignment report** to view the measured PDF.

You do **not** need Orange Dataset.zip to run the saved checkpoints. You only need it if you want to retrain. For retraining, extract it to `data/raw` and run `python run_pipeline.py --data-root data/raw --epochs 10 --image-size 96`.

**Scope warning:** The Greenpod Labs assignment requested avocado ripening stages. This is an orange-condition dataset and cannot be described as avocado classification.
