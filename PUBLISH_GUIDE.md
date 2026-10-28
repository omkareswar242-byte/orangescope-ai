# Put the project on GitHub and publish the Flask website

## 1. GitHub link
- Create a new public GitHub repository called `orangescope-ai`.
- In the extracted `orange_disease_classification` folder, upload all the files and folders (NOT the ZIP itself). Keep `app.py`, `orange_cv`, `models`, `web`, `results`, `report`, `.python-version`, `render.yaml`, and `requirements-deploy.txt` at the repository root.
- When GitHub shows your files, copy its URL. This is the GitHub link for the company form.

## 2. Public app link (Render)
- Open https://dashboard.render.com
- New -> Web Service -> connect your GitHub `orangescope-ai` repository.
- Name: `orangescope-ai` (or another available name).
- Language: Python 3
- Branch: main
- Root Directory: leave empty (if your repo's root is the project folder as described above).
- Build Command: `pip install -r requirements-deploy.txt`
- Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app`
- Choose Free if available; Render's 512 MB memory may not be enough to run both PyTorch and scikit-learn together. Free instances also sleep when idle. If the service is repeatedly killed due to insufficient memory, a higher-memory host/plan is required.
- After it says Live, open the generated public HTTPS link, test Use sample and uploading one image; submit THAT URL in Published Link.

## Important
- The model scores in the report are not verified production benchmarks.
- The original assignment specifies AVOCADO ripening, whereas this demo classifies ORANGE conditions. Get written approval for a change of scope before calling it the requested assignment.
- Never submit a localhost URL as a published website.
