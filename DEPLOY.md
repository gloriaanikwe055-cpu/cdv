# Deploying the dashboard to Streamlit Community Cloud

The interactive dashboard (`06_gui/dashboard.py`) is a Streamlit app. Streamlit
Community Cloud is free and purpose-built for it. **Vercel/Netlify will not work**
— Streamlit needs a long-running server, not serverless functions.

The repo already contains everything the deployed app needs:
- `01_data_generation/output/synthetic_cvd.parquet` — the dataset (committed)
- `04_evaluation/output/best_model.pkl` — the frozen model (committed)
- `.streamlit/config.toml` — theme + headless server
- `requirements.txt` — dependencies

If the dataset were ever missing on the host, the app regenerates it from seed 42
automatically, so the demo never shows an empty page.

## Step 1 — Push this repo to GitHub (one time)

Create an empty repo on github.com (e.g. `cvd-detection`), then from this folder:

```powershell
# if `git` is not on your PATH, use the full path to git.exe you installed
git remote add origin https://github.com/<your-username>/cvd-detection.git
git branch -M main
git push -u origin main
```

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/cvd-detection`
   - **Branch:** `main`
   - **Main file path:** `06_gui/dashboard.py`
4. Open **Advanced settings** and set **Python version = 3.11** (matches how it was
   built and tested).
5. Click **Deploy**. The first build installs the dependencies and takes ~2–4 minutes.

You'll get a public URL like `https://<your-app>.streamlit.app` — that's the link
to share for the meeting.

## Updating the deployed app

Streamlit Cloud redeploys automatically on every `git push` to `main`.

If you re-run the pipeline and the frozen model changes, re-commit the two runtime
artefacts (they are otherwise git-ignored) and push:

```powershell
git add -f 01_data_generation/output/synthetic_cvd.parquet 04_evaluation/output/best_model.pkl
git commit -m "refresh deployed dataset + model"
git push
```

## Notes

- The app is read-only: it loads the frozen model and the dataset; it never trains
  on the host and never touches real patient data.
- The disclaimer "SYNTHETIC data — not for clinical use" is shown on every page.
