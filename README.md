# NEXUS — AI Radiomics Intelligence Platform

> **BE 645 · AI & Radiomics · University of Louisville · Spring 2026**
> Final Project — Option 3: Graphical User Interface (GUI) Design

NEXUS is a fully browser-based radiomics pipeline application. It wraps a **Flask REST API** backend around the course helper library (`HMB_Spring_2026_Helpers.py`) and serves a polished **single-page HTML frontend**. The four-tab interface guides users through the complete six-step AI workflow — from raw image folders all the way through feature extraction, machine learning classification, deep learning training, and prediction on new images.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [One-Time Setup: Fix the HTML Path](#one-time-setup-fix-the-html-path)
6. [Running the Application](#running-the-application)
7. [Tab-by-Tab Usage Guide](#tab-by-tab-usage-guide)
8. [Dataset Format](#dataset-format)
9. [Deep Learning Tab — GPU Setup](#deep-learning-tab--gpu-setup)
10. [Troubleshooting](#troubleshooting)

---

## Features

| Tab | Pipeline Phase | What it does |
|-----|---------------|--------------|
| **Feature Extraction** | Preprocessing | Loads image folders, pairs masks automatically, extracts FOF / GLCM / GLRLM / GLSZM features, exports CSV |
| **ML Classification** | Model training & evaluation | Uploads CSV, configures scaler / classifier / feature-selection, shows confusion matrix and full metrics, exports `.pkl` model |
| **Deep Learning / CNN** | Model training & evaluation | Trains a pretrained backbone (MobileNetV2, ResNet50, VGG16, EfficientNetB0, InceptionV3) via a GPU-capable subprocess, streams live training log |
| **Predict New Image** | Deployment | Runs inference with either an ML (`.pkl`) or CNN (`.keras`) model on a single unseen image |

---

## Project Structure

```
.
├── Pipeline_Application.py          # Flask API backend (all endpoints)
├── Pipeline Application Website.html  # Single-page frontend (served by Flask)
├── HMB_Spring_2026_Helpers.py       # Course helper library (radiomics functions)
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** The `be645/` folder (a bundled Conda Python runtime used for GPU CNN training) is excluded from this repository via `.gitignore` because it is several gigabytes in size. See the [Deep Learning Tab — GPU Setup](#deep-learning-tab--gpu-setup) section for how to replicate it.

---

## Prerequisites

- **Python 3.10** (the helper library targets CPython 3.10; other 3.x versions may work but are untested)
- **pip** (comes with Python)
- A modern browser (Chrome, Edge, or Firefox recommended)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **TensorFlow is optional.** The CNN tabs (Tab 3 and CNN prediction in Tab 4) are automatically disabled if TensorFlow is not installed. All other tabs remain fully functional. See [Deep Learning Tab — GPU Setup](#deep-learning-tab--gpu-setup) if you want CNN support.

---

## One-Time Setup: Fix the HTML Path

Before running the server for the first time, open `Pipeline_Application.py` in any text editor and find the `index()` route near line 287:

```python
@app.route("/")
def index():
    path = r"D:\Spring26\AI in Raadiomics\Final project\Final"   # <-- CHANGE THIS
    return send_from_directory(path, "Pipeline Application Website.HTML")
```

Replace the hardcoded path with the actual directory where you cloned/placed the project files. For example:

**Windows:**
```python
path = r"C:\Users\YourName\Documents\nexus-radiomics"
```

**macOS / Linux:**
```python
path = "/home/yourname/nexus-radiomics"
```

Save the file. This only needs to be done once.

---

## Running the Application

```bash
python Pipeline_Application.py
```

You should see output similar to:

```
Checking dependencies ...
Dependencies OK.

[TF] TensorFlow found — all tabs enabled.   # or "NOT found" if TF is not installed
HMB LOADED SUCCESSFULLY
===================================================
  NEXUS Flask API v2.3  —  http://localhost:5000
===================================================
 * Running on http://127.0.0.1:5000
```

Open your browser and go to **http://localhost:5000**.
The green dot in the top navigation bar will turn on once the API is reachable.

To stop the server, press `Ctrl + C` in the terminal.

---

## Tab-by-Tab Usage Guide

### Tab 1 — Feature Extraction

1. Enter the **full path** to your dataset root folder (see [Dataset Format](#dataset-format) below).
2. Choose **target size** for ROI extraction (default 128 × 128 px).
3. Toggle the feature families you want: **FOF**, **GLCM**, **GLRLM**, **GLSZM**.
4. Click **Extract Features**.
5. A progress bar streams in real time. When done, a preview table appears and you can **download the CSV**.

### Tab 2 — ML Classification

1. **Upload** the feature CSV produced in Tab 1 (or any compatible CSV with a `Class` column).
2. Optionally upload a separate **test CSV**; otherwise the app splits your training data automatically.
3. Configure: **Scaler**, **Classifier**, **Feature Selection** method and ratio.
4. Click **Train Model**.
5. Review the **confusion matrix** and full metric table, then **download the `.pkl` model** for later prediction.

### Tab 3 — Deep Learning / CNN

> Requires TensorFlow and a separate `be645` Python environment. See [Deep Learning Tab — GPU Setup](#deep-learning-tab--gpu-setup).

1. Set the **path to the `be645` `python.exe`** (or `python` on Linux/macOS).
2. Set your **dataset directory** and an **output directory** for saved model files.
3. Pick a **pretrained backbone** and configure training hyperparameters.
4. Click **Start CNN Training** — live logs stream directly from the subprocess.
5. When training finishes, the best model is saved as `Model.keras` in your output directory.

### Tab 4 — Predict New Image

**ML Prediction:**
1. Upload your `.pkl` model file (or use the session model from Tab 2).
2. Upload a grayscale image (and optionally its mask).
3. Click **Predict** — the predicted class and per-class probabilities are displayed.

**CNN Prediction:**
1. Upload your `.keras` model file.
2. Upload the `class_indices.json` file from your CNN output directory (optional but recommended).
3. Upload an RGB image.
4. Click **Predict** — the class label and confidence score are displayed.

---

## Dataset Format

The application expects a root dataset directory with **one subdirectory per class**:

```
dataset/
├── benign/
│   ├── img001.png
│   ├── img001_mask.png      # optional — suffix must be _mask
│   ├── img002.png
│   └── ...
├── malignant/
│   ├── img001.png
│   ├── img001_mask.png
│   └── ...
└── normal/
    └── ...
```

**Rules:**
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`
- Mask files must share the **same filename stem** as the image with `_mask` appended before the extension (e.g., `scan01.png` → `scan01_mask.png`).
- If no mask is found for an image, the **entire image** is used as the region of interest automatically — no images are skipped silently.
- Class labels are inferred from the subdirectory names.

---

## Deep Learning Tab — GPU Setup

The CNN training tab spawns a subprocess using a **separate Python environment** that has TensorFlow (and optionally CUDA) installed. This keeps the lightweight Flask server process separate from the heavy GPU workload.

### Option A — Use the be645 Conda environment (recommended for course workflow)

If you have Anaconda or Miniconda installed and the `be645` environment already exists on your machine, simply point Tab 3 at its `python.exe`:

```
C:\Users\<you>\anaconda3\envs\be645\python.exe
```

### Option B — Create a fresh TensorFlow environment

```bash
# CPU-only
pip install tensorflow-cpu

# GPU (requires NVIDIA drivers + CUDA 11/12 + cuDNN)
pip install tensorflow
```

Then point Tab 3 at the `python.exe` of whichever environment has TensorFlow installed.

### Verifying GPU access

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the list is non-empty, TensorFlow can see your GPU and training will be accelerated.

---

## Troubleshooting

### "HMB helpers unavailable" on startup

`HMB_Spring_2026_Helpers.py` must be in the **same directory** as `Pipeline_Application.py`. Check that both files are present and that you are running Python from the correct folder.

### API dot stays grey / "Offline" in the browser

- Confirm the Flask server is running and printed `Running on http://127.0.0.1:5000`.
- Check that your browser is pointing to `http://localhost:5000` (not HTTPS).
- Some browsers or antivirus software block localhost connections on non-standard ports — try disabling temporarily or using `127.0.0.1:5000` directly.

### Feature extraction produces 0 features

- Verify the dataset folder path is correct and contains class subdirectories.
- Open the **Debug Dataset** panel on Tab 1 (if available) to run a quick sanity check per class.
- Make sure at least one feature family (FOF, GLCM, GLRLM, or GLSZM) is selected.

### Prediction error: "BitGenerator / numpy incompatible"

The `.pkl` model was saved in a different Python/NumPy environment. Retrain the model inside the current environment (Tab 2) and use the freshly downloaded `.pkl`.

### CNN training silently fails

- Verify the `be645` python path points to a valid executable and that TensorFlow is installed in that environment.
- Check the streaming log in the browser for the first error line — it usually identifies a missing package or an invalid dataset path.

---

## License

This project was developed as a course assignment for **BE 645 — AI & Radiomics, Spring 2026** at the University of Louisville. The `HMB_Spring_2026_Helpers.py` library is proprietary course material provided by the instructor and is included here solely for reproducibility of the project.
