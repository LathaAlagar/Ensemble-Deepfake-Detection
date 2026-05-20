# Ensemble-Deepfake-Detection


A deepfake detection system for images and videos that uses an ensemble of pretrained and custom deep learning models. The project includes a Streamlit app, a training script, and a calibration helper for tuning decision thresholds.

## 🚀 Project Highlights

- Detects `REAL` vs `FAKE` content in images and videos
- Uses an ensemble of multiple models for more stable predictions
- Supports custom model training via `train_best_model.py`
- Includes threshold calibration using `calibrate_thresholds.py`
- Provides video frame-level override logic for suspicious content

## 📁 Repository Structure

- `main.py` — Streamlit app for image/video detection
- `train_best_model.py` — Train and fine-tune an EfficientNet-based model
- `calibrate_thresholds.py` — Find optimal validation thresholds
- `requirements.txt` — Python dependencies
- `Dataset/` — Dataset split into `Train`, `Validation`, `Test`
- Model files: `mobilenet_model.h5`, `resnet_model.h5`, `efficientnet_model.h5`, `cnn_lstm_deepfake.keras`, optional `best_model.h5`

## 📂 Dataset Layout

Dataset Download

1. Download the dataset: https://drive.google.com/file/d/1xpB29HPRXY0h0KqeultZmMxLpqaKPghD/view?usp=sharing

2. Extract the ZIP file to your project folder


The expected folder structure is:

```
Dataset/
  Train/
    Fake/
    Real/
  Validation/
    Fake/
    Real/
  Test/
    Fake/
    Real/
```

Put all training images and validation images in these directories before training or calibration.

## ⚙️ Setup

1. Open a terminal in the project folder.
2. Create a virtual environment:

```powershell
python -m venv .venv
```

3. Activate it:

```powershell
.venv\Scripts\activate
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

## 🧠 Training a New Model

To train a new custom `best_model.h5`, run:

```powershell
python train_best_model.py
```

What the script does:

- loads data from `Dataset/Train`
- validates using `Dataset/Validation`
- trains in two phases: feature extraction and fine-tuning
- saves the best model as `best_model.h5`

Once `best_model.h5` exists, `main.py` will include it in the ensemble automatically.

## 🔧 Calibrating Thresholds

Run threshold calibration with:

```powershell
python calibrate_thresholds.py
```

For a faster sampled evaluation use:

```powershell
python calibrate_thresholds.py --sample 1000
```

This script evaluates the validation dataset, prints model-specific thresholds, and creates `calibration_results.csv`.

## 🌐 Using the App

Start the Streamlit application:

```powershell
streamlit run main.py
```

Open the local URL shown in the terminal and then:

- choose `Image` or `Video`
- upload your file
- review the model probabilities and ensemble result
- adjust sidebar thresholds if needed

## ✅ Recommended Thresholds

Based on validation calibration, these defaults are recommended:

- `Image / Video threshold`: `0.70`
- `Frame fake threshold`: `0.66`

## 📝 Notes

- Ensure all required model files are present in the project root.
- If the predictions are inconsistent, retrain `best_model.h5` and recalibrate thresholds.
- Use the `Dataset/Validation` folder to tune thresholds for your data distribution.

## 📌 GitHub Usage

To publish this project to GitHub:

```powershell
git init
.gitignore # add if needed
git add .
git commit -m "Initial fake content detection project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Replace `<your-repo-url>` with your GitHub repository URL.

---
