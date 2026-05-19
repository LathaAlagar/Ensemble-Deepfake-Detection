Advanced Fake Content Detection Project

This project detects deepfake images and videos using an ensemble of pretrained models.

Files:
- main.py              : Streamlit app for image/video detection
- train_best_model.py  : Training script for a new EfficientNet-based classifier
- calibrate_thresholds.py : Validation helper that recommends thresholds on Dataset/Validation
- requirements.txt    : Python package dependencies
- Dataset/            : Training, validation, and test image folders
- mobilenet_model.h5, resnet_model.h5, efficientnet_model.h5, cnn_lstm_deepfake.keras : pretrained models

Setup:
1. Open a terminal in the project folder.
2. Create a virtual environment:
   python -m venv .venv
3. Activate the environment:
   .venv\Scripts\activate
4. Install dependencies:
   pip install -r requirements.txt

Training:
- To train a new best model, run:
  python train_best_model.py

Calibration:
- To find the best image threshold from validation data, run:
  python calibrate_thresholds.py
- The script writes `calibration_results.csv` with per-model and ensemble performance.

Running the app:
- Start Streamlit:
  streamlit run main.py

Usage:
- Upload an image or video in the web app.
- Adjust image/video thresholds from the sidebar if needed.
- If you train `best_model.h5`, it will automatically be included in the ensemble.

Notes:
- If models are missing, the app shows an error message.
- For best results, train `best_model.h5` and calibrate thresholds before using the app.

Recommended thresholds (from validation calibration):
- mobilenet: 0.66 (Acc 0.8560, F1 0.8590)
- resnet: 0.72 (Acc 0.5015, F1 0.6669)
- efficientnet: 0.40 (Acc 0.5000, F1 0.6667)
- ensemble (recommended default): 0.70 (Acc 0.8505, F1 0.8549)

The app `main.py` now uses `0.70` as the default image and video threshold and `0.66` as the per-frame fake threshold.
