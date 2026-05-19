import argparse
import os
import csv
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODEL_PATHS = {
    "mobilenet": "mobilenet_model.h5",
    "resnet": "resnet_model.h5",
    "efficientnet": "efficientnet_model.h5",
}

BEST_MODEL_PATH = "best_model.h5"
if os.path.exists(BEST_MODEL_PATH):
    MODEL_PATHS["best_model"] = BEST_MODEL_PATH


def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
                print(f"Loaded {name}")
            except Exception as exc:
                print(f"Failed to load {name}: {exc}")
        else:
            print(f"Missing {path}")
    return models


def preprocess(img, model_name):
    size = models[model_name].input_shape[1:3]
    img = cv2.resize(img, size)

    if model_name == "mobilenet":
        img = mobilenet_preprocess(img)
    elif model_name == "resnet":
        img = resnet_preprocess(img)
    else:
        img = img / 255.0

    return np.expand_dims(img, axis=0)


def load_images(folder_path, sample_size=None):
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(folder_path, file_name))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if sample_size is not None and sample_size > 0 and len(images) > sample_size:
        rng = np.random.default_rng(1234)
        indices = rng.choice(len(images), size=sample_size, replace=False)
        images = [images[i] for i in indices]
    return images


def evaluate_model(preds_real, preds_fake, thresholds):
    results = []
    y_true = np.concatenate([np.ones(len(preds_real)), np.zeros(len(preds_fake))])

    for threshold in thresholds:
        y_pred = np.concatenate([
            (preds_real >= threshold).astype(int),
            (preds_fake >= threshold).astype(int)
        ])

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1": f1,
        })

    return results


parser = argparse.ArgumentParser(description="Calibrate threshold values using validation images.")
parser.add_argument("--sample", type=int, default=0, help="Number of validation images per class to sample for a faster run")
args = parser.parse_args()

models = load_models()
if len(models) == 0:
    print("No models loaded. Exiting.")
    exit(1)

val_path = "Dataset/Validation"
real_images = load_images(os.path.join(val_path, "Real"), sample_size=args.sample)
fake_images = load_images(os.path.join(val_path, "Fake"), sample_size=args.sample)

print(f"\nLoaded {len(real_images)} REAL images")
print(f"Loaded {len(fake_images)} FAKE images")

if len(real_images) == 0 or len(fake_images) == 0:
    print("No validation images available. Exiting.")
    exit(1)

preds_real = {name: [] for name in models}
preds_fake = {name: [] for name in models}

print("\nPredicting validation images...")
for label, images, preds in [("REAL", real_images, preds_real), ("FAKE", fake_images, preds_fake)]:
    for i, img in enumerate(images, start=1):
        for name in models:
            pred = models[name].predict(preprocess(img, name), verbose=0)[0][0]
            preds[name].append(pred)

        if i % max(1, len(images) // 5) == 0:
            print(f"  {label}: {i}/{len(images)}")

for name in preds_real:
    preds_real[name] = np.array(preds_real[name])
    preds_fake[name] = np.array(preds_fake[name])

ensemble_real = np.mean([preds_real[name] for name in models], axis=0)
ensemble_fake = np.mean([preds_fake[name] for name in models], axis=0)

thresholds = np.arange(0.40, 0.80, 0.02)

all_results = []
summary = {}
for name in list(models.keys()) + ["ensemble"]:
    real_arr = ensemble_real if name == "ensemble" else preds_real[name]
    fake_arr = ensemble_fake if name == "ensemble" else preds_fake[name]
    results = evaluate_model(real_arr, fake_arr, thresholds)

    best = max(results, key=lambda x: (x["f1"], x["accuracy"]))
    summary[name] = best
    all_results.extend([{"model": name, **row} for row in results])

    print(f"\n{name.upper()} best threshold: {best['threshold']:.2f} | Acc: {best['accuracy']:.4f} | Prec: {best['precision']:.4f} | Rec: {best['recall']:.4f} | F1: {best['f1']:.4f}")

csv_path = "calibration_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["model", "threshold", "accuracy", "precision", "recall", "f1"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nSaved calibration details to {csv_path}")
print("\nRecommended thresholds:")
for name, best in summary.items():
    print(f"  {name}: {best['threshold']:.2f} (Acc {best['accuracy']:.4f}, F1 {best['f1']:.4f})")
