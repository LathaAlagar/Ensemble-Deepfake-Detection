import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from PIL import Image

from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Deepfake Detection System")

st.title("🧠 Deepfake Detection in Images and Videos Using Ensemble Deep Learning")


# =============================
# THRESHOLDS (calibrated defaults)
# =============================
IMAGE_THRESHOLD = 0.70
FRAME_THRESHOLD = 0.66


# =============================
# MODEL PATHS
# =============================
MODEL_PATHS = {
    "mobilenet": "mobilenet_model.h5",
    "resnet": "resnet_model.h5",
    "efficientnet": "efficientnet_model.h5",
    "cnn_lstm": "cnn_lstm_deepfake.keras"
}

BEST_MODEL_PATH = "best_model.h5"
if os.path.exists(BEST_MODEL_PATH):
    MODEL_PATHS["best_model"] = BEST_MODEL_PATH


# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    models = {}
    missing = []

    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except Exception as exc:
                st.error(f"Failed to load {name} ({path}): {exc}")
        else:
            missing.append(path)

    if missing:
        st.sidebar.error("Missing model files: " + ", ".join(missing))

    return models


models = load_models()

if len(models) == 0:
    st.stop()

# Per-model weighting sliders (useful to down-weight models that misclassify)
st.sidebar.header("⚖️ Model Weights")
model_weights = {}
for name in models.keys():
    # default 1.0, allow 0 to 2 range
    model_weights[name] = st.sidebar.slider(
        f"Weight: {name}", min_value=0.0, max_value=2.0, value=1.0, step=0.1
    )


# =============================
# IMAGE PREPROCESS
# =============================
def preprocess(img, model_name, model):

    size = model.input_shape[1:3]

    img = cv2.resize(img, size)

    if model_name == "mobilenet":
        img = mobilenet_preprocess(img)

    elif model_name == "resnet":
        img = resnet_preprocess(img)

    else:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# =============================
# VIDEO FRAME EXTRACTION
# =============================
def extract_frames(video_path, max_frames=50):

    cap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:

        step = max(1, total_frames // max_frames)

        for i in range(0, total_frames, step):

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            ret, frame = cap.read()

            if ret:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)

                if len(frames) >= max_frames:

                    break

    else:

        # Fallback to original method if frame count unavailable

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:

                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

            if len(frames) >= max_frames:

                break

    cap.release()

    return frames


# =============================
# CNN LSTM PREDICTION
# =============================
def predict_cnn_lstm(frames, model):

    try:

        input_shape = model.input_shape

        time_steps = input_shape[1]
        h = input_shape[2]
        w = input_shape[3]

        processed = []

        for f in frames[:time_steps]:

            f = cv2.resize(f, (h, w))
            f = f / 255.0
            processed.append(f)

        while len(processed) < time_steps:
            processed.append(processed[-1])

        arr = np.array(processed)

        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr, verbose=0)[0][0]

        return pred

    except:
        return None


# =============================
# ENSEMBLE DECISION
# =============================
# Supports multiple ensemble strategies to reduce false positives/negatives.
#  - average: final based on the mean probability.
#  - majority: final based on whether most models vote REAL.
#  - all: final is REAL only if every model votes REAL.
#
# These methods let you tune behavior when models disagree.

def ensemble_decision(predictions, threshold, method="average", weights=None):

    # If weights provided, apply them to compute weighted average and weighted votes
    if weights is None:
        weights = [1.0] * len(predictions)

    weights = np.array(weights)
    preds = np.array(predictions)

    if weights.sum() == 0:
        avg = float(np.mean(preds))
    else:
        avg = float(np.sum(preds * weights) / np.sum(weights))

    # Weighted vote counts
    real_weight = float(np.sum(weights[preds >= threshold]))
    fake_weight = float(np.sum(weights[preds < threshold]))

    if method == "average":
        final = 1 if avg >= threshold else 0
    elif method == "majority":
        final = 1 if real_weight > fake_weight else 0
    elif method == "all":
        # require all models (with non-zero weight) to be >= threshold
        required = sum(1 for w, p in zip(weights, preds) if w > 0)
        real_with_weight = sum(1 for w, p in zip(weights, preds) if (w > 0 and p >= threshold))
        final = 1 if real_with_weight == required else 0
    else:
        final = 1 if avg >= threshold else 0

    return avg, real_weight, fake_weight, final


def video_fake_override(model_preds, threshold=0.45, min_fraction=0.4):
    if not model_preds:
        return False

    fake_frames = sum(1 for p in model_preds if p < threshold)
    avg_pred = np.mean(model_preds)
    return (fake_frames / len(model_preds)) >= min_fraction and avg_pred < threshold


# =============================
# INPUT TYPE
# =============================
mode = st.radio("Select Input Type", ["Image", "Video"])

# Let the user choose the ensemble strategy used for the final decision.
ensemble_method = st.radio(
    "Ensemble method",
    ["average", "majority", "all"],
    index=0,
    help="Choose how model outputs are combined into the final REAL/FAKE decision."
)

# Separate ensemble for videos
video_ensemble_method = st.radio(
    "Video Ensemble method",
    ["average", "majority", "all"],
    index=0,
    help="Choose how model outputs are combined for video classification."
)

# Add threshold tuning
st.sidebar.header("⚙️ Threshold Tuning")
image_threshold_slider = st.sidebar.slider(
    "Image Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
    help="Adjust image classification threshold (higher = stricter about calling it REAL)"
)
video_threshold_slider = st.sidebar.slider(
    "Video Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
    help="Adjust video classification threshold (higher = stricter about calling it REAL)"
)
video_frame_fake_threshold = st.sidebar.slider(
    "Video Frame Fake Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.66,
    step=0.05,
    help="Treat individual frames as fake when the predicted REAL probability falls below this value."
)
video_frame_fake_fraction = st.sidebar.slider(
    "Video Fake Frame Fraction",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    help="If this fraction of frames are fake for a model, mark that model as suspicious."
)

st.sidebar.markdown(
    "**Tip:** If fake videos are being labeled REAL, increase the video threshold or increase the frame fake threshold/fraction."
)


# =============================
# IMAGE DETECTION
# =============================
if mode == "Image":

    file = st.file_uploader("Upload Image", ["jpg","jpeg","png"])

    if file:

        img = np.array(Image.open(file).convert("RGB"))

        st.image(img, use_container_width=True)

        predictions = []
        pred_weights = []

        for name, model in models.items():

            if name == "cnn_lstm":
                continue

            try:

                processed = preprocess(img, name, model)

                pred = model.predict(processed, verbose=0)[0][0]

                predictions.append(pred)
                pred_weights.append(model_weights.get(name, 1.0))

                st.write(f"{name} probability: {pred:.4f}")

            except:
                pass


        if predictions:

            avg, real_votes, fake_votes, final = ensemble_decision(
                predictions, image_threshold_slider, ensemble_method, weights=pred_weights
            )

            st.subheader("Final Result")

            st.write(f"Average Probability: {avg:.4f}")
            st.write(f"Real Votes: {real_votes}  |  Fake Votes: {fake_votes}")
            
            # Diagnostic info
            with st.expander("📊 Model Statistics"):
                st.write(f"Min: {np.min(predictions):.4f} | Max: {np.max(predictions):.4f} | Std: {np.std(predictions):.4f}")
                st.write(f"Threshold used: {image_threshold_slider}")
                st.bar_chart({"Model Predictions": predictions})

            if final == 1:
                st.success("✅ REAL Image")
            else:
                st.error("⚠️ FAKE Image")


# =============================
# VIDEO DETECTION
# =============================
if mode == "Video":

    file = st.file_uploader("Upload Video", ["mp4","avi","mov"])

    if file:

        st.video(file)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        temp_file.write(file.read())

        temp_path = temp_file.name

        temp_file.close()

        frames = extract_frames(temp_path)

        try:
            os.remove(temp_path)
        except:
            pass


        if frames:

            predictions = []
            pred_weights = []

            video_fake_models = []
            for name, model in models.items():

                try:

                    if name == "cnn_lstm":

                        p = predict_cnn_lstm(frames, model)

                        if p is not None:

                            predictions.append(p)
                            pred_weights.append(model_weights.get(name, 1.0))

                            st.write(f"{name} probability: {p:.4f}")

                    else:

                        model_preds = []

                        for f in frames:

                            processed = preprocess(f, name, model)

                            p = model.predict(processed, verbose=0)[0][0]

                            model_preds.append(p)

                        avg_model = np.mean(model_preds)
                        med_model = np.median(model_preds)

                        if video_fake_override(model_preds, threshold=video_frame_fake_threshold, min_fraction=video_frame_fake_fraction):
                            video_fake_models.append(name)

                        predictions.append(avg_model)
                        pred_weights.append(model_weights.get(name, 1.0))

                        st.write(f"{name} avg probability: {avg_model:.4f}")
                        st.write(f"{name} median probability: {med_model:.4f}")

                except:
                    pass


            if predictions:
                avg, real_votes, fake_votes, final = ensemble_decision(
                    predictions, video_threshold_slider, video_ensemble_method, weights=pred_weights
                )

                override_fake = len(video_fake_models) >= 2 and avg < video_threshold_slider
                if override_fake:
                    final = 0

                st.write(f"Average Probability: {avg:.4f}")
                st.write(f"Real Votes: {real_votes}  |  Fake Votes: {fake_votes}")

                if override_fake:
                    st.warning(
                        "⚠️ Fake override triggered by multiple suspicious models: "
                        + ", ".join(video_fake_models)
                    )
                
                # Diagnostic info
                with st.expander("📊 Model Statistics"):
                    st.write(f"Min: {np.min(predictions):.4f} | Max: {np.max(predictions):.4f} | Std: {np.std(predictions):.4f}")
                    st.write(f"Threshold used: {video_threshold_slider}")
                    st.write(f"Video fake override models: {video_fake_models}")
                    st.bar_chart({"Model Predictions": predictions})

                if final == 1:
                    st.success("✅ REAL Video")
                else:
                    st.error("⚠️ FAKE Video")