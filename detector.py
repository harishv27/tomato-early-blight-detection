"""Model wrapper: lesion segmentation + Grad-CAM explanation for one leaf image.

The model is a MobileNetV2 U-Net. Input: 256x256 RGB scaled to
[-1, 1]. Output: 256x256 map with the probability that each pixel is an early
blight lesion.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "early_blight_unet.keras")

IMG_SIZE = 256             # model input size
PIXEL_THRESHOLD = 0.5      # probability above which a pixel counts as lesion
HEALTHY_MAX_AREA = 0.2     # % of the image; below this the leaf is reported healthy
MODERATE_MIN_AREA = 3.0    # % of the image; severity bands for early blight
SEVERE_MIN_AREA = 10.0
LOW_CONFIDENCE = 60.0      # % below which the user is asked to retake the photo
GRADCAM_LAYER = "conv2d_7"  # 128x128 decoder feature maps used for Grad-CAM

LESION_COLOR = np.array([255, 40, 40], dtype=np.float32)

# compile=False: the custom loss/metrics (total_loss, dice_coef, iou) are only needed for training
model = keras.saving.load_model(MODEL_PATH, compile=False)
grad_model = keras.Model(model.input, [model.get_layer(GRADCAM_LAYER).output, model.output])


@tf.function
def _forward(x):
    """One pass that returns the lesion mask and its Grad-CAM map.

    Grad-CAM for segmentation: the score is the summed probability of the
    predicted lesion pixels, and each feature map is weighted by the mean
    gradient of that score.
    """
    with tf.GradientTape() as tape:
        features, mask = grad_model(x, training=False)
        lesion = tf.cast(mask > PIXEL_THRESHOLD, tf.float32)
        score = tf.reduce_sum(mask * lesion)
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return mask[0, ..., 0], cam[0]


def _resize_map(values, size):
    """Resize a 0..1 float map to a PIL size, returning floats 0..1."""
    img = Image.fromarray((values * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _jet(v):
    """Blue -> green -> yellow -> red colour map for values 0..1."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255


def lesion_overlay(image, mask):
    """Tint the predicted lesion pixels red on the original image."""
    lesion = _resize_map(mask, image.size) > PIXEL_THRESHOLD
    out = np.asarray(image, dtype=np.float32).copy()
    out[lesion] = 0.45 * out[lesion] + 0.55 * LESION_COLOR
    return Image.fromarray(out.astype(np.uint8))


def heatmap_overlay(image, cam):
    """Blend the Grad-CAM heat map over the image; cold regions stay untouched."""
    cam = _resize_map(cam, image.size)
    alpha = np.clip(cam * 1.4, 0, 0.75)[..., None]
    out = (1 - alpha) * np.asarray(image, dtype=np.float32) + alpha * _jet(cam)
    return Image.fromarray(out.astype(np.uint8))


def analyse(image):
    """Run the model on a PIL RGB image and return the diagnosis and explanation maps."""
    started = time.perf_counter()
    resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0   # MobileNetV2 scaling [-1, 1]
    mask, cam = (t.numpy() for t in _forward(tf.constant(x[None])))

    lesion = mask > PIXEL_THRESHOLD
    area = float(lesion.mean() * 100)
    diseased = area >= HEALTHY_MAX_AREA

    if diseased:
        confidence = float(mask[lesion].mean() * 100)
        severity = "Severe" if area >= SEVERE_MIN_AREA else "Moderate" if area >= MODERATE_MIN_AREA else "Mild"
    else:
        confidence = float((1 - mask.max()) * 100) if not lesion.any() else float((1 - mask[lesion].mean()) * 100)
        severity = None

    return {
        "disease_key": "early_blight" if diseased else "healthy",
        "diseased": diseased,
        "severity": severity,
        "lesion_area": round(area, 2),
        "confidence": round(confidence, 1),
        "low_confidence": confidence < LOW_CONFIDENCE,
        "mask": mask,
        "cam": cam,
        "inference_ms": round((time.perf_counter() - started) * 1000),
    }


# Warm up so the first real request does not pay the graph-tracing cost.
_forward(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))
