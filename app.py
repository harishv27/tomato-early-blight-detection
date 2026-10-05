"""Explainable Tomato Early Blight Detection - Flask web application.

Flow: leaf image (camera or gallery) -> preprocessing -> MobileNetV2 U-Net
lesion segmentation -> Grad-CAM explanation -> advisory from the SQLite
knowledge base -> result dashboard in the browser.

Run:  .venv/bin/python app.py   then open http://127.0.0.1:5050
"""
import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

import detector
import knowledge

MAX_DISPLAY_SIDE = 800    # result images are drawn on a copy no larger than this

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
knowledge.init_db()


def to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Please choose a leaf image."), 400
    try:
        image = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a readable image."), 400

    image.thumbnail((MAX_DISPLAY_SIDE, MAX_DISPLAY_SIDE))
    result = detector.analyse(image)
    mask, cam = result.pop("mask"), result.pop("cam")
    advisory = knowledge.get_advisory(result["disease_key"])

    return jsonify(
        **result,
        label=advisory["name"],
        advisory=advisory,
        original=to_data_url(image),
        lesions=to_data_url(detector.lesion_overlay(image, mask)),
        heatmap=to_data_url(detector.heatmap_overlay(image, cam)) if result["diseased"] else None,
    )


@app.route("/health")
def health():
    return jsonify(status="ok", model=detector.model.name, input_size=detector.IMG_SIZE)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 10 MB."), 413


if __name__ == "__main__":
    # host 0.0.0.0 lets a phone on the same Wi-Fi open the page; use 127.0.0.1 to keep it local only
    app.run(host="127.0.0.1", port=5050, debug=False)
