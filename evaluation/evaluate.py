"""Send every test image through the running web app and report its performance.

Usage (with the app running on port 5050):
    .venv/bin/python evaluation/evaluate.py

Reads   test_images/<class>/*.jpg
Writes  results.csv   one row per image
        summary.json  confusion matrix, metrics and response times
"""
import csv
import glob
import json
import os
import statistics
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:5050/predict"
HERE = os.path.dirname(os.path.abspath(__file__))
# folder -> expected label (None = disease the model was not trained for)
CLASSES = {
    "early_blight": "Early Blight",
    "healthy": "Healthy",
    "other_late_blight": None,
    "other_septoria": None,
}


def post_image(path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: image/jpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as res:
        out = json.load(res)
    out["response_ms"] = (time.perf_counter() - started) * 1000
    return out


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


rows = []
for folder, expected in CLASSES.items():
    for path in sorted(glob.glob(os.path.join(HERE, "test_images", folder, "*.jpg"))):
        r = post_image(path)
        rows.append({
            "file": os.path.relpath(path, HERE), "folder": folder, "expected": expected or "(not early blight)",
            "predicted": r["label"], "severity": r["severity"] or "", "lesion_area": r["lesion_area"],
            "confidence": r["confidence"], "low_confidence": r["low_confidence"],
            "inference_ms": r["inference_ms"], "response_ms": round(r["response_ms"], 1),
        })

with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Two-class metrics, with Early Blight as the positive class.
main = [r for r in rows if r["folder"] in ("early_blight", "healthy")]
tp = sum(r["folder"] == "early_blight" and r["predicted"] == "Early Blight" for r in main)
fn = sum(r["folder"] == "early_blight" and r["predicted"] == "Healthy" for r in main)
fp = sum(r["folder"] == "healthy" and r["predicted"] == "Early Blight" for r in main)
tn = sum(r["folder"] == "healthy" and r["predicted"] == "Healthy" for r in main)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0

summary = {
    "images": {folder: sum(r["folder"] == folder for r in rows) for folder in CLASSES},
    "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    "accuracy": round((tp + tn) / len(main) * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "specificity": round(tn / (tn + fp) * 100, 2) if tn + fp else 0.0,
    "f1": round(2 * precision * recall / (precision + recall) * 100, 2) if precision + recall else 0.0,
    "severity_counts": {s: sum(r["folder"] == "early_blight" and r["severity"] == s for r in rows) for s in ("Mild", "Moderate", "Severe")},
    "low_confidence_count": sum(bool(r["low_confidence"]) for r in main),
    "out_of_scope_flagged_as_early_blight": {
        folder: sum(r["folder"] == folder and r["predicted"] == "Early Blight" for r in rows)
        for folder in CLASSES if CLASSES[folder] is None
    },
    "inference_ms": {
        "mean": round(statistics.mean(r["inference_ms"] for r in rows), 1),
        "median": statistics.median(r["inference_ms"] for r in rows),
        "p95": percentile([r["inference_ms"] for r in rows], 95),
        "max": max(r["inference_ms"] for r in rows),
    },
    "response_ms": {
        "mean": round(statistics.mean(r["response_ms"] for r in rows), 1),
        "median": round(statistics.median(r["response_ms"] for r in rows), 1),
        "p95": percentile([r["response_ms"] for r in rows], 95),
        "max": max(r["response_ms"] for r in rows),
    },
}
with open(os.path.join(HERE, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
