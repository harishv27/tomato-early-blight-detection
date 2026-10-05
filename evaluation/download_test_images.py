"""Download a random sample of PlantVillage tomato leaf images for testing.

Usage:  .venv/bin/python evaluation/download_test_images.py

Saves the images to evaluation/test_images/<class>/. The sample is fixed by the
random seed, so every run downloads the same files.
Source: https://github.com/spMohanty/PlantVillage-Dataset
"""
import concurrent.futures
import json
import os
import random
import urllib.parse
import urllib.request

REPO = "spMohanty/PlantVillage-Dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
# PlantVillage folder -> (local folder, number of images)
SAMPLE = {
    "Tomato___Early_blight": ("early_blight", 250),
    "Tomato___healthy": ("healthy", 250),
    "Tomato___Late_blight": ("other_late_blight", 75),
    "Tomato___Septoria_leaf_spot": ("other_septoria", 75),
}


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tomato-early-blight"})
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.load(res)


def download(job):
    url, dest = job
    if os.path.exists(dest):
        return True
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except OSError:
            pass
    return False


folders = {d["name"]: d["sha"] for d in api(f"https://api.github.com/repos/{REPO}/contents/raw/color")}
random.seed(42)
jobs = []
for source, (name, count) in SAMPLE.items():
    tree = api(f"https://api.github.com/repos/{REPO}/git/trees/{folders[source]}")
    files = sorted(item["path"] for item in tree["tree"] if item["type"] == "blob")
    os.makedirs(os.path.join(HERE, "test_images", name), exist_ok=True)
    for i, filename in enumerate(random.sample(files, min(count, len(files)))):
        url = f"https://raw.githubusercontent.com/{REPO}/master/raw/color/{source}/{urllib.parse.quote(filename)}"
        jobs.append((url, os.path.join(HERE, "test_images", name, f"{name}_{i:03d}.jpg")))

with concurrent.futures.ThreadPoolExecutor(16) as pool:
    done = sum(pool.map(download, jobs))
print(f"{done} of {len(jobs)} images ready in evaluation/test_images/")
