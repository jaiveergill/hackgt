"""Download the HaGRID images listed in data/gestures/hagrid/labels.json (HaGRID, CC BY-SA 4.0, 30k-sample 384p mirror).

Each image is fetched with one HTTP range request into the 1 GB zip (labels.json stores its offset), so the whole
subset (~15 MB) downloads in seconds without the rest of the archive.

  python scripts/fetch_hagrid.py
"""
import os, sys, json, struct
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "gestures", "hagrid")
URL = "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"


def fetch(url, name, lab):
    dest = os.path.join(DATA, name)
    if os.path.exists(dest):
        return
    off, size = lab["zip_offset"], lab["zip_size"]
    r = requests.get(url, headers={"Range": f"bytes={off}-{off + 30 + 1024 + size}"}, timeout=60)
    if r.status_code != 206:
        raise RuntimeError(f"{name}: server ignored the Range request (HTTP {r.status_code})")
    b = r.content
    if b[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{name}: no zip local header at offset {off}")
    method, = struct.unpack("<H", b[8:10])
    if method != 0:
        raise RuntimeError(f"{name}: zip member is compressed (method {method}); only stored members are supported")
    n, e = struct.unpack("<HH", b[26:30])  # local header: file name + extra field lengths
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b[30 + n + e:30 + n + e + size])


def main():
    labels = json.load(open(os.path.join(DATA, "labels.json")))
    url = requests.head(URL, allow_redirects=True, timeout=30).url  # resolve the CDN redirect once
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda kv: fetch(url, *kv), labels.items()))
    missing = [k for k in labels if not os.path.exists(os.path.join(DATA, k))]
    print(f"{len(labels) - len(missing)}/{len(labels)} images in {os.path.relpath(DATA, ROOT)}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
