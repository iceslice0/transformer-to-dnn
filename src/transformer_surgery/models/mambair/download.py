"""Dataset download for the MambaIR SR path (DIV2K train + Set5 benchmark).

The reference experiment assumed datasets pre-staged under ``/data/datasets``; here we fetch them
on demand into a project data root, mirroring how the Pet loaders auto-download. Sources are the
canonical, no-auth mirrors:

* DIV2K: ``https://data.vision.ee.ethz.ch/cvl/DIV2K/`` (train HR + LR bicubic per scale)
* Set5 (and the other SR benchmarks): the EDSR ``benchmark.tar`` from
  ``https://cv.snu.ac.kr/research/EDSR/benchmark.tar``

Resulting layout under ``root`` (matches the config paths):

    root/DIV2K/DIV2K_train_HR/
    root/DIV2K/DIV2K_train_LR_bicubic/X{scale}/
    root/SRBenchmarks/benchmark/Set5/{HR,LR_bicubic/X{scale}}/
"""

from __future__ import annotations

import os
import ssl
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from typing import Optional

DIV2K_BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
BENCHMARK_URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"


def default_sr_paths(root: str, scale: int) -> dict:
    """Canonical HR/LR directories under ``root`` for the given scale."""
    return {
        "train_hr": os.path.join(root, "DIV2K", "DIV2K_train_HR"),
        "train_lr": os.path.join(root, "DIV2K", "DIV2K_train_LR_bicubic", f"X{scale}"),
        "val_hr": os.path.join(root, "SRBenchmarks", "benchmark", "Set5", "HR"),
        "val_lr": os.path.join(root, "SRBenchmarks", "benchmark", "Set5", "LR_bicubic", f"X{scale}"),
    }


def _download_attempt(url: str, tmp: str, *, timeout: int) -> None:
    """One download pass; resumes ``tmp`` via HTTP Range when a partial file exists."""
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": "transformer-surgery/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted URLs)
        status = getattr(resp, "status", resp.getcode())
        mode = "ab"
        if existing and status != 206:  # server ignored Range -> restart from scratch
            existing = 0
            mode = "wb"
        content_len = int(resp.headers.get("Content-Length", 0))
        total = existing + content_len if content_len else 0
        done = existing
        step = max(1, total // 20) if total else 0
        next_mark = (done // step + 1) * step if step else 0
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if step and done >= next_mark:
                    print(f"    {done / 1e6:.0f} MB / {total / 1e6:.0f} MB", flush=True)
                    next_mark += step


def _download(url: str, dest: str, *, attempts: int = 6, timeout: int = 60) -> None:
    """Download with retries + resume; robust to intermittent TLS/connection drops."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {url}", flush=True)
    last_err: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            _download_attempt(url, tmp, timeout=timeout)
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, ssl.SSLError, OSError, TimeoutError) as e:
            last_err = e
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if attempt < attempts:
                backoff = min(30, 2 ** attempt)
                print(
                    f"    attempt {attempt}/{attempts} failed ({type(e).__name__}); "
                    f"{got / 1e6:.0f} MB kept, retrying in {backoff}s ...",
                    flush=True,
                )
                time.sleep(backoff)
    raise RuntimeError(
        f"Failed to download {url} after {attempts} attempts: {last_err}. "
        f"Re-running the download resumes from the partial file ({tmp}); "
        f"or download it manually to {dest}."
    )


def _extract_zip(archive: str, dest_dir: str) -> None:
    print(f"  extracting {os.path.basename(archive)} -> {dest_dir}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def _extract_tar(archive: str, dest_dir: str) -> None:
    print(f"  extracting {os.path.basename(archive)} -> {dest_dir}", flush=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir)


def _fetch_archive(url: str, cache_dir: str, dest_dir: str, kind: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    archive = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.isfile(archive):
        _download(url, archive)
    (_extract_tar if kind == "tar" else _extract_zip)(archive, dest_dir)


def ensure_sr_datasets(root: str, scale: int, *, keep_archives: bool = False) -> dict:
    """Download/extract any missing DIV2K-train / Set5 pieces; returns the resolved paths."""
    root = os.path.abspath(root)
    paths = default_sr_paths(root, scale)
    cache_dir = os.path.join(root, "_downloads")

    if not os.path.isdir(paths["train_hr"]):
        _fetch_archive(f"{DIV2K_BASE}/DIV2K_train_HR.zip", cache_dir, os.path.join(root, "DIV2K"), "zip")
    if not os.path.isdir(paths["train_lr"]):
        _fetch_archive(
            f"{DIV2K_BASE}/DIV2K_train_LR_bicubic_X{scale}.zip", cache_dir, os.path.join(root, "DIV2K"), "zip"
        )
    if not os.path.isdir(paths["val_hr"]) or not os.path.isdir(paths["val_lr"]):
        _fetch_archive(BENCHMARK_URL, cache_dir, os.path.join(root, "SRBenchmarks"), "tar")

    for key, path in paths.items():
        if not os.path.isdir(path):
            raise RuntimeError(f"SR dataset dir still missing after download: {key}={path}")
    if not keep_archives and os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            try:
                os.remove(os.path.join(cache_dir, name))
            except OSError:
                pass
    return paths


def main(argv: Optional[list] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download DIV2K train + Set5 benchmark for MambaIR SR.")
    ap.add_argument("--root", default="./data", help="Data root (default ./data).")
    ap.add_argument("--scale", type=int, default=2, help="SR scale to fetch LR for (default 2).")
    ap.add_argument("--keep-archives", action="store_true", help="Keep downloaded zips/tar.")
    args = ap.parse_args(argv)
    paths = ensure_sr_datasets(args.root, args.scale, keep_archives=args.keep_archives)
    print("SR datasets ready:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
