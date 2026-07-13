"""
Speed cache: mirror outputs/dr_imagefolder/ (symlinks -> 4288x2848 JPEGs) into
outputs/dr_imagefolder_cache/ with images resized to shorter-side 512 px.

Training decodes ~0.4 MP instead of ~12 MP per image (~30x less JPEG-decode work)
with negligible quality loss for 224-px RandomResizedCrop / 256->224 CenterCrop.
Structure, filenames, split and class folders are preserved exactly, so the cache
is a drop-in `data_path` for the notebook.

Run:  python pipeline/build_resized_cache.py [--size 512] [--workers 24]
"""
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SRC = os.path.join(C.OUT_DIR, "dr_imagefolder")
DST = os.path.join(C.OUT_DIR, "dr_imagefolder_cache")


def _resize_one(job):
    src, dst, size = job
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = size / min(w, h)
            if s < 1.0:  # only downscale
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.BICUBIC)
            im.save(dst, "JPEG", quality=95)
        return None
    except Exception as e:  # noqa
        return f"{src}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, help="shorter-side target px")
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    args = ap.parse_args()

    assert os.path.isdir(SRC), f"missing {SRC} -- run materialize_imagefolder.py first"
    jobs = []
    for split in ["train", "val", "test"]:
        for cls in sorted(os.listdir(os.path.join(SRC, split))):
            sd = os.path.join(SRC, split, cls)
            dd = os.path.join(DST, split, cls)
            os.makedirs(dd, exist_ok=True)
            for fn in os.listdir(sd):
                out = os.path.join(dd, fn)
                # resolve symlink target for reading (real bytes)
                src = os.path.realpath(os.path.join(sd, fn))
                if os.path.exists(out):
                    continue  # resumable
                jobs.append((src, out, args.size))

    print(f"resizing {len(jobs)} images -> {DST} (shorter side {args.size}px, {args.workers} workers)")
    if not jobs:
        print("nothing to do (cache already complete)")
        return
    errors, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_resize_one, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r:
                errors.append(r)
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"done: {done} images, {len(errors)} errors")
    for e in errors[:10]:
        print("  [err]", e)

    # verify counts match source
    for split in ["train", "val", "test"]:
        for cls in sorted(os.listdir(os.path.join(SRC, split))):
            n_src = len(os.listdir(os.path.join(SRC, split, cls)))
            n_dst = len(os.listdir(os.path.join(DST, split, cls)))
            assert n_src == n_dst, f"{split}/{cls}: src {n_src} != dst {n_dst}"
    print(f"cache verified, counts match source. data_path = {DST}")


if __name__ == "__main__":
    main()
