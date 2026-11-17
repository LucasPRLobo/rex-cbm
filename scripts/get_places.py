"""Download a few Places365 scenes per category to use as real backgrounds.

Uses the same water/land scene categories as the Waterbirds dataset (Sagawa et al., 2020).
Streams the Places365 validation archive (256x256, ~525 MB) and keeps only the chosen images,
so nothing large is stored.

  python scripts/get_places.py --n 10      # -> data/backgrounds/<category>/*.jpg
"""
import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://data.csail.mit.edu/places/places365"
CATEGORIES = {  # folder name -> Places365 category
    "ocean": "/o/ocean",
    "lake": "/l/lake/natural",
    "bamboo_forest": "/b/bamboo_forest",
    "broadleaf_forest": "/f/forest/broadleaf",
}


def stream_tar(url):
    return tarfile.open(fileobj=urllib.request.urlopen(url), mode="r|*")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="images per category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "backgrounds")
    args = parser.parse_args()

    # 1. File lists: which validation image belongs to which category
    texts = {}
    with stream_tar(f"{BASE}/filelist_places365-standard.tar") as tar:
        for member in tar:
            name = Path(member.name).name
            if name in ("categories_places365.txt", "places365_val.txt"):
                texts[name] = tar.extractfile(member).read().decode()
    cat_id = {line.split()[0]: int(line.split()[1]) for line in texts["categories_places365.txt"].splitlines()}
    by_cat = {}
    for line in texts["places365_val.txt"].splitlines():
        fname, c = line.split()
        by_cat.setdefault(int(c), []).append(fname)

    rng = np.random.default_rng(args.seed)
    wanted = {}  # file name -> folder
    for folder, cat in CATEGORIES.items():
        files = sorted(by_cat[cat_id[cat]])
        for fname in rng.choice(files, size=args.n, replace=False):
            wanted[fname] = folder
    print(f"selected {len(wanted)} images; streaming val_256.tar (~525 MB)...", flush=True)

    # 2. Stream the image archive and keep only the selected files
    for folder in CATEGORIES:
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    found = 0
    with stream_tar(f"{BASE}/val_256.tar") as tar:
        for member in tar:
            fname = Path(member.name).name
            if fname in wanted:
                (args.out / wanted[fname] / fname).write_bytes(tar.extractfile(member).read())
                found += 1
                if found == len(wanted):
                    break
    print(f"saved {found}/{len(wanted)} images to {args.out}")


if __name__ == "__main__":
    main()
