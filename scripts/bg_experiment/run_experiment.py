"""Background-swap experiment on the CUB Concept Bottleneck Models.

Pastes each test bird (CUB segmentation mask) onto new backgrounds and records the prediction,
the true-class confidence and all 112 concept probabilities. Optionally runs ReX on the composites.

Run from the repo root:

  python scripts/bg_experiment/run_experiment.py swap --models joint independent sequential         # grey/black/white/blur/bird_mean/bg_mean/bg_only
  python scripts/bg_experiment/run_experiment.py brightness --models joint independent sequential   # 7 grey levels, black -> white
  python scripts/bg_experiment/run_experiment.py mask --models joint independent sequential         # black, bird mask grown/shrunk
  python scripts/bg_experiment/run_experiment.py places --models joint independent sequential       # real scenes (get_places.py first)
  python scripts/bg_experiment/run_experiment.py rex --model joint --n-changed 400 --n-stable 200   # ReX on a subset (slow)
  python scripts/bg_experiment/run_experiment.py summary                                            # print the result tables

Backgrounds are defined in COMPOSITES and grouped into experiments in EXPERIMENTS.
The model-only stages take minutes on a GPU. rex takes ~1-2 min per image, saves one file per image and
skips finished images, so it can be stopped and rerun. Use --shard i/n to split it across GPUs.
Results go to outputs/bg_swap/ (change with --out).

Files: data_setup.py (constants, images, composites), models.py (load_model),
model_only.py (run_model_only), rex_runs.py (ReX), this file (command line + summary).
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # ReX prints a progress bar per call; must be set before tqdm loads

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, so `import run_cbm` works

import pandas as pd
import torch

from run_cbm import ROOT
from data_setup import DEVICE
from model_only import run_model_only
from rex_runs import load_rex, run_rex


def summary(out_dir):
    pd.set_option("display.width", 200)
    cols = ["correct", "true_conf", "n_flipped", "mean_change", "bg_brightness"]
    for kind in ("swap", "brightness", "mask"):
        for path in sorted(out_dir.glob(f"{kind}_*.pkl")):
            df = pd.read_pickle(path)
            print(f"\n== {path.stem} ({df.idx.nunique()} images)")
            print(df.groupby("background", sort=False)[cols].mean().round(3))
    for path in sorted(out_dir.glob("places_*.pkl")):
        df = pd.read_pickle(path)
        print(f"\n== {path.stem} ({df.idx.nunique()} images): accuracy, bird type x scene")
        print(df.pivot_table(index="bird_type", columns="bg_category", values="correct", sort=False).round(3))
        print(df.groupby("bg_category", sort=False)[cols].mean().round(3))
    for rex_dir in sorted(out_dir.glob("rex_*")):
        df = load_rex(out_dir, rex_dir.name.removeprefix("rex_"))
        if df.empty:
            continue
        print(f"\n== {rex_dir.name} ({df.idx.nunique()} images)")
        cols = ["correct", "size", "bird_share", "enrichment"]
        print(df.groupby(["group", "background"], sort=False)[cols].median().round(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["swap", "brightness", "mask", "places", "rex", "summary"])
    parser.add_argument("--models", nargs="+", default=["joint"], help="swap/brightness: models to run")
    parser.add_argument("--model", default="joint", help="rex: model to explain")
    parser.add_argument("--every", type=int, default=1, help="swap/brightness: use every n-th test image")
    parser.add_argument("--limit", type=int, default=None, help="only the first n images (for testing)")
    parser.add_argument("--images", type=int, nargs="+", help="rex: explicit test indices instead of selection")
    parser.add_argument("--n-changed", type=int, default=400)
    parser.add_argument("--n-stable", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1", help="rex: run part i of n, e.g. 0/2 and 1/2 on two GPUs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "bg_swap")
    args = parser.parse_args()

    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""), flush=True)
    if args.command in ("swap", "brightness", "mask", "places"):
        run_model_only(args.command, args.models, args.every, args.limit, args.out)
    elif args.command == "rex":
        run_rex(args.model, args.images, args.n_changed, args.n_stable, args.threshold, args.seed,
                args.shard, args.limit, args.out)
    else:
        summary(args.out)


if __name__ == "__main__":
    main()
