"""Run per-concept ReX explanations over CBM variants and save results for analysis.

Results are written one file per (model, image), so a crash or a Ctrl-C loses nothing
and re-running skips finished images. `merge()` combines them into a single pickle:

    {model_name: {image_id: {"probs": array, "concepts": {concept_idx: result_dict}}}}

Usage:
    python run_concept_experiment.py --n-images 1          # time one image first
    python run_concept_experiment.py --n-images 50
    python run_concept_experiment.py --merge               # write results.pkl
"""
import argparse
import gzip
import pickle
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scripts.run_cbm import load_test_records, CBM
from scripts.concept_experiment import explain_all_concepts

# `concept`, `independent` and `sequential` all use ConceptModel__Seed1 for their
# concept outputs, so their concept-level explanations are identical. Run it once
# (as "concept") plus the two joint models, which have their own concept layers.
DEFAULT_MODELS = ["concept", "joint", "joint_sigmoid"]
SEEDS = (10, 50, 100)


def slim(result, n_seeds):
    """Shrink one concept's result so many images fit on disk."""
    out = dict(result)
    # freq is k/n_seeds; store the integer seed count (uint8) instead of float32
    out["freq_count"] = np.rint(result["freq"] * n_seeds).astype(np.uint8)
    out["n_seeds"] = n_seeds
    out["responsibility"] = np.asarray(result["responsibility"]).astype(np.float16)
    for key in ("freq", "consensus", "rex"):     # consensus = freq_count >= n_seeds / 2
        out.pop(key, None)
    return out


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)                            # atomic: no half-written files


def run(args):  # one file per (model, image); finished files are skipped
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)

    records = load_test_records()
    # CUB test records are usually ordered by class, so take a seeded random sample,
    # not the first N (which would cover only a few species).
    if args.n_images:
        records = random.Random(0).sample(list(records), args.n_images)

    n_inactive = None if args.n_inactive < 0 else args.n_inactive   # -1 = all inactive
    config = dict(seeds=SEEDS, iters=args.iters, search_limit=args.search_limit,
                  batch_size=args.batch_size, n_inactive=n_inactive,
                  inactive_mode=args.inactive_mode, margin=args.margin)

    for model_name in args.models:
        cbm = CBM(model_name, device=device)
        for i, record in enumerate(records):
            image_id = str(record["id"])
            path = out_dir / model_name / f"{image_id}.pkl.gz"
            if path.exists():
                continue
            t0 = time.time()
            try:
                out = explain_all_concepts(
                    cbm, record, device, seeds=SEEDS,
                    n_inactive=n_inactive, inactive_mode=args.inactive_mode,
                    margin=args.margin,
                    iters=args.iters, search_limit=args.search_limit,
                    batch_size=args.batch_size,
                )
                results = out["concepts"]
            except Exception:
                print(f"[{model_name}] image {image_id} FAILED")
                traceback.print_exc()
                continue
            save(path, {
                "model": model_name,
                "id": image_id,
                "config": config,
                "probs": out["probs"],          # all concept probabilities for this image
                "concepts": {k: slim(v, len(SEEDS)) for k, v in results.items()},
            })
            print(f"[{model_name}] {i + 1}/{len(records)}  image {image_id}  "
                  f"{len(results)} concepts  {time.time() - t0:.0f}s")
        del cbm
        if device == "cuda":
            torch.cuda.empty_cache()


def merge(out_dir, merged_path):
    merged = {}
    for model_dir in sorted(Path(out_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        merged[model_dir.name] = {}
        for f in sorted(model_dir.glob("*.pkl.gz")):
            with gzip.open(f, "rb") as fh:
                d = pickle.load(fh)
            merged[model_dir.name][d["id"]] = {"probs": d["probs"], "concepts": d["concepts"]}
    with open(merged_path, "wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {merged_path}: "
          + ", ".join(f"{m}={len(v)} images" for m, v in merged.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--n-images", type=int, default=None,
                   help="random subset size (default: all test images)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--search-limit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-inactive", type=int, default=10,
                   help="inactive concepts to explain per image, closest to firing first "
                        "(0 = none, -1 = all; all is ~4x the cost of active-only)")
    p.add_argument("--inactive-mode", choices=["near_threshold", "off"],
                   default="near_threshold")
    p.add_argument("--margin", type=float, default=1.0,
                   help="near_threshold: logits below the current value to place the threshold")
    p.add_argument("--merge", action="store_true",
                   help="combine per-image files into results.pkl and exit")
    args = p.parse_args()

    if args.merge:
        merge(args.out_dir, Path(args.out_dir) / "results.pkl")
    else:
        run(args)