"""Background-swap experiment on the CUB Concept Bottleneck Models.

Pastes each test bird (CUB segmentation mask) onto new backgrounds and records the prediction,
the true-class confidence and all 112 concept probabilities. Optionally runs ReX on the composites.

  python scripts/bg_swap.py swap --models joint independent sequential         # original/grey/black/white/blur
  python scripts/bg_swap.py brightness --models joint independent sequential   # 7 grey levels, black -> white
  python scripts/bg_swap.py rex --model joint --n-changed 400 --n-stable 200   # ReX on a subset (slow)
  python scripts/bg_swap.py summary                                            # print the result tables

swap/brightness take minutes on a GPU. rex takes ~1-2 min per image, saves one file per image and
skips finished images, so it can be stopped and rerun. Use --shard i/n to split it across GPUs.
Results go to outputs/bg_swap/ (change with --out).
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # ReX prints a progress bar per call; must be set before tqdm loads

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from run_cbm import DATA, ROOT, SELECTED_ATTRIBUTES, TEST_TRANSFORM, image_path, load_checkpoint, load_test_records

warnings.filterwarnings("ignore", message="source code of class")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOL = 299
MODEL_SHAPE = ["N", 3, RESOL, RESOL]
CLASSES = [line.split(" ", 1)[1] for line in (DATA / "CUB_200_2011" / "classes.txt").read_text().splitlines()]
_ATTRS = (DATA / "CUB_200_2011" / "attributes" / "attributes.txt").read_text().splitlines()
CONCEPT_NAMES = [_ATTRS[i].split(" ", 1)[1] for i in SELECTED_ATTRIBUTES]


# ---------- data ----------

def get_input(record):
    """Normalised [3, 299, 299] image tensor and bool [299, 299] bird mask (CUB majority vote), on DEVICE."""
    path = image_path(record)
    img = TEST_TRANSFORM(Image.open(path).convert("RGB")).to(DEVICE)
    seg_path = Path(str(path).replace("/images/", "/segmentations/")).with_suffix(".png")
    seg = TF.center_crop(Image.open(seg_path).convert("L"), RESOL)
    mask = torch.from_numpy(np.array(seg) > 127).to(DEVICE)
    return img, mask


def to_normalised(pixel_value):
    """Pixel value in [0, 1] -> value after Normalize(mean=0.5, std=2)."""
    return (pixel_value - 0.5) / 2


def tensor_to_pil(x):
    return TF.to_pil_image((x.detach().cpu() * 2 + 0.5).clamp(0, 1))


def build_composites(img, mask, kind):
    """Return (names, [K, 3, 299, 299] batch). The first entry is always the original image."""
    if kind == "swap":
        backgrounds = {
            "grey": torch.full_like(img, to_normalised(0.5)),
            "black": torch.full_like(img, to_normalised(0.0)),
            "white": torch.full_like(img, to_normalised(1.0)),
            "blur": TF.gaussian_blur(img, kernel_size=51, sigma=20),
        }
    elif kind == "brightness":
        backgrounds = {f"level_{p:.2f}": torch.full_like(img, to_normalised(p)) for p in np.linspace(0, 1, 7)}
    elif kind == "rex":
        backgrounds = {
            "black": torch.full_like(img, to_normalised(0.0)),
            "white": torch.full_like(img, to_normalised(1.0)),
            "blur": TF.gaussian_blur(img, kernel_size=51, sigma=20),
        }  # grey is skipped: ReX hides pixels by setting them to grey, so it cannot test a grey background
    else:
        raise ValueError(kind)
    names = ["original"] + list(backgrounds)
    batch = torch.stack([img] + [torch.where(mask, img, bg) for bg in backgrounds.values()])
    return names, batch


def background_brightness(x, mask):
    """Mean pixel value (0-1) of the non-bird region."""
    pixels = (x * 2 + 0.5).clamp(0, 1)[:, ~mask]
    return pixels.mean().item() if pixels.numel() else float("nan")


# ---------- models ----------

def load_model(name):
    """Return (fn, module): fn maps a [N, 3, 299, 299] batch to (class_logits, concept_logits)."""
    if name == "joint":
        joint = load_checkpoint("Joint0.01Model__Seed1").to(DEVICE)

        def fn(x):
            out = joint(x)
            return out[0], torch.cat(out[1:], dim=1)
        return fn, joint

    concept_model = load_checkpoint("ConceptModel__Seed1").to(DEVICE)
    if name == "independent":  # head trained on true concepts, so it takes probabilities
        head = load_checkpoint("IndependentModel_WithVal___Seed1").to(DEVICE)

        def fn(x):
            c = torch.cat(concept_model(x), dim=1)
            return head(torch.sigmoid(c)), c
    elif name == "sequential":  # head trained on predicted logits
        head = load_checkpoint("SequentialModel_WithVal__Seed1").to(DEVICE)

        def fn(x):
            c = torch.cat(concept_model(x), dim=1)
            return head(c), c
    else:
        raise ValueError(name)
    return fn, concept_model


# ---------- swap / brightness: model only ----------

@torch.no_grad()
def run_model_only(kind, model_names, every, limit, out_dir):
    records = load_test_records()
    idx_list = list(range(0, len(records), every))[:limit]
    models = {name: load_model(name)[0] for name in model_names}
    rows = {name: [] for name in model_names}
    start = time.time()

    for n, idx in enumerate(idx_list, 1):
        record = records[idx]
        img, mask = get_input(record)
        names, batch = build_composites(img, mask, kind)
        brightness = [background_brightness(x, mask) for x in batch]
        label = record["class_label"]
        bird_frac = mask.float().mean().item()

        for name, fn in models.items():
            class_logits, concept_logits = fn(batch)
            probs = torch.softmax(class_logits, dim=1)
            concepts = torch.sigmoid(concept_logits)
            top_conf, pred = probs.max(dim=1)
            flipped = ((concepts > 0.5) != (concepts[0] > 0.5)).sum(dim=1)
            change = (concepts - concepts[0]).abs().mean(dim=1)
            for k, bg in enumerate(names):
                rows[name].append({
                    "idx": idx,
                    "background": bg,
                    "bg_brightness": brightness[k],
                    "bird_frac": bird_frac,
                    "true_label": label,
                    "true_class": CLASSES[label],
                    "pred_label": pred[k].item(),
                    "pred_class": CLASSES[pred[k].item()],
                    "top_conf": top_conf[k].item(),
                    "true_conf": probs[k, label].item(),
                    "correct": pred[k].item() == label,
                    "n_flipped": flipped[k].item(),
                    "mean_change": change[k].item(),
                    "concepts": concepts[k].cpu().numpy().astype(np.float32),
                })
        if n % 200 == 0 or n == len(idx_list):
            print(f"  {kind}: {n}/{len(idx_list)} images, {time.time() - start:.0f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        df = pd.DataFrame(rows[name])
        df.to_pickle(out_dir / f"{kind}_{name}.pkl")
        df.drop(columns="concepts").to_csv(out_dir / f"{kind}_{name}.csv", index=False)
        print(f"wrote {out_dir / f'{kind}_{name}.pkl'} ({len(df)} rows)", flush=True)


# ---------- ReX ----------

def make_prediction_function(fn):
    from rex_xai.responsibility.prediction import from_pytorch_tensor

    def prediction_function(mutants, target=None, raw=False):
        with torch.no_grad():
            class_logits, _ = fn(mutants)
        if raw:
            return torch.softmax(class_logits, dim=1)
        return from_pytorch_tensor(class_logits, target=target)
    return prediction_function


def explain(x, module, prediction_function, threshold, seed):
    """Run ReX on one normalised [3, 299, 299] tensor. Returns (target label, target conf, bool mask, stats)."""
    from rex_xai.input.input_data import Data
    from rex_xai.lib import ReX

    rex = ReX(module, MODEL_SHAPE, DEVICE, "RGB", prediction_function)
    rex.data = Data(tensor_to_pil(x), MODEL_SHAPE, DEVICE, mode="RGB")  # what set_rgb_image does, minus the file
    rex.set_transformed_data(x.unsqueeze(0))
    rex.set_target()
    rex.args.minimum_confidence_threshold = threshold
    rex.args.seed = seed
    rex.calculate_responsibility().generate_explanation_object()
    stats = dict(rex.stats)
    rex.extract_sufficient_explanation()
    mask = torch.as_tensor(rex.explanation.sufficiency_mask)
    assert (mask[0] == mask[1]).all() and (mask[0] == mask[2]).all()
    target = rex.data.target
    return int(target.classification), float(target.confidence), mask[0].bool().cpu(), stats


def choose_rex_images(model_name, n_changed, n_stable, seed, out_dir):
    """Images whose correctness changes across backgrounds + a random stable set, in a shuffled order
    so that any finished prefix is a representative mix."""
    swap = pd.read_pickle(out_dir / f"swap_{model_name}.pkl")
    changed = swap.groupby("idx")["correct"].nunique() > 1
    rng = np.random.default_rng(seed)
    ch = rng.permutation(changed[changed].index.to_numpy())[:n_changed]
    st = rng.permutation(changed[~changed].index.to_numpy())[:n_stable]
    order = rng.permutation(np.concatenate([ch, st])).tolist()
    print(f"{changed.sum()} changed / {(~changed).sum()} stable images in swap_{model_name}; "
          f"selected {len(ch)} + {len(st)}", flush=True)
    return order, set(ch.tolist())


def run_rex(model_name, images, n_changed, n_stable, threshold, seed, shard, limit, out_dir):
    records = load_test_records()
    if images:
        order, changed = images, set()
    else:
        order, changed = choose_rex_images(model_name, n_changed, n_stable, seed, out_dir)
    shard_i, shard_n = map(int, shard.split("/"))
    order = order[shard_i::shard_n][:limit]

    rex_dir = out_dir / f"rex_{model_name}"
    rex_dir.mkdir(parents=True, exist_ok=True)
    todo = [idx for idx in order if not (rex_dir / f"{idx:05d}.pkl").exists()]
    print(f"shard {shard}: {len(order)} images, {len(order) - len(todo)} already done, {len(todo)} to run", flush=True)

    fn, module = load_model(model_name)
    prediction_function = make_prediction_function(fn)
    start = time.time()

    for n, idx in enumerate(todo, 1):
        t0 = time.time()
        record = records[idx]
        img, mask = get_input(record)
        names, batch = build_composites(img, mask, "rex")
        label = record["class_label"]
        bird_frac = mask.float().mean().item()
        rows = []
        for bg, x in zip(names, batch):
            target, conf, expl, stats = explain(x, module, prediction_function, threshold, seed)
            size = expl.float().mean().item()
            share = (expl & mask.cpu()).sum().item() / max(expl.sum().item(), 1)
            rows.append({
                "idx": idx,
                "background": bg,
                "group": "changed" if idx in changed else ("stable" if not images else "given"),
                "bird_frac": bird_frac,
                "true_label": label,
                "true_class": CLASSES[label],
                "target_label": target,
                "target_class": CLASSES[target],
                "target_conf": conf,
                "correct": target == label,
                "size": size,
                "bird_share": share,
                "enrichment": share / bird_frac if bird_frac > 0 else float("nan"),
                "explanation": expl.numpy(),
                "stats": stats,
                "threshold": threshold,
                "seed": seed,
            })
        tmp = rex_dir / f"{idx:05d}.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(rows, f)
        os.replace(tmp, rex_dir / f"{idx:05d}.pkl")  # only complete files count as done

        elapsed = time.time() - start
        eta_h = elapsed / n * (len(todo) - n) / 3600
        print(f"[{n}/{len(todo)}] idx {idx}: {time.time() - t0:.0f}s, eta {eta_h:.1f}h", flush=True)


def load_rex(out_dir, model_name):
    files = sorted((out_dir / f"rex_{model_name}").glob("*.pkl"))
    rows = [row for f in files for row in pickle.load(open(f, "rb"))]
    return pd.DataFrame(rows)


# ---------- summary ----------

def summary(out_dir):
    pd.set_option("display.width", 200)
    for kind in ("swap", "brightness"):
        for path in sorted(out_dir.glob(f"{kind}_*.pkl")):
            df = pd.read_pickle(path)
            print(f"\n== {path.stem} ({df.idx.nunique()} images)")
            cols = ["correct", "true_conf", "n_flipped", "mean_change", "bg_brightness"]
            print(df.groupby("background", sort=False)[cols].mean().round(3))
    for rex_dir in sorted(out_dir.glob("rex_*")):
        df = load_rex(out_dir, rex_dir.name.removeprefix("rex_"))
        if df.empty:
            continue
        print(f"\n== {rex_dir.name} ({df.idx.nunique()} images)")
        cols = ["correct", "size", "bird_share", "enrichment"]
        print(df.groupby(["group", "background"], sort=False)[cols].median().round(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["swap", "brightness", "rex", "summary"])
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
    if args.command in ("swap", "brightness"):
        run_model_only(args.command, args.models, args.every, args.limit, args.out)
    elif args.command == "rex":
        run_rex(args.model, args.images, args.n_changed, args.n_stable, args.threshold, args.seed,
                args.shard, args.limit, args.out)
    else:
        summary(args.out)


if __name__ == "__main__":
    main()
