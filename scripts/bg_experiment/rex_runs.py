"""ReX on the composites: one file per image, resumable, can be split across GPUs with --shard."""
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from run_cbm import load_test_records
from data_setup import CLASSES, DEVICE, EXPERIMENTS, MODEL_SHAPE, build_composites, get_input, tensor_to_pil
from models import load_model


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
    swap = swap[swap.background.isin(["original", "grey", "black", "white", "blur"])]
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
        names, batch = build_composites(img, mask, EXPERIMENTS["rex"])
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
