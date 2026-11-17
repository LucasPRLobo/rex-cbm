"""Model-only experiments (swap, brightness, mask, places): no ReX, minutes on a GPU."""
import time

import numpy as np
import pandas as pd
import torch

from run_cbm import load_test_records
from data_setup import (BIRD_TYPE, CLASSES, EXPERIMENTS, MASK_K, background_brightness, build_composites,
                        get_input, load_places, resize_mask)
from models import load_model


@torch.no_grad()
def run_model_only(kind, model_names, every, limit, out_dir):
    records = load_test_records()
    places = load_places() if kind == "places" else None
    idx_list = list(range(0, len(records), every))[:limit]
    skipped = []
    models = {name: load_model(name)[0] for name in model_names}
    rows = {name: [] for name in model_names}
    start = time.time()

    for n, idx in enumerate(idx_list, 1):
        record = records[idx]
        img, mask = get_input(record)
        if not mask.any():  # no bird pixels in the segmentation: nothing to paste
            skipped.append(idx)
            continue
        names, batch = build_composites(img, mask, EXPERIMENTS[kind], places)
        brightness = [background_brightness(x, mask) for x in batch]  # always over the original non-bird region
        label = record["class_label"]
        bird_frac = mask.float().mean().item()
        mask_frac = {name: resize_mask(mask, k).float().mean().item() for name, k in MASK_K.items() if name in names}

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
                    "bg_category": bg.split("/")[0],
                    "bg_brightness": brightness[k],
                    "bird_frac": bird_frac,
                    "mask_k": MASK_K.get(bg, 0),                  # mask experiment: pixels grown (+) / shrunk (-)
                    "mask_frac": mask_frac.get(bg, bird_frac),    # share of the image taken from the original
                    "true_label": label,
                    "true_class": CLASSES[label],
                    "bird_type": BIRD_TYPE[label],
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

    if skipped:
        print(f"  skipped {len(skipped)} images with an empty bird mask: {skipped[:10]}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        df = pd.DataFrame(rows[name])
        df.to_pickle(out_dir / f"{kind}_{name}.pkl")
        df.drop(columns="concepts").to_csv(out_dir / f"{kind}_{name}.csv", index=False)
        print(f"wrote {out_dir / f'{kind}_{name}.pkl'} ({len(df)} rows)", flush=True)
