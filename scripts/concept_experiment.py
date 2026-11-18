"""Run ReX causal explanations on individual concepts (or class outputs) of a CBM.

Concept k is presented to ReX as a 2-class problem [concept off, concept on]; only
concepts the CBM predicts as ON are explained (the ReX target is then class 1).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from rex_xai.lib import ReX
from rex_xai.responsibility.prediction import Prediction

from scripts.run_cbm import CBM, ConceptView, load_image, image_path, load_test_records
from .frozen_concept import FrozenClassView

MODEL_SHAPE = ("N", 3, 299, 299)


# --------------------------------------------------------------------------- #
# Prediction function
# --------------------------------------------------------------------------- #
def make_prediction_function(logits_fn):
    """ReX prediction function for ANY model that maps images -> (B, C) logits.

    concept k      : logits_fn = ConceptView(cbm, k)          (C = 2: [off, on])
    class output   : logits_fn = cbm.class_logits             (C = 200)
    frozen concepts: logits_fn = FrozenClassView(cbm, idx, vals)

    ReX calls f(mutants, target) -> list[Prediction], and f(mutants, raw=True)
    -> (B, C) probabilities (used by the insertion/deletion evaluation).
    """
    @torch.no_grad()
    def prediction_function(mutants, target=None, raw=False):
        probs = F.softmax(logits_fn(mutants), dim=1)
        if raw:
            return probs
        conf, cls = probs.max(dim=1)
        preds = []
        for i in range(probs.shape[0]):
            t_conf = probs[i, target.classification].item() if target is not None else None
            preds.append(
                Prediction(
                    pred=cls[i].item(),
                    conf=conf[i].item(),
                    target=target,           # Prediction object; class is extracted inside
                    target_confidence=t_conf,
                )
            )
        return preds

    return prediction_function


# --------------------------------------------------------------------------- #
# Concept selection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def active_concepts(cbm, record, device="cuda", threshold=0.5):
    """Indices of concepts the CBM predicts as ON, plus all concept probabilities."""
    x = load_image(record).unsqueeze(0).to(device)
    p = torch.sigmoid(cbm.concept_logits(x))[0]
    return (p > threshold).nonzero().flatten().tolist(), p.cpu().numpy()


@torch.no_grad()
def inactive_concepts(cbm, record, device="cuda", threshold=0.5):
    """Indices of concepts predicted OFF, closest to firing first, plus all probabilities."""
    x = load_image(record).unsqueeze(0).to(device)
    p = torch.sigmoid(cbm.concept_logits(x))[0]
    idx = (p < threshold).nonzero().flatten()
    idx = idx[p[idx].argsort(descending=True)]
    return idx.tolist(), p.cpu().numpy()


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def crop_image(img, resol):
    return transforms.CenterCrop(resol)(img)


def get_bird_mask(record, resol=299, device="cpu"):
    """Whole-bird segmentation mask, cropped exactly like the model input."""
    segmentation_path = Path(
        str(image_path(record)).replace("\\images\\", "\\segmentations\\")
    ).with_suffix(".png")
    seg_img = Image.open(segmentation_path).convert("L")
    cropped = torch.from_numpy(np.array(crop_image(seg_img, resol)))
    return (cropped > 127).to(device)


def to_tensor(img):
    """Same normalisation as the CBM training pipeline (apply AFTER cropping)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
    ])(img)


# --------------------------------------------------------------------------- #
# ReX setup
# --------------------------------------------------------------------------- #
def setup_rex(cbm, k, record, device="cuda", iters=20, search_limit=1000,
              batch_size=64, view=None):
    """Build a ReX object ready to run.

    Default: explain concept k. For a class-level run pass `view`, an nn.Module or
    callable mapping images -> (B, C) logits (e.g. FrozenClassView); k is then unused.
    """
    logits_fn = view if view is not None else ConceptView(cbm, k)
    rex = ReX(
        model=cbm,
        model_shape=MODEL_SHAPE,
        device=device,
        mode="RGB",
        prediction_function=make_prediction_function(logits_fn),
    )
    image = rex.set_rgb_image(image_path(record))
    cropped_img = crop_image(image, 299)

    rex.args.iters = iters
    rex.args.search_limit = search_limit
    rex.args.batch_size = batch_size           # ReX default is 1, which is very slow
    rex.set_input(cropped_img)
    rex.set_transformed_data(to_tensor(cropped_img).unsqueeze(0).to(device))
    rex.set_target()
    return rex, image


# --------------------------------------------------------------------------- #
# Multi-seed running and aggregation
# --------------------------------------------------------------------------- #
def to_2d(mask):
    """(1,3,H,W) or (3,H,W) bool tensor -> (H,W), union over leading dims."""
    while mask.dim() > 2:
        mask = mask.any(0)
    return mask


def run_rex_seeds(rex, seeds, threshold=0.0):
    """Run ReX once per seed. Returns (masks, stats, responsibility maps) keyed by seed."""
    masks, stats, resps = {}, {}, {}
    rex.args.minimum_confidence_threshold = threshold
    target_cls = rex.data.target.classification

    for seed in seeds:
        rex.args.seed = seed                    # applied inside calculate_responsibility
        rex.calculate_responsibility().generate_explanation_object()
        stats[seed] = dict(rex.stats)
        resps[seed] = np.array(rex.maps.get(target_cls))
        rex.extract_sufficient_explanation()
        masks[seed] = to_2d(rex.explanation.sufficiency_mask).clone()

    return masks, stats, resps


def aggregate_seeds(masks, stats, majority=0.5):
    """Collapse per-seed ReX outputs into one result.

    masks: {seed: bool tensor (H, W)}    stats: {seed: dict}
    """
    stack = torch.stack([m.float().cpu() for m in masks.values()])   # (S, H, W)
    freq = stack.mean(0)                    # share of seeds that picked each pixel
    consensus = freq >= majority            # majority-vote mask

    b = stack.bool()
    ious = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            union = (b[i] | b[j]).sum().item()
            ious.append((b[i] & b[j]).sum().item() / union if union else 1.0)

    mean_stats, std_stats = {}, {}
    for key in next(iter(stats.values())):
        vals = [s[key] for s in stats.values() if key in s]
        numeric = all(isinstance(v, (int, float, np.integer, np.floating))
                      and not isinstance(v, bool) for v in vals)
        if numeric:
            mean_stats[key] = float(np.mean(vals))
            std_stats[key] = float(np.std(vals))
        else:
            mean_stats[key] = vals[0]

    return {
        "freq": freq.numpy(),
        "consensus": consensus.numpy(),
        "mean_area": float(b.float().mean(dim=(1, 2)).mean()),
        "consensus_area": float(consensus.float().mean()),
        "pairwise_iou": float(np.mean(ious)) if ious else 1.0,
        "stats_mean": mean_stats,
        "stats_std": std_stats,
    }


# --------------------------------------------------------------------------- #
# Explain every active concept for one image
# --------------------------------------------------------------------------- #
def explain_all_active(cbm, record, device="cuda", concepts=None,
                       seeds=(10, 50, 100), **rex_kwargs):
    """Explain every active concept (or the subset `concepts`) for one image.

    rex_kwargs (iters, search_limit, batch_size) are passed to setup_rex.
    """
    active, probs = active_concepts(cbm, record, device)
    todo = active if concepts is None else [k for k in concepts if k in active]
    results = {}

    for k in todo:
        try:
            rex, _ = setup_rex(cbm, k, record, device, **rex_kwargs)
            target = rex.data.target

            if target.classification != 1:      # ReX disagrees with active_concepts
                print(f"concept {k:3d}  skipped: ReX target is 'off' (p={probs[k]:.3f})")
                continue
            if abs(target.confidence - probs[k]) > 1e-2:
                print(f"concept {k:3d}  WARNING: rex_conf={target.confidence:.3f} "
                      f"!= p={probs[k]:.3f} (check preprocessing)")

            masks, stats, resps = run_rex_seeds(rex, seeds)
            agg = aggregate_seeds(masks, stats)
            agg["responsibility"] = np.mean(list(resps.values()), axis=0)
        except Exception as e:
            print(f"concept {k:3d}  failed: {e}")
            continue

        agg.update(concept=k, prob=float(probs[k]), rex_conf=float(target.confidence))
        results[k] = agg
        print(f"concept {k:3d}  p={probs[k]:.3f}  rex_conf={target.confidence:.3f}  "
              f"area={agg['mean_area']:.3%}  seed_iou={agg['pairwise_iou']:.2f}")

    return results