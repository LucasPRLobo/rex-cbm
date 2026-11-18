"""Run ReX causal explanations on individual concepts (or class outputs) of a CBM.

Concept k is presented to ReX as a 2-class problem [concept off, concept on]; only
concepts the CBM predicts as ON are explained (the ReX target is then class 1).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    p = image_path(record)               # works on Windows and Linux
    segmentation_path = Path(
        *["segmentations" if part == "images" else part for part in p.parts]
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
# Explain concepts for one image
# --------------------------------------------------------------------------- #
class ShiftedConceptView(nn.Module):
    """Concept k with its decision threshold moved to logit `tau` (on iff z > tau)."""

    def __init__(self, cbm, k, tau):
        super().__init__()
        self.cbm, self.k, self.tau = cbm, k, float(tau)

    def forward(self, x):
        z = self.cbm.concept_logits(x)[:, self.k] - self.tau
        return torch.stack([torch.zeros_like(z), z], dim=1)


def _explain_one(cbm, k, record, device, seeds, p, view, expect_cls, expect_conf,
                 **rex_kwargs):
    """Multi-seed ReX for one concept. Returns an aggregated dict, or None if skipped."""
    rex, _ = setup_rex(cbm, k, record, device, view=view, **rex_kwargs)
    target = rex.data.target

    if target.classification != expect_cls:      # ReX disagrees with the probabilities
        print(f"concept {k:3d}  skipped: ReX target is class {target.classification}, "
              f"expected {expect_cls} (p={p:.3f})")
        return None
    if abs(target.confidence - expect_conf) > 1e-2:
        print(f"concept {k:3d}  WARNING: rex_conf={target.confidence:.3f} != "
              f"expected {expect_conf:.3f} (check preprocessing)")

    masks, stats, resps = run_rex_seeds(rex, seeds)
    agg = aggregate_seeds(masks, stats)
    agg["responsibility"] = np.mean(list(resps.values()), axis=0)
    agg.update(concept=k, prob=float(p), rex_conf=float(target.confidence),
               target_class=int(target.classification))
    return agg


def explain_all_concepts(cbm, record, device="cuda", seeds=(10, 50, 100),
                         n_inactive=10, inactive_mode="near_threshold", margin=1.0,
                         concepts=None, **rex_kwargs):
    """Explain every active concept, plus the inactive ones closest to firing.

    Active concepts (p > 0.5): ReX explains why the concept is ON.

    Inactive concepts (p <= 0.5), the `n_inactive` closest to firing first
    (None = all of them, 0 = none; all 112 is ~4x the cost of active-only):
      inactive_mode="near_threshold": the decision threshold is lowered to `margin`
          logits below the concept's current logit, so the concept counts as "on";
          ReX then finds the pixels that keep its evidence from falling further.
          This explains the near-miss evidence, not the concept firing.
      inactive_mode="off": ReX explains why the concept stays OFF. Usually small or
          arbitrary, because masking rarely turns a concept on.

    Returns {"probs": (n_concepts,) array, "concepts": {k: result dict}}; each result
    has "status" ("active"/"inactive") and "mode" fields.
    """
    _, probs = active_concepts(cbm, record, device)        # one forward pass
    order = np.argsort(-probs)
    active = [int(k) for k in np.flatnonzero(probs > 0.5)]
    inactive = [int(k) for k in order if probs[k] <= 0.5]  # closest to firing first

    if concepts is not None:
        keep = set(concepts)
        active = [k for k in active if k in keep]
        inactive = [k for k in inactive if k in keep]
    if n_inactive is not None:
        inactive = inactive[:n_inactive]

    results = {}
    jobs = [(k, "active") for k in active] + [(k, "inactive") for k in inactive]
    for k, status in jobs:
        p = float(probs[k])
        if status == "active":
            mode, view, expect_cls, expect_conf = "on", None, 1, p
        elif inactive_mode == "off":
            mode, view, expect_cls, expect_conf = "off", None, 0, 1.0 - p
        else:
            pc = float(np.clip(p, 1e-6, 1 - 1e-6))
            tau = float(np.log(pc / (1 - pc))) - margin
            mode = "near_threshold"
            view = ShiftedConceptView(cbm, k, tau)
            expect_cls, expect_conf = 1, 1.0 / (1.0 + np.exp(-margin))
        try:
            agg = _explain_one(cbm, k, record, device, seeds, p, view,
                               expect_cls, expect_conf, **rex_kwargs)
        except Exception as e:
            print(f"concept {k:3d}  failed: {e}")
            continue
        if agg is None:
            continue
        agg.update(status=status, mode=mode)
        results[k] = agg
        print(f"concept {k:3d}  {status:8s} {mode:14s} p={p:.3f}  "
              f"area={agg['mean_area']:.3%}  seed_iou={agg['pairwise_iou']:.2f}")

    return {"probs": probs.astype(np.float32), "concepts": results}


def explain_all_active(cbm, record, device="cuda", concepts=None,
                       seeds=(10, 50, 100), **rex_kwargs):
    """Active concepts only (kept for backwards compatibility)."""
    out = explain_all_concepts(cbm, record, device, seeds=seeds, n_inactive=0,
                               concepts=concepts, **rex_kwargs)
    return out["concepts"]