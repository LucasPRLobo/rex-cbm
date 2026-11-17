"""Load the pretrained CUB Concept Bottleneck Models on DEVICE."""
import torch

from run_cbm import load_checkpoint
from data_setup import DEVICE


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
