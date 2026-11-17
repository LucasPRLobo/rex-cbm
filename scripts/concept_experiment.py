"""Run ReX causal explanations on individual concepts of a concept bottleneck model.

Each concept k is presented to ReX as a 2-class problem: [concept off, concept on].
Only concepts the CBM predicts as ON are explained (the ReX target is then class 1).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from rex_xai.input.input_data import Data
from rex_xai.lib import ReX
from rex_xai.responsibility.prediction import Prediction
from scripts.run_cbm import CBM, load_image,image_path  # your wrapper from earlier
from pathlib import Path
from .frozen_concept import FrozenClassView
MODEL_SHAPE = ("N", 3, 299, 299)

# Must match the transforms the CBMs were trained/evaluated with.
transform = transforms.Compose([
    transforms.CenterCrop(299),
    transforms.ToTensor(),                                   # (3, H, W) in [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
])


def preprocess(path, device) -> Data:
    """Build the Data object ReX needs.

    ReX.set_rgb_image() does NOT populate data.data, so we build it ourselves:
      data.input -> PIL image (cropped identically), used only for visualisation
      data.data  -> normalised tensor (1, 3, 299, 299) fed to the model
    """
    img = Image.open(path).convert("RGB")
    data = Data(img, MODEL_SHAPE, device, mode="RGB")
    data.data = transform(img).unsqueeze(0).to(device)
    data.input = transforms.functional.center_crop(img, 299)
    return data


def make_prediction_function(cbm, k,concept=True,frozen=False):
    """ReX prediction function for concept k.

    ReX calls it as f(mutants, target) -> list[Prediction], and as
    f(mutants, raw=True) -> (B, n_classes) probabilities (used by the
    insertion/deletion evaluation).
    """
    @torch.no_grad()
    def frozen_concept_prediction_function(mutants, target=None, raw=False):
        #

    @torch.no_grad()
    def class_prediction_function(mutants, target=None, raw=False):
        z = cbm.class_logits(mutants)                 # (B,) concept logit
        logits = torch.stack([torch.zeros_like(z), z], dim=1)  # (B, 2): [off, on]
        probs = F.softmax(logits, dim=1)                       # [:, 1] == sigmoid(z)
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
                    target=target,              # Prediction object; class is extracted inside
                    target_confidence=t_conf,
                )
            )


    @torch.no_grad()
    def prediction_function(mutants, target=None, raw=False):
        z = cbm.concept_logits(mutants)[:, k]                 # (B,) concept logit
        logits = torch.stack([torch.zeros_like(z), z], dim=1)  # (B, 2): [off, on]
        probs = F.softmax(logits, dim=1)                       # [:, 1] == sigmoid(z)
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
                    target=target,              # Prediction object; class is extracted inside
                    target_confidence=t_conf,
                )
            )
        return preds

    return prediction_function if concept else class_prediction_function
@torch.no_grad()
def active_concepts(cbm, record, device="cuda", threshold=0.5):
    """Indices of concepts the CBM predicts as ON, plus all concept probabilities."""
    data : torch.Tensor = load_image(record).unsqueeze(0).to(device)
    p = torch.sigmoid(cbm.concept_logits(data.data))[0]
    return (p > threshold).nonzero().flatten().tolist(), p.cpu().numpy()

@torch.no_grad()
def inactive_concepts(cbm, record, device="cuda", threshold=0.5):
    """Indices of concepts the CBM predicts as OF, plus all concept probabilities."""
    data : torch.Tensor = load_image(record).unsqueeze(0).to(device)
    p = torch.sigmoid(cbm.concept_logits(data.data))[0]
    return (p < threshold).nonzero().flatten().tolist(), p.cpu().numpy()

def crop_image(img, resol):
    transform = transforms.Compose([
        transforms.CenterCrop(resol)
    ])

    return transform(img)


def get_bird_mask(record, resol=299):
    segmentation_path = Path(str(image_path(record)).replace("\\images\\", "\\segmentations\\")).with_suffix(".png")
    seg_img = Image.open(segmentation_path).convert("L")
    cropped_seg_img = crop_image(seg_img, resol)
    cropped_tensor = torch.from_numpy(np.array(cropped_seg_img))
    return (cropped_tensor > 127).to(device)

def to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
    ])

    return transform(img)

def setup_rex(cbm,
              k,
              record, 
              device="cuda",
              iters=20, 
              search_limit=1000,
              concept = True):
    
    
    rex = ReX(model=cbm,
            model_shape=MODEL_SHAPE,
            device=device,
            mode="RGB",
            prediction_function=make_prediction_function(cbm, k,concept=concept),
        )
    img_path = image_path(record)
    image = rex.set_rgb_image(img_path)
    cropped_img  = crop_image(image,299)

    rex.args.iters = iters
    rex.args.search_limit = search_limit
    rex.set_input(cropped_img)
    transformed_img = to_tensor(cropped_img).unsqueeze(0).to(device)
    rex.set_transformed_data(transformed_img)
    rex.set_target() 
    return rex,image      



def explain_concept(rex, 
                    target_classification=1,
                    batch_size=64,
                    only_if_on=True):
    """Run ReX for one concept. Returns a dict, or None if the concept is off."""  
    # target = top class of the [off, on] pair

    target = rex.data.target
    if only_if_on and target.classification != target_classification:
        return None

    rex.calculate_responsibility()
    rex.extract_sufficient_explanation()

    # explanation mask has the shape of data.data: (1, 3, H, W) bool -> (H, W)
    mask = rex.explanation.sufficiency_mask[0].any(0).cpu().numpy()
    resp = rex.maps.get(target.classification)               # (H, W) responsibility map

    return {
        "concept": k,
        "target_class": target.classification,
        "target_confidence": target.confidence,   # should equal sigmoid(logit) for "on"
        "mask": mask,
        "area_fraction": float(mask.mean()),
        "responsibility": resp,
        "stats": rex.stats,
        "rex": rex,                                # keep for rex.show() etc.
    }


def explain_all_active(cbm, img_path, device="cuda", concepts=None, **kwargs):
    """Explain every active concept (or the subset `concepts`) for one image."""
    active, probs = active_concepts(cbm, img_path, device)
    todo = active if concepts is None else [k for k in concepts if k in active]
    results = {}
    for k in todo:
        out = explain_concept(cbm, k, img_path, device, **kwargs)
        if out is not None:
            results[k] = out
            print(f"concept {k:3d}  p={probs[k]:.3f}  "
                  f"rex_conf={out['target_confidence']:.3f}  "
                  f"area={out['area_fraction']:.3%}")
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cbm = CBM("joint", device=device)            # switch model by name
    img = "path/to/cub_image.jpg"

    # 1) quick test of a single concept; check rex_conf ~= p (preprocessing sanity check)
    active, probs = active_concepts(cbm, img, device)
    print(f"{len(active)} active concepts")
    k = active[0]
    out = explain_concept(cbm, k, img, device)
    print(f"concept {k}: p={probs[k]:.4f} vs ReX conf={out['target_confidence']:.4f}")

    # 2) all active concepts for this image
    results = explain_all_active(cbm, img, device, iters=20, search_limit=1000)