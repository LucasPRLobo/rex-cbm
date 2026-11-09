"""Run the pretrained CUB Concept Bottleneck Models on CPU.

Loads the authors' checkpoints and reproduces the repo's test preprocessing without its .cuda() calls.

  python scripts/run_cbm.py demo --n-images 5    # predictions + concepts for a few test images
  python scripts/run_cbm.py eval --every 10      # accuracy on every 10th test image
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "ConceptBottleneck"))  # checkpoints unpickle classes from CUB.template_model

# The 112 of 312 CUB attributes used by the paper, 0-based (from CUB/generate_new_data.py).
SELECTED_ATTRIBUTES = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64,
    69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149,
    151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 208,
    209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262,
    268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
]

# Same as the non-training branch of load_data in CUB/dataset.py: crop only, no resize.
TEST_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
])


def load_checkpoint(name):
    path = DATA / "models" / name / "best_model_1.pth"
    return torch.load(path, map_location="cpu", weights_only=False).eval()


def load_models():
    """Return {name: fn}, where fn maps images to (class_logits, concept_logits)."""
    concept_model = load_checkpoint("ConceptModel__Seed1")
    independent_head = load_checkpoint("IndependentModel_WithVal___Seed1")
    sequential_head = load_checkpoint("SequentialModel_WithVal__Seed1")
    joint = load_checkpoint("Joint0.01Model__Seed1")
    joint_sigmoid = load_checkpoint("Joint0.01SigmoidModel__Seed1")

    def run_joint(model):
        def run(x):
            out = model(x)
            return out[0], torch.cat(out[1:], dim=1)
        return run

    def run_independent(x):  # head was trained on true concepts, so it takes probabilities
        c = torch.cat(concept_model(x), dim=1)
        return independent_head(torch.sigmoid(c)), c

    def run_sequential(x):  # head was trained on predicted logits
        c = torch.cat(concept_model(x), dim=1)
        return sequential_head(c), c

    return {
        "independent": run_independent,
        "sequential": run_sequential,
        "joint": run_joint(joint),
        "joint_sigmoid": run_joint(joint_sigmoid),
    }


def load_test_records():
    with open(DATA / "CUB_processed" / "test.pkl", "rb") as f:
        return pickle.load(f)


def image_path(record):
    return DATA / "CUB_200_2011" / record["img_path"].split("CUB_200_2011/")[1]


def load_image(record):
    return TEST_TRANSFORM(Image.open(image_path(record)).convert("RGB"))


def class_names():
    lines = (DATA / "CUB_200_2011" / "classes.txt").read_text().splitlines()
    return [line.split(" ", 1)[1].split(".", 1)[1].replace("_", " ") for line in lines]


def concept_names():
    lines = (DATA / "CUB_200_2011" / "attributes" / "attributes.txt").read_text().splitlines()
    names = [line.split(" ", 1)[1] for line in lines]
    return [names[i].replace("has_", "").replace("::", ": ") for i in SELECTED_ATTRIBUTES]


@torch.no_grad()
def evaluate(models, records, batch_size):
    labels = np.array([r["class_label"] for r in records])
    concepts = np.array([r["attribute_label"] for r in records])
    correct = {name: 0 for name in models}
    concept_correct = {name: 0 for name in models}
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        x = torch.stack([load_image(r) for r in batch])
        for name, model in models.items():
            class_logits, concept_logits = model(x)
            correct[name] += (class_logits.argmax(1).numpy() == labels[start:start + batch_size]).sum()
            hard = (concept_logits > 0).numpy()  # sigmoid(logit) > 0.5
            concept_correct[name] += (hard == concepts[start:start + batch_size]).sum()
        print(f"  {start + len(batch)}/{len(records)}", flush=True)
    return {
        name: {
            "class_accuracy": float(correct[name] / len(records)),
            "concept_accuracy": float(concept_correct[name] / concepts.size),
            "n_images": len(records),
        }
        for name in models
    }


@torch.no_grad()
def demo(models, records, n_images, top_k):
    classes, concepts = class_names(), concept_names()
    indices = np.linspace(0, len(records) - 1, n_images).astype(int)  # evenly spaced, not hand-picked
    results = []
    for index in indices:
        record = records[index]
        x = load_image(record).unsqueeze(0)
        true_concepts = np.array(record["attribute_label"])
        entry = {
            "test_index": int(index),
            "image": str(image_path(record).relative_to(DATA)),
            "true_class": classes[record["class_label"]],
            "models": {},
        }
        for name, model in models.items():
            class_logits, concept_logits = model(x)
            class_probs = torch.softmax(class_logits, dim=1)[0]
            concept_probs = torch.sigmoid(concept_logits)[0].numpy()
            top = np.argsort(-concept_probs)[:top_k]
            entry["models"][name] = {
                "predicted_class": classes[int(class_probs.argmax())],
                "correct": int(class_probs.argmax()) == record["class_label"],
                "confidence": float(class_probs.max()),
                "concept_accuracy": float(((concept_probs > 0.5) == true_concepts).mean()),
                "top_concepts": [
                    {"concept": concepts[i], "prob": float(concept_probs[i]), "label": int(true_concepts[i])}
                    for i in top
                ],
            }
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["demo", "eval"])
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "stage2")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_test_records()
    models = load_models()
    if args.command == "demo":
        result = demo(models, records, args.n_images, args.top_k)
    else:
        result = evaluate(models, records[::args.every], args.batch_size)
    out_path = args.out_dir / f"{args.command}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
