# ReX on Concept Bottleneck Models

KCL research project. We apply ReX (causal explanations, Dr. Hana Chockler) to the
Concept Bottleneck Models of Koh et al. (2020) on CUB-200-2011, to test whether the
models really predict from the concepts only, or whether information leaks through
(for example from the background).

Current experiments:
- **Background swap**: paste the bird (CUB segmentation mask) on new backgrounds and
  compare predictions, concepts and ReX explanations.
- **Per-concept explanations**: freeze the concepts from the first model, explain each
  concept with ReX, and check whether the class prediction uses their combination.

## Layout

```
scripts/      shared Python scripts (run from the repo root)
data/         NOT in git, see below
ConceptBottleneck/, ReX/   NOT in git, cloned separately
notebooks/, outputs/       NOT in git (local only for now)
```

## Setup

1. Environment (we use a conda env called `rex`, Python 3.13):
   ```bash
   conda create -n rex python=3.13
   conda activate rex
   pip install -r requirements.txt
   ```
2. Clone the two upstream repos into the repo root (the model checkpoints unpickle
   classes from `ConceptBottleneck/CUB/template_model.py`):
   ```bash
   git clone https://github.com/yewsiang/ConceptBottleneck.git
   git clone https://github.com/ReX-XAI/ReX.git   # optional, for reading the source
   ```
3. Put the data under `data/` with this layout:
   ```
   data/CUB_200_2011/        images/, segmentations/, classes.txt, attributes/attributes.txt
   data/CUB_processed/       train.pkl, val.pkl, test.pkl
   data/models/              ConceptModel__Seed1/, IndependentModel_WithVal___Seed1/,
                             SequentialModel_WithVal__Seed1/, Joint0.01Model__Seed1/, ...
                             (each with best_model_1.pth)
   ```
   CUB-200-2011, the processed pickles and the pretrained checkpoints come from the
   CodaLab worksheet linked in the ConceptBottleneck README. The segmentations are a
   separate download from the CUB-200-2011 page (Caltech).

## Conventions

- Preprocessing: `CenterCrop(299)`, `ToTensor()`, `Normalize(mean=0.5, std=2)` (no resize).
- Joint model output: `out[0]` = class logits, `out[1:]` = 112 concept logits.
- The 112 concepts are the 0-based attribute positions in `SELECTED_ATTRIBUTES`
  (`scripts/run_cbm.py`).
- ReX: `minimum_confidence_threshold = 0.9`; set `rex.args.seed` for reproducible runs.
- Paths are relative to the repo root (`Path(__file__).resolve().parents[1]`), so don't
  hard-code absolute paths.

## Quick check

```bash
python scripts/run_cbm.py demo --n-images 5
python scripts/run_cbm.py eval --every 10   # Joint ~79% class / ~97% concept accuracy
```
