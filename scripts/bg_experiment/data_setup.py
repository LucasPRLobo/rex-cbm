"""Shared constants, image loading and the background composites (COMPOSITES / EXPERIMENTS)."""
import warnings
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image

from run_cbm import DATA, SELECTED_ATTRIBUTES, TEST_TRANSFORM, image_path

warnings.filterwarnings("ignore", message="source code of class")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOL = 299
MODEL_SHAPE = ["N", 3, RESOL, RESOL]
CLASSES = [line.split(" ", 1)[1] for line in (DATA / "CUB_200_2011" / "classes.txt").read_text().splitlines()]
_ATTRS = (DATA / "CUB_200_2011" / "attributes" / "attributes.txt").read_text().splitlines()
CONCEPT_NAMES = [_ATTRS[i].split(" ", 1)[1] for i in SELECTED_ATTRIBUTES]

# Waterbirds (Sagawa et al., 2020) split: seabirds and waterfowl are "water", every other class is "land".
WATER_KEYWORDS = ["Albatross", "Auklet", "Cormorant", "Frigatebird", "Fulmar", "Gull", "Jaeger", "Kittiwake",
                  "Pelican", "Puffin", "Tern", "Gadwall", "Grebe", "Mallard", "Merganser", "Guillemot",
                  "Pacific_Loon"]
BIRD_TYPE = ["water" if any(k in c for k in WATER_KEYWORDS) else "land" for c in CLASSES]
PLACES_DIR = DATA / "backgrounds"

# ----Functions----

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


def load_places(places_dir=PLACES_DIR):
    """{"<category>/<file stem>": normalised [3, 299, 299] tensor} from data/backgrounds/<category>/*.jpg."""
    if not places_dir.exists():
        raise SystemExit(f"{places_dir} not found: run scripts/get_places.py first")
    resize = TF.resize
    places = {}
    for path in sorted(places_dir.glob("*/*.jpg")):
        scene = resize(Image.open(path).convert("RGB"), RESOL)  # Places365 val images are 256x256
        places[f"{path.parent.name}/{path.stem}"] = TEST_TRANSFORM(scene).to(DEVICE)
    print(f"loaded {len(places)} backgrounds from {places_dir}", flush=True)
    return places


def solid(img, p):
    """Background of one grey level p in [0, 1] (0 = black, 1 = white), same shape as img."""
    return torch.full_like(img, to_normalised(p))


def mean_colour(img, region):
    """Background filled with the average colour of img over a bool region."""
    return img[:, region].mean(dim=1)[:, None, None].expand_as(img).clone()


def paste(img, mask, background):
    """Bird pixels from img, everything else from background."""
    return torch.where(mask, img, background)


def bbox(mask):
    """Bool mask of the bird's bounding box."""
    box = torch.zeros_like(mask)
    ys, xs = torch.where(mask)
    if len(ys):
        box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return box


def resize_mask(mask, k):
    """k > 0 grows the mask by k pixels, k < 0 shrinks it."""
    m = mask.float()[None, None]  # [1, 1, H, W]
    if k > 0:
        m = F.max_pool2d(m, 2 * k + 1, stride=1, padding=k)
    elif k < 0:
        m = 1 - F.max_pool2d(1 - m, 2 * -k + 1, stride=1, padding=-k)
    return m[0, 0].bool()



# Every composite is a function (img, mask) -> [3, 299, 299] image. Add a new background by adding an entry.
COMPOSITES = {
    "grey": lambda img, mask: paste(img, mask, solid(img, 0.5)),
    "black": lambda img, mask: paste(img, mask, solid(img, 0.0)),
    "white": lambda img, mask: paste(img, mask, solid(img, 1.0)),
    "blur": lambda img, mask: paste(img, mask, TF.gaussian_blur(img, kernel_size=51, sigma=20)),
    # the bird's own average colour: minimum contrast between bird and background
    "bird_mean": lambda img, mask: paste(img, mask, mean_colour(img, mask)),
    # the original background's average colour: same overall colour, no layout or detail
    "bg_mean": lambda img, mask: paste(img, mask, mean_colour(img, ~mask)),
    # bird removed: its bounding box is greyed out (a bird-shaped hole would still show the silhouette)
    "bg_only": lambda img, mask: torch.where(bbox(mask), solid(img, 0.5), img),
}
BRIGHTNESS_LEVELS = np.linspace(0, 1, 7)
for p in BRIGHTNESS_LEVELS:
    COMPOSITES[f"level_{p:.2f}"] = lambda img, mask, p=p: paste(img, mask, solid(img, p))  # p=p: keep this p
MASK_K = {f"black_k{k:+d}": k for k in (-4, -2, 2, 4, 8)}  # bird on black with the mask grown/shrunk k pixels
for name, k in MASK_K.items():
    COMPOSITES[name] = lambda img, mask, k=k: paste(img, resize_mask(mask, k), solid(img, 0.0))

# Each experiment is a list of composite names; "original" is always added first.
EXPERIMENTS = {
    "swap": ["grey", "black", "white", "blur", "bird_mean", "bg_mean", "bg_only"],
    "brightness": [f"level_{p:.2f}" for p in BRIGHTNESS_LEVELS],
    "mask": ["black"] + list(MASK_K),
    "places": [],  # the loaded scenes are passed as `extra`
    # grey is skipped for ReX: ReX hides pixels by setting them to grey, so it cannot test a grey background
    "rex": ["black", "white", "blur"],
}


def build_composites(img, mask, names, extra=None):
    """Return (names, [K, 3, 299, 299] batch): the original, then each named composite, then the bird
    pasted on each background in `extra` ({name: background tensor}, e.g. loaded scenes)."""
    out = {"original": img}
    out.update({name: COMPOSITES[name](img, mask) for name in names})
    if extra:
        out.update({name: paste(img, mask, bg) for name, bg in extra.items()})
    return list(out), torch.stack(list(out.values()))


def background_brightness(x, mask):
    """Mean pixel value (0-1) of the non-bird region."""
    pixels = (x * 2 + 0.5).clamp(0, 1)[:, ~mask]
    return pixels.mean().item() if pixels.numel() else float("nan")

