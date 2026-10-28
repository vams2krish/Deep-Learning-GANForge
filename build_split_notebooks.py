"""
Generator for 12 standalone, single-GAN notebooks extracted from
GANForge_Master_Notebook.ipynb.

Why regenerate from build_notebook.py instead of slicing the .ipynb file
directly? Because `build_notebook.py` already holds the exact, bug-fixed
cell content as a flat Python list (`cells`), indexed 1:1 with notebook
cell order -- importing it re-runs the same md()/code() calls (cheap,
pure string building, no torch/network calls happen at import time) and
hands us that list back without needing json parsing.

Every split notebook = [title cell] + [Section 0 utility cells] + [that
GAN's own section cells]. All 11 of the 12 sections are fully
self-contained already. Section 10 (ESRGAN) is the one exception -- in
the master notebook it deliberately reuses Section 9 (SRGAN)'s data
pipeline, VGG perceptual-loss network, and psnr() metric ("no reason to
redefine or re-download the same data" makes sense in one combined
notebook). Split on its own, those references would be undefined names --
so ESRGAN's cells are patched below to inline its own copy of that
shared infrastructure, and its SRGAN-vs-ESRGAN comparison plot is swapped
for a Bicubic-vs-ESRGAN comparison (mirroring exactly what Section 9's
own visualization already does), removing the cross-file dependency
entirely.
"""
import json
import os

import build_notebook as bn  # noqa: E402  (importing populates bn.cells; build() is guarded by __main__ and never runs)

OUT_DIR = "standalone_notebooks"
os.makedirs(OUT_DIR, exist_ok=True)

SETUP_RANGE = (1, 7)  # Section 0: imports, device, seed, shared helpers (bn.cells[0] is the master title/TOC, skipped)

# (output filename stem, notebook-cell index range in bn.cells, display name)
SECTIONS = [
    ("01_GAN",               (7, 22),    "GAN", "Unconditional generation from noise"),
    ("02_cGAN",               (22, 37),   "cGAN", "Class-conditional generation"),
    ("03_ACGAN",              (37, 52),   "AC-GAN", "Conditional generation + auxiliary classification"),
    ("04_ProgressiveGAN",     (52, 68),   "Progressive GAN", "Stable high-resolution generation via progressive growing"),
    ("05_StyleGAN2",          (68, 84),   "StyleGAN2", "Photorealistic controllable face generation"),
    ("06_BigGAN",             (84, 100),  "BigGAN", "Large-scale class-conditional high-quality generation"),
    ("07_StackGAN",           (100, 116), "StackGAN", "Text-to-image generation (two-stage)"),
    ("08_CycleGAN",           (116, 131), "CycleGAN", "Unpaired image-to-image translation"),
    ("09_SRGAN",              (131, 146), "SRGAN", "Single-image super-resolution"),
    ("10_ESRGAN",             (146, 161), "ESRGAN", "Enhanced perceptual super-resolution"),
    ("11_GANSynth",           (161, 176), "GANSynth", "Audio / music generation"),
    ("12_GameGAN",            (176, 191), "GameGAN", "Game environment / next-frame simulation"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    src = text.lstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def title_cell(display_name, one_liner):
    return md(rf"""
# GANForge -- {display_name} (Standalone Notebook)

**{one_liner}**

This is a **self-contained extract** of the {display_name} section from
the [GANForge Master Notebook](GANForge_Master_Notebook.ipynb) -- every
shared setup utility it needs (imports, device selection, seeding,
`weights_init`, `show_images`, checkpointing, etc.) is included below in
Section 0, so this notebook runs **top to bottom on its own**, in Colab
or locally, without needing any other file.

> **⚠️ Compute note.** See the Section 0 GPU warning below. Heavier
> architectures (Progressive GAN, StyleGAN2, BigGAN, StackGAN, GameGAN)
> include a lightweight from-scratch training loop sized for a single
> GPU (or CPU, slowly) plus, where realistically available, code to load
> strong pretrained weights for inference.
""")


# ---------------------------------------------------------------------------
# ESRGAN-specific patch (removes its cross-reference to the SRGAN section)
# ---------------------------------------------------------------------------
ESRGAN_INTRO_PATCH = md(r"""
<a id="10"></a>
## ESRGAN -- Enhanced perceptual super-resolution

### 10.1 Problem Statement & Why this GAN

Wang et al. (2018) refined SRGAN's recipe to remove specific, identifiable
artifacts: SRGAN's BatchNorm layers were found to introduce faint
grid-like artifacts when train/test statistics differ even slightly, so
ESRGAN **removes BatchNorm entirely**. Its residual blocks are replaced
with **Residual-in-Residual Dense Blocks (RRDB)** -- densely-connected
convolutions (à la DenseNet) inside a residually-scaled block inside
another residually-scaled block -- for much deeper, more stable feature
reuse. Its discriminator is upgraded from "is this real or fake" to a
**Relativistic average GAN (RaGAN)**: "is the real image *more realistic
than* the fake one, on average" -- a relative, not absolute, judgment.
Together these solve the same problem as SRGAN (single-image
super-resolution) with **measurably fewer artifacts and sharper,
more natural-looking textures**.

### 10.2 Data Collection & Preparation

Same bicubic-degradation (LR, HR) pairing recipe SRGAN uses: HR images
come from STL-10 (falling back to synthetic textures if offline), LR is
a 4x bicubic-downsampled version built on the fly. **Swap in DIV2K
(`eugenesiow/Div2k` on Hugging Face Datasets, or the official DIV2K
download) for a production-quality run.**
""")

ESRGAN_DATA_PATCH = code(r"""
# ---------------------------------------------------------------------------
# (LR, HR) pairing pipeline -- identical recipe to the SRGAN section of the
# GANForge Master Notebook, duplicated here so this notebook is self-
# contained. Constant names keep the "SRGAN_" prefix since they describe
# the shared super-resolution pipeline convention, not anything SRGAN-model-specific.
# ---------------------------------------------------------------------------
SRGAN_HR_SIZE = 96          # STL-10's native resolution -- convenient, no cropping needed
SRGAN_SCALE = 4
SRGAN_LR_SIZE = SRGAN_HR_SIZE // SRGAN_SCALE   # 24x24 input -> 96x96 output
SRGAN_BATCH_SIZE = 16

hr_transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

try:
    srgan_hr_base = torchvision.datasets.STL10(root="data", split="train", download=True, transform=hr_transform)
    print(f"Loaded real STL-10 (as HR source; production setups should use DIV2K): {len(srgan_hr_base)} images")
except Exception as e:
    print(f"STL-10 download failed ({e}); falling back to synthetic textures.")
    rng = np.random.default_rng(SEED)
    tex = torch.rand(500, 3, SRGAN_HR_SIZE, SRGAN_HR_SIZE) * 2 - 1
    srgan_hr_base = TensorDataset(tex, torch.zeros(500, dtype=torch.long))

class SRDataset(Dataset):
    '''Builds (LR, HR) PAIRS on the fly: HR is the original image, LR is a
    4x bicubic-downsampled version. Bicubic downsampling is the SRGAN
    paper's degradation model -- swap this for a learned/realistic
    degradation model (blur+noise+JPEG) for production-quality real-world
    super-resolution.'''
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        hr, _ = self.base[i]
        lr = F.interpolate(hr.unsqueeze(0), size=SRGAN_LR_SIZE, mode="bicubic", align_corners=False, antialias=True)
        return lr.squeeze(0).clamp(-1, 1), hr

esrgan_dataset = SRDataset(srgan_hr_base)
esrgan_loader = DataLoader(esrgan_dataset, batch_size=SRGAN_BATCH_SIZE, shuffle=True, drop_last=True)
lr_batch_e, hr_batch_e = next(iter(esrgan_loader))
show_images(F.interpolate(lr_batch_e[:8], size=SRGAN_HR_SIZE, mode="nearest"), title="LR inputs (nearest-upsampled for display)", nrow=8, figsize=(12, 2))
show_images(hr_batch_e[:8], title="HR targets", nrow=8, figsize=(12, 2))
""")

ESRGAN_LOSS_PATCH = code(r"""
# ---------------------------------------------------------------------------
# VGG19 perceptual-loss network and the PSNR metric -- generic super
# -resolution infrastructure (identical to the SRGAN section of the
# GANForge Master Notebook), defined locally here so this notebook has no
# dependency on any other file.
# ---------------------------------------------------------------------------
def build_vgg_feature_extractor():
    '''Perceptual ("content") loss network: a pretrained VGG19's early
    conv-block features, frozen. Comparing SR/HR images in THIS feature
    space (rather than raw pixels) rewards texture/structure similarity
    the way a human perceives it, not literal per-pixel agreement.'''
    try:
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features[:36].to(DEVICE).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        print("Loaded pretrained VGG19 for perceptual loss.")
        return vgg
    except Exception as e:
        print(f"Pretrained VGG19 unavailable ({e}); falling back to pixel-space MSE as the content loss.")
        return None

srgan_vgg = build_vgg_feature_extractor()
srgan_vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
srgan_vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

def psnr(sr, hr):
    '''Peak Signal-to-Noise Ratio in dB -- the standard quantitative metric
    for super-resolution (higher = closer to the ground-truth HR image).'''
    mse = F.mse_loss(denorm(sr), denorm(hr)).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))

def esrgan_content_loss(sr, hr):
    if srgan_vgg is None:
        return F.l1_loss(sr, hr)
    sr_n = (denorm(sr) - srgan_vgg_mean) / srgan_vgg_std
    hr_n = (denorm(hr) - srgan_vgg_mean) / srgan_vgg_std
    return F.l1_loss(srgan_vgg(sr_n), srgan_vgg(hr_n))   # ESRGAN uses L1 (not L2/MSE) in feature space -- slightly less blur-prone

def esrgan_relativistic_loss(logits_real, logits_fake):
    '''Relativistic average GAN loss (Jolicoeur-Martineau, 2018):
    D judges "is REAL more realistic than the AVERAGE fake" and vice
    versa, instead of an absolute real/fake call. This gives G a gradient
    even from images D would already call "real" in an absolute sense,
    as long as they are less convincing than the current fakes on
    average -- the paper found this measurably sharper than vanilla GAN
    loss for super-resolution specifically.'''
    rel_real = logits_real - logits_fake.mean()
    rel_fake = logits_fake - logits_real.mean()
    loss_d = F.binary_cross_entropy_with_logits(rel_real, torch.ones_like(rel_real)) + \
             F.binary_cross_entropy_with_logits(rel_fake, torch.zeros_like(rel_fake))
    loss_g = F.binary_cross_entropy_with_logits(rel_real, torch.zeros_like(rel_real)) + \
             F.binary_cross_entropy_with_logits(rel_fake, torch.ones_like(rel_fake))
    return loss_d, loss_g

ESRGAN_LAMBDA_PIXEL = 1e-2      # small direct L1 pixel term for extra stability early in training
ESRGAN_LAMBDA_ADV = 5e-3
esrgan_opt_G = torch.optim.Adam(G_esrgan.parameters(), lr=1e-4, betas=(0.9, 0.999))
esrgan_opt_D = torch.optim.Adam(D_esrgan.parameters(), lr=1e-4, betas=(0.9, 0.999))
ESRGAN_EPOCHS = 6
""")

ESRGAN_VIZ_PATCH = code(r"""
gs, ds, psnrs = zip(*esrgan_history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(gs, label="G"); axes[0].plot(ds, label="D"); axes[0].legend(); axes[0].set_title("Losses")
axes[1].plot(psnrs); axes[1].set_title("PSNR (dB)")
plt.show()

# Bicubic upsampling (a naive, non-learned baseline) vs ESRGAN vs ground truth.
with torch.no_grad():
    lr4, hr4 = lr_batch_e[:4].to(DEVICE), hr_batch_e[:4].to(DEVICE)
    bicubic = F.interpolate(lr4, size=SRGAN_HR_SIZE, mode="bicubic", align_corners=False)
    sr_esrgan = esrgan_upscale(lr4)
fig, axes = plt.subplots(4, 3, figsize=(6, 8))
for row in range(4):
    for col, (img, name) in enumerate(zip([bicubic[row], sr_esrgan[row], hr4[row]], ["Bicubic", "ESRGAN", "Ground truth"])):
        axes[row, col].imshow(denorm(img).permute(1, 2, 0).cpu()); axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(name)
plt.tight_layout(); plt.show()
print(f"Bicubic PSNR: {psnr(bicubic, hr4):.2f}dB | ESRGAN PSNR: {psnr(sr_esrgan, hr4):.2f}dB "
      f"(remember: higher PSNR != more realistic, see 10.9)")
""")


def patch_esrgan(section_cells):
    '''section_cells is bn.cells[146:161] (15 cells). Indices below are
    relative offsets within that slice, derived from inspecting the master
    notebook: 0 = intro md (10.1+10.2), 1 = data cell, 5 = loss cell,
    11 = visualization cell. See module docstring for why these differ
    from the master.'''
    patched = list(section_cells)
    patched[0] = ESRGAN_INTRO_PATCH   # "<a id=10>" + 10.1 + 10.2 markdown
    patched[1] = ESRGAN_DATA_PATCH    # was: esrgan_loader = srgan_loader (one-liner reuse)
    patched[5] = ESRGAN_LOSS_PATCH    # was: comment + esrgan_content_loss reusing srgan_vgg
    patched[11] = ESRGAN_VIZ_PATCH    # was: SRGAN vs ESRGAN comparison using srgan_upscale
    return patched


def build():
    setup_cells = bn.cells[SETUP_RANGE[0]:SETUP_RANGE[1]]

    for stem, (lo, hi), name, one_liner in SECTIONS:
        section_cells = bn.cells[lo:hi]
        if stem == "10_ESRGAN":
            section_cells = patch_esrgan(section_cells)

        nb_cells = [title_cell(name, one_liner)] + setup_cells + section_cells

        nb = {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.10"},
                "colab": {"name": f"GANForge_{stem}.ipynb", "provenance": [], "gpuType": "T4"},
                "accelerator": "GPU",
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        out_path = os.path.join(OUT_DIR, f"GANForge_{stem}.ipynb")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Wrote {out_path} ({len(nb_cells)} cells)")


if __name__ == "__main__":
    build()
