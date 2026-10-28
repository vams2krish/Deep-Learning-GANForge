"""
Generator script for GANForge_Master_Notebook.ipynb

Why a generator script instead of hand-writing the .ipynb JSON?
- Cell source needs real Python/Markdown with normal quoting; writing that
  inside escaped JSON strings by hand is extremely error prone at this scale.
- This script builds a plain list of (markdown|code) cells using ordinary
  triple-quoted strings, then serializes to nbformat v4 JSON at the end.
No external deps (nbformat is not installed in this environment) -- we just
emit the minimal valid nbformat v4.5 structure by hand.
"""
import json

cells = []


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str):
    # strip a single leading newline if present for tidiness, keep rest as-is
    src = text.lstrip("\n")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })


\
# ---------------------------------------------------------------------------
# TITLE + TABLE OF CONTENTS
# ---------------------------------------------------------------------------
md(r"""
# GANForge Master Notebook -- 12 GANs End-to-End

A single, self-contained, educational tour of twelve landmark GAN
architectures, each solving a genuinely different problem: unconditional
generation, class-conditional generation, high-resolution synthesis, style
control, text-to-image, unpaired translation, super-resolution, audio
synthesis, and game-frame prediction.

**How this notebook is built, and why it looks the way it does**

Training the *original* StyleGAN2, BigGAN, ProGAN, StackGAN or GameGAN from
their papers takes many V100/A100-GPU-days per model and terabytes of data
(FFHQ, JFT-300M/ImageNet-21k, DIV2K, NSynth, thousands of game rollouts).
That is not reproducible in a single notebook session. So every section
below gives you **two things**:

1. A **lightweight "from-scratch" version** of the real architecture
   (same building blocks: progressive growing, style modulation, self-attn +
   spectral norm, RRDB blocks, cycle-consistency, etc.) trained on a small
   dataset for a small number of epochs, so it **actually finishes running**
   on a single Colab GPU (or even CPU, slowly) and produces visible,
   if imperfect, results.
2. Where a strong **pretrained checkpoint** is realistically loadable
   (BigGAN, StyleGAN2, CycleGAN), code to load it for **inference-only**
   high-quality generation -- wrapped in `try/except` because Colab/offline
   environments do not always have network access to model hubs.

Every dataset loader below **first tries the real, named dataset** and,
if the download is unavailable in your environment, **falls back to a small
synthetically generated stand-in** so that *every single cell in this
notebook can be executed top-to-bottom without crashing*, even fully
offline. Swap in the real dataset (paths are clearly marked) for
research-quality results.

> **⚠️ Warning -- Compute expectations.** Sections 4-7 and 12 (Progressive
> GAN, StyleGAN2, BigGAN, StackGAN, GameGAN) are the heavy ones. On a free
> Colab T4 the lightweight versions in this notebook take roughly
> 5-30 minutes each at the default (small) epoch counts. On CPU only,
> expect 5-10x slower -- reduce `EPOCHS`, `IMG_SIZE`, and `BATCH_SIZE`
> further, they are all exposed as variables at the top of each section.

## Table of Contents

0. [Global Setup & Utilities](#0)
1. [Original GAN -- unconditional generation from noise](#1)
2. [cGAN -- class-conditional generation](#2)
3. [AC-GAN -- conditional generation + auxiliary classification](#3)
4. [Progressive GAN -- stable high-resolution generation via progressive growing](#4)
5. [StyleGAN / StyleGAN2 -- photorealistic controllable face generation](#5)
6. [BigGAN -- large-scale class-conditional high-quality generation](#6)
7. [StackGAN -- text-to-image generation (two-stage)](#7)
8. [CycleGAN -- unpaired image-to-image translation](#8)
9. [SRGAN -- single-image super-resolution](#9)
10. [ESRGAN -- enhanced perceptual super-resolution](#10)
11. [GANSynth -- audio / music generation](#11)
12. [GameGAN -- game environment / next-frame simulation](#12)
13. [Unified Inference Hub (Bonus)](#13)
14. [Final Notes & Recommendations](#14)
""")

\
# ---------------------------------------------------------------------------
# SECTION 0 -- GLOBAL SETUP & UTILITIES
# ---------------------------------------------------------------------------
md(r"""
<a id="0"></a>
## 0. Global Setup & Utilities

Everything shared across all twelve GANs lives here: installs, imports,
device selection, seeding, and small helper functions (`weights_init`,
`show_images`, `save_checkpoint`, `Logger`, an `EMA` helper, and a spectral
-norm-aware conv wrapper). Re-using these keeps every later section short
and focused on what makes *that* GAN different.

**🚀 Pro Tip.** In Google Colab: `Runtime -> Change runtime type -> T4 GPU`
before running anything. Everything here auto-detects CUDA and degrades
gracefully to CPU (slow but functional) if none is found.
""")

code(r"""
# Colab-only: uncomment to install packages not preinstalled in Colab.
# All of these are optional -- every dataset loader below has an offline
# synthetic fallback, so the notebook still runs without them.
# !pip install -q torchaudio datasets soundfile pillow-simd torchmetrics
# !pip install -q pytorch-fid clean-fid
""")

code(r"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import io
import math
import time
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:
    # Minimal shim so the notebook still runs if tqdm is somehow missing.
    def tqdm(iterable, *args, **kwargs):
        return iterable

warnings.filterwarnings("ignore")  # GAN training is noisy (deprecation spam); keep output readable

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
""")

code(r"""
# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
# We centralize device selection so every model/tensor in the notebook is
# created consistently. All training loops below check DEVICE.type == "cuda"
# to decide whether to use mixed precision (torch.cuda.amp), which roughly
# halves memory use and often speeds up training 2-3x on modern GPUs.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU found -- running on CPU. Reduce EPOCHS/BATCH_SIZE in "
          "every section for reasonable runtimes.")
""")

code(r"""
# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# GANs are notoriously sensitive to initialization -- fixing every RNG makes
# results reproducible across reruns, which matters a lot when you are
# debugging "why did my generator collapse this time".
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
""")

code(r"""
# ---------------------------------------------------------------------------
# Shared helper functions used across every GAN section below
# ---------------------------------------------------------------------------

def weights_init(m: nn.Module):
    '''DCGAN-style weight init (Radford et al., 2015).

    Why: GAN training is a saddle-point optimization between two networks
    that constantly move each other's loss landscape. Starting both nets
    from small, zero-centered weights (N(0, 0.02)) avoids one network
    dominating early purely due to init scale, which is a common cause of
    early, unrecoverable mode collapse.

    Uses isinstance (NOT a substring match on the class name) because many
    custom container classes in this notebook -- PGAN_ConvBlock,
    ModulatedConv2d, StyleConvBlock -- have "Conv" in their own name; a
    naive `"Conv" in classname` check would incorrectly try to initialize
    THOSE containers directly (they have no `.weight` of their own) instead
    of only their actual nn.Conv2d/nn.Linear/nn.BatchNorm2d leaf children.
    '''
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and getattr(m, "affine", False):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_images(images: torch.Tensor, title: str = "", nrow: int = 8, figsize=(8, 8), normalize=True):
    '''Display a batch of image tensors (B, C, H, W) in [-1, 1] or [0, 1] as a grid.'''
    images = images.detach().cpu()
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, value_range=(-1, 1) if normalize else None, padding=2)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title(title)
    npimg = grid.numpy()
    if npimg.shape[0] == 1:
        plt.imshow(npimg[0], cmap="gray")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def save_checkpoint(state: dict, path: str):
    '''Persist model/optimizer state so long training runs survive a Colab disconnect.'''
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=map_location or DEVICE)


class EMA:
    '''Exponential Moving Average of generator weights.

    Why: instantaneous generator weights oscillate a lot during adversarial
    training (they are chasing a moving discriminator). Averaging weights
    over time (as StyleGAN2 and BigGAN both do) gives a much smoother,
    higher-quality generator for *inference*, at zero extra training cost.
    '''
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


class AvgMeter:
    '''Running mean tracker for loss logging.'''
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def denorm(x: torch.Tensor) -> torch.Tensor:
    '''Map a tensor from [-1, 1] (tanh output range) back to [0, 1] for display/saving.'''
    return (x.clamp(-1, 1) + 1) / 2


CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
print("Shared utilities ready.")
""")

\
# ---------------------------------------------------------------------------
# SECTION 1 -- ORIGINAL GAN
# ---------------------------------------------------------------------------
md(r"""
<a id="1"></a>
## 1. Original GAN -- Unconditional generation from noise

### 1.1 Problem Statement & Why this GAN

Goodfellow et al. (2014) posed generative modeling as a **two-player
minimax game**: a *Generator* `G` maps random noise `z ~ N(0, I)` to samples
that try to fool a *Discriminator* `D`, while `D` tries to tell real data
from `G`'s fakes. Neither network is ever told *what* to draw -- there is no
label, no condition, nothing but noise in and an image out. This is the
**unconditional generation** problem: learn the data distribution `p(x)`
well enough to sample convincingly from it.

We implement a DCGAN-style (Radford et al., 2015) convolutional
Generator/Discriminator pair -- the architecture that made the original
minimax idea actually stable enough to train reliably -- on MNIST digits.

💡 **Insight.** Every other GAN in this notebook is a variation on this
same adversarial game: cGAN/AC-GAN add conditioning, ProGAN/StyleGAN scale
the architecture, CycleGAN/SRGAN change what "real vs fake" is computed
over. Understanding this section well makes every later section a diff,
not a rewrite.

### 1.2 Data Collection & Preparation
""")

code(r"""
# MNIST is tiny (~11MB), auto-downloads via torchvision, and trains fast --
# ideal for demonstrating the *original* unconditional GAN game clearly and
# quickly. We normalize to [-1, 1] because our Generator's final activation
# is Tanh (standard DCGAN choice: bounded, zero-centered output).
GAN_IMG_SIZE = 32       # upsampled from MNIST's native 28x28 for clean conv strides (32 = 2^5)
GAN_BATCH_SIZE = 128
GAN_NZ = 100             # latent noise dimensionality
GAN_EPOCHS = 5            # small on purpose -- see Section 0 compute warning; raise for better samples

gan_transform = T.Compose([
    T.Resize(GAN_IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])

try:
    gan_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=gan_transform)
    print(f"Loaded real MNIST: {len(gan_dataset)} images")
except Exception as e:
    # Offline fallback: procedurally generated blob "digits" so the rest of
    # the section still runs end-to-end without internet access.
    print(f"MNIST download failed ({e}); falling back to synthetic blobs.")
    rng = np.random.default_rng(SEED)
    fake_imgs = torch.zeros(2000, 1, GAN_IMG_SIZE, GAN_IMG_SIZE)
    yy, xx = torch.meshgrid(torch.arange(GAN_IMG_SIZE), torch.arange(GAN_IMG_SIZE), indexing="ij")
    for i in range(2000):
        cx, cy = rng.uniform(8, 24, 2)
        r = rng.uniform(4, 10)
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2).float()
        fake_imgs[i, 0] = blob * 2 - 1
    gan_dataset = TensorDataset(fake_imgs, torch.zeros(2000, dtype=torch.long))

gan_loader = DataLoader(gan_dataset, batch_size=GAN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
real_batch, _ = next(iter(gan_loader))
show_images(real_batch[:64], title="Real MNIST samples", nrow=8)
""")

md(r"### 1.3 Model Architecture")

code(r"""
class GAN_Generator(nn.Module):
    '''DCGAN generator: a stack of transposed convolutions that upsample a
    1x1 noise vector into a full image, doubling spatial size each layer.
    BatchNorm + ReLU everywhere except the output (Tanh, to match our
    [-1, 1] data normalization) is the classic DCGAN recipe for stable
    gradients through a deep generator.'''
    def __init__(self, nz=GAN_NZ, ngf=64, nc=1):
        super().__init__()
        self.net = nn.Sequential(
            # z: (nz, 1, 1) -> (ngf*4, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # -> (ngf*2, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # -> (ngf, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # -> (nc, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class GAN_Discriminator(nn.Module):
    '''Mirror-image convolutional binary classifier: real (1) vs fake (0).
    LeakyReLU (not ReLU) is used throughout -- DCGAN found this crucial for
    keeping gradients flowing to the generator through the discriminator's
    negative activations, avoiding dead-neuron style discriminator collapse.'''
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)  # raw logits (no Sigmoid -- we use BCEWithLogitsLoss for numerical stability)


nc_gan = real_batch.shape[1]
G_gan = GAN_Generator(nc=nc_gan).to(DEVICE).apply(weights_init)
D_gan = GAN_Discriminator(nc=nc_gan).to(DEVICE).apply(weights_init)
print(G_gan)
print(D_gan)
""")

md(r"### 1.4 Loss Functions & Optimizers")

code(r"""
# Non-saturating minimax loss (Goodfellow et al.): the generator maximizes
# log(D(G(z))) instead of minimizing log(1 - D(G(z))) -- same fixed point,
# but far stronger gradients early in training when D easily rejects G.
gan_criterion = nn.BCEWithLogitsLoss()

# Adam with beta1=0.5 (not the default 0.9) is the standard DCGAN choice:
# high beta1 momentum tends to cause oscillation/instability in the
# adversarial min-max setting.
gan_opt_G = torch.optim.Adam(G_gan.parameters(), lr=2e-4, betas=(0.5, 0.999))
gan_opt_D = torch.optim.Adam(D_gan.parameters(), lr=2e-4, betas=(0.5, 0.999))

GAN_REAL_LABEL, GAN_FAKE_LABEL = 1.0, 0.0
gan_fixed_noise = torch.randn(64, GAN_NZ, device=DEVICE)  # fixed across training to visualize learning progress on the SAME latents
""")

md(r"### 1.5 Training Loop")

code(r"""
gan_scaler_G = torch.cuda.amp.GradScaler(enabled=USE_AMP)
gan_scaler_D = torch.cuda.amp.GradScaler(enabled=USE_AMP)
gan_loss_hist = {"G": AvgMeter(), "D": AvgMeter()}
gan_history = []

for epoch in range(GAN_EPOCHS):
    pbar = tqdm(gan_loader, desc=f"[GAN] epoch {epoch+1}/{GAN_EPOCHS}")
    for real, _ in pbar:
        real = real.to(DEVICE)
        b = real.size(0)

        # ---- Train Discriminator: maximize log(D(real)) + log(1 - D(G(z))) ----
        D_gan.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits_real = D_gan(real)
            loss_d_real = gan_criterion(logits_real, torch.full((b,), GAN_REAL_LABEL, device=DEVICE))
            noise = torch.randn(b, GAN_NZ, device=DEVICE)
            fake = G_gan(noise)
            logits_fake = D_gan(fake.detach())
            loss_d_fake = gan_criterion(logits_fake, torch.full((b,), GAN_FAKE_LABEL, device=DEVICE))
            loss_d = loss_d_real + loss_d_fake
        gan_scaler_D.scale(loss_d).backward()
        gan_scaler_D.step(gan_opt_D)
        gan_scaler_D.update()

        # ---- Train Generator: maximize log(D(G(z))) (non-saturating trick) ----
        G_gan.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits_fake_for_g = D_gan(fake)
            loss_g = gan_criterion(logits_fake_for_g, torch.full((b,), GAN_REAL_LABEL, device=DEVICE))
        gan_scaler_G.scale(loss_g).backward()
        gan_scaler_G.step(gan_opt_G)
        gan_scaler_G.update()

        gan_loss_hist["D"].update(loss_d.item())
        gan_loss_hist["G"].update(loss_g.item())
        pbar.set_postfix(loss_D=f"{loss_d.item():.3f}", loss_G=f"{loss_g.item():.3f}")

    gan_history.append((gan_loss_hist["G"].avg, gan_loss_hist["D"].avg))
    with torch.no_grad():
        G_gan.eval()
        sample = G_gan(gan_fixed_noise)
        G_gan.train()
    show_images(sample, title=f"GAN samples after epoch {epoch+1}")
    save_checkpoint({"G": G_gan.state_dict(), "D": D_gan.state_dict(), "epoch": epoch},
                     f"{CHECKPOINT_DIR}/gan_epoch{epoch+1}.pt")

print("Original GAN training complete.")
""")

md(r"### 1.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def gan_generate(n_samples: int = 16, seed: int = None) -> torch.Tensor:
    '''Clean generation entry point: noise-in, images-out ([-1, 1] range).'''
    if seed is not None:
        torch.manual_seed(seed)
    G_gan.eval()
    z = torch.randn(n_samples, GAN_NZ, device=DEVICE)
    imgs = G_gan(z)
    G_gan.train()
    return imgs

show_images(gan_generate(16), title="GAN inference: fresh samples", nrow=4)
""")

md(r"### 1.7 Visualization & Results")

code(r"""
# Loss curves: in a healthy GAN neither loss collapses to 0 -- both hover
# because D and G are constantly adapting to each other.
gs, ds = zip(*gan_history)
plt.figure(figsize=(6, 4))
plt.plot(gs, label="Generator loss")
plt.plot(ds, label="Discriminator loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("GAN training losses")
plt.show()

# Latent-space interpolation: walking a straight line in z-space and
# decoding each point shows whether the generator learned a *smooth*
# manifold (no sudden jumps) rather than memorizing isolated points.
@torch.no_grad()
def gan_interpolate(n_steps: int = 8):
    G_gan.eval()
    z1, z2 = torch.randn(1, GAN_NZ, device=DEVICE), torch.randn(1, GAN_NZ, device=DEVICE)
    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    imgs = G_gan(z_interp)
    G_gan.train()
    return imgs

show_images(gan_interpolate(8), title="Latent interpolation", nrow=8, figsize=(12, 2))
""")

md(r"### 1.8 User Input → Output Demo")

code(r"""
# "Interactive" cell: change N_SAMPLES_TO_GENERATE and RANDOM_SEED, re-run.
N_SAMPLES_TO_GENERATE = 25
RANDOM_SEED = 7

demo_samples = gan_generate(N_SAMPLES_TO_GENERATE, seed=RANDOM_SEED)
show_images(demo_samples, title=f"User demo: {N_SAMPLES_TO_GENERATE} samples, seed={RANDOM_SEED}", nrow=5)
""")

md(r"""
### 1.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** The non-saturating generator loss is what makes vanilla
  GAN training practical at all -- the original minimax loss gives
  vanishing gradients to `G` whenever `D` is confident, which is almost
  always early in training.
- ⚠️ **Warning.** No conditioning means you cannot ask this model for "a
  7" -- you only get a random sample from the learned distribution. This
  exact limitation motivates cGAN (next section).
- ⚠️ **Warning.** Vanilla DCGAN is prone to **mode collapse** (generator
  finds a few outputs that reliably fool D and stops exploring) and has no
  formal convergence guarantee -- both are addressed by later
  architectures in this notebook (WGAN-GP-style losses appear in
  StackGAN/BigGAN sections).

**This GAN solved the problem of: unconditional image generation --
learning to sample realistically from an unlabeled image distribution
given nothing but random noise.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 2 -- cGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="2"></a>
## 2. cGAN -- Class-conditional generation

### 2.1 Problem Statement & Why this GAN

Mirza & Osindero (2014) fixed the "you cannot ask for a 7" limitation of
Section 1 by feeding a **label `y`** into *both* `G` and `D`. The generator
now models `p(x | y)` -- generate a *specific requested class* -- and the
discriminator must judge realism *conditioned on that same label*, so it
can no longer be fooled by a realistic image of the wrong class. This is
the **class-conditional generation** problem: user picks the category, the
model draws it.

We train on FashionMNIST (10 clothing categories) so conditioning is
visually obvious (a "sneaker" looks nothing like a "coat").

### 2.2 Data Collection & Preparation
""")

code(r"""
CGAN_IMG_SIZE = 32
CGAN_BATCH_SIZE = 128
CGAN_NZ = 100
CGAN_N_CLASSES = 10
CGAN_EPOCHS = 5
CGAN_CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cgan_transform = T.Compose([T.Resize(CGAN_IMG_SIZE), T.ToTensor(), T.Normalize((0.5,), (0.5,))])

try:
    cgan_dataset = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=cgan_transform)
    print(f"Loaded real FashionMNIST: {len(cgan_dataset)} images")
except Exception as e:
    print(f"FashionMNIST download failed ({e}); falling back to synthetic per-class shapes.")
    rng = np.random.default_rng(SEED)
    n_per_class = 200
    imgs, labels = [], []
    yy, xx = torch.meshgrid(torch.arange(CGAN_IMG_SIZE), torch.arange(CGAN_IMG_SIZE), indexing="ij")
    for c in range(CGAN_N_CLASSES):
        for _ in range(n_per_class):
            cx, cy = rng.uniform(10, 22, 2)
            r = 3 + c  # radius encodes class -> a learnable, class-distinguishing synthetic signal
            blob = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2).float()
            imgs.append(blob * 2 - 1)
            labels.append(c)
    cgan_dataset = TensorDataset(torch.stack(imgs).unsqueeze(1), torch.tensor(labels))

cgan_loader = DataLoader(cgan_dataset, batch_size=CGAN_BATCH_SIZE, shuffle=True, drop_last=True)
real_batch, real_labels = next(iter(cgan_loader))
show_images(real_batch[:64], title="Real conditional dataset samples", nrow=8)
""")

md(r"### 2.3 Model Architecture")

code(r"""
class CGAN_Generator(nn.Module):
    '''Same DCGAN backbone as Section 1, but the class label is embedded
    into a vector and concatenated with the noise vector before the first
    transposed conv. This is the simplest and most common way to condition
    a generator: give it the extra information as extra input channels.'''
    def __init__(self, nz=CGAN_NZ, n_classes=CGAN_N_CLASSES, ngf=64, nc=1, embed_dim=50):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, y):
        y_emb = self.label_embed(y)                      # (B, embed_dim)
        zy = torch.cat([z, y_emb], dim=1)                 # (B, nz+embed_dim)
        return self.net(zy.view(zy.size(0), -1, 1, 1))


class CGAN_Discriminator(nn.Module):
    '''The label is embedded and projected to a full HxW "label map" that is
    concatenated as an extra image channel -- lets every conv layer see
    "what class am I judging" from the very first layer.'''
    def __init__(self, n_classes=CGAN_N_CLASSES, nc=1, ndf=64, img_size=CGAN_IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.label_embed = nn.Embedding(n_classes, img_size * img_size)
        self.net = nn.Sequential(
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        y_map = self.label_embed(y).view(-1, 1, self.img_size, self.img_size)
        xy = torch.cat([x, y_map], dim=1)
        return self.net(xy).view(-1)


nc_cgan = real_batch.shape[1]
G_cgan = CGAN_Generator(nc=nc_cgan).to(DEVICE).apply(weights_init)
D_cgan = CGAN_Discriminator(nc=nc_cgan).to(DEVICE).apply(weights_init)
""")

md(r"### 2.4 Loss Functions & Optimizers")

code(r"""
cgan_criterion = nn.BCEWithLogitsLoss()
cgan_opt_G = torch.optim.Adam(G_cgan.parameters(), lr=2e-4, betas=(0.5, 0.999))
cgan_opt_D = torch.optim.Adam(D_cgan.parameters(), lr=2e-4, betas=(0.5, 0.999))
cgan_fixed_noise = torch.randn(CGAN_N_CLASSES * 8, CGAN_NZ, device=DEVICE)
cgan_fixed_labels = torch.arange(CGAN_N_CLASSES, device=DEVICE).repeat_interleave(8)  # 8 samples per class, ordered -> easy visual check
""")

md(r"### 2.5 Training Loop")

code(r"""
cgan_history = []
for epoch in range(CGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter()}
    pbar = tqdm(cgan_loader, desc=f"[cGAN] epoch {epoch+1}/{CGAN_EPOCHS}")
    for real, y in pbar:
        real, y = real.to(DEVICE), y.to(DEVICE)
        b = real.size(0)

        D_cgan.zero_grad(set_to_none=True)
        logits_real = D_cgan(real, y)
        loss_d_real = cgan_criterion(logits_real, torch.ones(b, device=DEVICE))
        z = torch.randn(b, CGAN_NZ, device=DEVICE)
        fake_y = torch.randint(0, CGAN_N_CLASSES, (b,), device=DEVICE)  # ask for RANDOM classes, not the real batch's classes
        fake = G_cgan(z, fake_y)
        logits_fake = D_cgan(fake.detach(), fake_y)
        loss_d_fake = cgan_criterion(logits_fake, torch.zeros(b, device=DEVICE))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        cgan_opt_D.step()

        G_cgan.zero_grad(set_to_none=True)
        logits_fake_for_g = D_cgan(fake, fake_y)
        loss_g = cgan_criterion(logits_fake_for_g, torch.ones(b, device=DEVICE))
        loss_g.backward()
        cgan_opt_G.step()

        meters["D"].update(loss_d.item()); meters["G"].update(loss_g.item())
        pbar.set_postfix(loss_D=f"{loss_d.item():.3f}", loss_G=f"{loss_g.item():.3f}")

    cgan_history.append((meters["G"].avg, meters["D"].avg))
    with torch.no_grad():
        G_cgan.eval()
        sample = G_cgan(cgan_fixed_noise, cgan_fixed_labels)
        G_cgan.train()
    show_images(sample, title=f"cGAN samples after epoch {epoch+1} (rows = classes)", nrow=8)
    save_checkpoint({"G": G_cgan.state_dict(), "D": D_cgan.state_dict()}, f"{CHECKPOINT_DIR}/cgan_epoch{epoch+1}.pt")

print("cGAN training complete.")
""")

md(r"### 2.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def cgan_generate(class_id: int, n_samples: int = 8, seed: int = None) -> torch.Tensor:
    '''Generate n_samples of a user-specified class.'''
    if seed is not None:
        torch.manual_seed(seed)
    G_cgan.eval()
    z = torch.randn(n_samples, CGAN_NZ, device=DEVICE)
    y = torch.full((n_samples,), class_id, dtype=torch.long, device=DEVICE)
    imgs = G_cgan(z, y)
    G_cgan.train()
    return imgs

show_images(cgan_generate(class_id=7, n_samples=8), title=f"cGAN inference: class='{CGAN_CLASS_NAMES[7]}'", nrow=8, figsize=(10, 2))
""")

md(r"### 2.7 Visualization & Results")

code(r"""
gs, ds = zip(*cgan_history)
plt.figure(figsize=(6, 4)); plt.plot(gs, label="G"); plt.plot(ds, label="D")
plt.legend(); plt.title("cGAN training losses"); plt.xlabel("epoch"); plt.show()

# All classes side by side -- the core deliverable of conditional generation.
with torch.no_grad():
    G_cgan.eval()
    z = torch.randn(CGAN_N_CLASSES * 6, CGAN_NZ, device=DEVICE)
    y = torch.arange(CGAN_N_CLASSES, device=DEVICE).repeat_interleave(6)
    grid = G_cgan(z, y)
    G_cgan.train()
show_images(grid, title="One row per class: " + ", ".join(CGAN_CLASS_NAMES), nrow=6, figsize=(6, 12))
""")

md(r"### 2.8 User Input → Output Demo")

code(r"""
# Change this to any class name from CGAN_CLASS_NAMES and re-run.
USER_REQUESTED_CLASS = "Sneaker"
class_idx = CGAN_CLASS_NAMES.index(USER_REQUESTED_CLASS)
show_images(cgan_generate(class_idx, n_samples=16), title=f"User requested: '{USER_REQUESTED_CLASS}'", nrow=4)
""")

md(r"""
### 2.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Conditioning information can be injected almost anywhere
  (input concatenation, as here; label-conditional BatchNorm as in BigGAN;
  projection discriminator; cross-attention as in modern text-to-image
  models) -- the *choice of injection point* is one of the biggest levers
  for conditional-GAN quality.
- ⚠️ **Warning.** With a small/simple discriminator, cGAN can learn to
  mostly ignore the label and rely on the discriminator being lenient
  (**label leakage/ignoring**) -- always sanity-check that requesting
  different classes actually produces visibly different outputs (Section 2.7).

**This GAN solved the problem of: class-conditional generation --
letting the user choose *which* category to generate, not just generating
an arbitrary sample from the whole dataset.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 3 -- AC-GAN
# ---------------------------------------------------------------------------
md(r"""
<a id="3"></a>
## 3. AC-GAN -- Conditional generation + auxiliary classification

### 3.1 Problem Statement & Why this GAN

Odena et al. (2017) noticed that cGAN's discriminator only ever answers
"real or fake, *given* the label" -- it never has to *predict* the label
itself, so the gradient signal for "did you draw the right class" is
weak. AC-GAN's discriminator instead outputs **two heads**: real/fake
*and* a full class-probability distribution (an auxiliary classifier). The
generator is now pushed by two error signals -- fool the discriminator,
*and* be classified as the requested class -- which empirically yields
sharper, more class-faithful samples on harder, natural-image datasets.

We move to CIFAR-10 (real color photos, harder than MNIST/FashionMNIST) to
show AC-GAN's improvement matters most when the conditioning problem is
genuinely difficult.

### 3.2 Data Collection & Preparation
""")

code(r"""
# We use a 4-class CIFAR-10 subset (airplane, automobile, bird, cat) purely
# for training-time budget -- the architecture below scales to all 10
# classes unchanged if you set ACGAN_CLASS_IDS = list(range(10)).
ACGAN_CLASS_IDS = [0, 1, 2, 3]
ACGAN_CLASS_NAMES = ["airplane", "automobile", "bird", "cat"]
ACGAN_IMG_SIZE = 32
ACGAN_BATCH_SIZE = 128
ACGAN_NZ = 100
ACGAN_EPOCHS = 8

acgan_transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

try:
    full_cifar = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=acgan_transform)
    idx = [i for i, (_, y) in enumerate(full_cifar) if y in ACGAN_CLASS_IDS]
    acgan_dataset = torch.utils.data.Subset(full_cifar, idx)
    # remap original CIFAR labels {0,1,2,3} -> contiguous {0..3} (already contiguous here, but keep for generality)
    label_map = {orig: new for new, orig in enumerate(ACGAN_CLASS_IDS)}
    print(f"Loaded real CIFAR-10 subset: {len(acgan_dataset)} images across {len(ACGAN_CLASS_IDS)} classes")
except Exception as e:
    print(f"CIFAR-10 download failed ({e}); falling back to synthetic colored-shape classes.")
    rng = np.random.default_rng(SEED)
    n_per_class, imgs, labels = 300, [], []
    for c in range(len(ACGAN_CLASS_IDS)):
        color = torch.tensor([[1., -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1]][c])
        for _ in range(n_per_class):
            img = torch.randn(3, ACGAN_IMG_SIZE, ACGAN_IMG_SIZE) * 0.05 + color.view(3, 1, 1)
            imgs.append(img.clamp(-1, 1)); labels.append(c)
    acgan_dataset = TensorDataset(torch.stack(imgs), torch.tensor(labels))
    label_map = {c: c for c in range(len(ACGAN_CLASS_IDS))}

class RemapLabels(Dataset):
    '''Thin wrapper: remap arbitrary source labels to contiguous 0..K-1 ids the Embedding layers expect.'''
    def __init__(self, base, label_map):
        self.base, self.label_map = base, label_map
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, self.label_map[int(y)]

acgan_dataset = RemapLabels(acgan_dataset, label_map)
acgan_loader = DataLoader(acgan_dataset, batch_size=ACGAN_BATCH_SIZE, shuffle=True, drop_last=True)
real_batch, real_labels = next(iter(acgan_loader))
show_images(real_batch[:64], title="Real AC-GAN training samples", nrow=8)
""")

md(r"### 3.3 Model Architecture")

code(r"""
ACGAN_N_CLASSES = len(ACGAN_CLASS_IDS)

class ACGAN_Generator(nn.Module):
    '''Same conditioning-by-concatenation idea as cGAN, scaled up with one
    extra upsampling stage (32x32 RGB instead of grayscale) and more
    channels since natural images carry far more information than digits.'''
    def __init__(self, nz=ACGAN_NZ, n_classes=ACGAN_N_CLASSES, ngf=64, nc=3, embed_dim=50):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, y):
        y_emb = self.label_embed(y)
        zy = torch.cat([z, y_emb], dim=1)
        return self.net(zy.view(zy.size(0), -1, 1, 1))


class ACGAN_Discriminator(nn.Module):
    '''Shared convolutional trunk -> two heads:
    - `adv_head`: real/fake logit (unconditional -- unlike cGAN, AC-GAN's
      discriminator is NOT told the label up front; it has to recover it).
    - `cls_head`: class logits over ACGAN_N_CLASSES, trained with
      cross-entropy on BOTH real images (their true label) and fake images
      (the label G was asked to draw) -- this is what forces G to be
      class-faithful, not just realistic.'''
    def __init__(self, n_classes=ACGAN_N_CLASSES, nc=3, ndf=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
        )
        self.adv_head = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.cls_head = nn.Conv2d(ndf * 4, n_classes, 4, 1, 0, bias=False)

    def forward(self, x):
        feat = self.trunk(x)
        return self.adv_head(feat).view(-1), self.cls_head(feat).view(feat.size(0), -1)


G_acgan = ACGAN_Generator().to(DEVICE).apply(weights_init)
D_acgan = ACGAN_Discriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 3.4 Loss Functions & Optimizers")

code(r"""
acgan_adv_criterion = nn.BCEWithLogitsLoss()   # real vs fake
acgan_cls_criterion = nn.CrossEntropyLoss()    # which class
acgan_opt_G = torch.optim.Adam(G_acgan.parameters(), lr=2e-4, betas=(0.5, 0.999))
acgan_opt_D = torch.optim.Adam(D_acgan.parameters(), lr=2e-4, betas=(0.5, 0.999))
acgan_fixed_noise = torch.randn(ACGAN_N_CLASSES * 8, ACGAN_NZ, device=DEVICE)
acgan_fixed_labels = torch.arange(ACGAN_N_CLASSES, device=DEVICE).repeat_interleave(8)
""")

md(r"### 3.5 Training Loop")

code(r"""
acgan_history = []
for epoch in range(ACGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter(), "acc": AvgMeter()}
    pbar = tqdm(acgan_loader, desc=f"[AC-GAN] epoch {epoch+1}/{ACGAN_EPOCHS}")
    for real, y in pbar:
        real, y = real.to(DEVICE), y.to(DEVICE)
        b = real.size(0)

        # ---- Discriminator: adversarial loss (real vs fake) + classification loss (on REAL images only here) ----
        D_acgan.zero_grad(set_to_none=True)
        adv_real, cls_real = D_acgan(real)
        loss_d_adv_real = acgan_adv_criterion(adv_real, torch.ones(b, device=DEVICE))
        loss_d_cls_real = acgan_cls_criterion(cls_real, y)

        z = torch.randn(b, ACGAN_NZ, device=DEVICE)
        fake_y = torch.randint(0, ACGAN_N_CLASSES, (b,), device=DEVICE)
        fake = G_acgan(z, fake_y)
        adv_fake, cls_fake = D_acgan(fake.detach())
        loss_d_adv_fake = acgan_adv_criterion(adv_fake, torch.zeros(b, device=DEVICE))
        # Classification loss on fakes too: teaches D to recognize class-defining
        # features regardless of real/fake, which in turn sharpens the
        # gradient G receives about "did I draw the requested class".
        loss_d_cls_fake = acgan_cls_criterion(cls_fake, fake_y)

        loss_d = loss_d_adv_real + loss_d_adv_fake + loss_d_cls_real + loss_d_cls_fake
        loss_d.backward()
        acgan_opt_D.step()

        # ---- Generator: fool D AND be classified as the requested class ----
        G_acgan.zero_grad(set_to_none=True)
        adv_fake_g, cls_fake_g = D_acgan(fake)
        loss_g_adv = acgan_adv_criterion(adv_fake_g, torch.ones(b, device=DEVICE))
        loss_g_cls = acgan_cls_criterion(cls_fake_g, fake_y)
        loss_g = loss_g_adv + loss_g_cls
        loss_g.backward()
        acgan_opt_G.step()

        acc = (cls_real.argmax(1) == y).float().mean().item()
        meters["D"].update(loss_d.item()); meters["G"].update(loss_g.item()); meters["acc"].update(acc)
        pbar.set_postfix(loss_D=f"{loss_d.item():.3f}", loss_G=f"{loss_g.item():.3f}", D_cls_acc=f"{acc:.2f}")

    acgan_history.append((meters["G"].avg, meters["D"].avg, meters["acc"].avg))
    with torch.no_grad():
        G_acgan.eval()
        sample = G_acgan(acgan_fixed_noise, acgan_fixed_labels)
        G_acgan.train()
    show_images(sample, title=f"AC-GAN samples after epoch {epoch+1} (rows=classes)", nrow=8)
    save_checkpoint({"G": G_acgan.state_dict(), "D": D_acgan.state_dict()}, f"{CHECKPOINT_DIR}/acgan_epoch{epoch+1}.pt")

print("AC-GAN training complete.")
""")

md(r"### 3.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def acgan_generate(class_id: int, n_samples: int = 8, seed: int = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    G_acgan.eval()
    z = torch.randn(n_samples, ACGAN_NZ, device=DEVICE)
    y = torch.full((n_samples,), class_id, dtype=torch.long, device=DEVICE)
    imgs = G_acgan(z, y)
    G_acgan.train()
    return imgs

show_images(acgan_generate(2, 8), title=f"AC-GAN inference: class='{ACGAN_CLASS_NAMES[2]}'", nrow=8, figsize=(10, 2))
""")

md(r"### 3.7 Visualization & Results")

code(r"""
gs, ds, accs = zip(*acgan_history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(gs, label="G"); axes[0].plot(ds, label="D"); axes[0].legend(); axes[0].set_title("Losses")
axes[1].plot(accs); axes[1].set_title("D's auxiliary classifier accuracy on real images")
plt.show()

# Quantitative sanity check: does the AUXILIARY CLASSIFIER agree that
# generated images match the class they were conditioned on? This is a
# cheap proxy for the "Inception Score"-style self-consistency metrics
# used in the AC-GAN paper.
with torch.no_grad():
    D_acgan.eval()
    z = torch.randn(200, ACGAN_NZ, device=DEVICE)
    y_req = torch.randint(0, ACGAN_N_CLASSES, (200,), device=DEVICE)
    fakes = G_acgan(z, y_req)
    _, cls_logits = D_acgan(fakes)
    agreement = (cls_logits.argmax(1) == y_req).float().mean().item()
    D_acgan.train()
print(f"Self-consistency (D's classifier agrees with requested class): {agreement*100:.1f}%")
""")

md(r"### 3.8 User Input → Output Demo")

code(r"""
USER_REQUESTED_CLASS = "cat"
class_idx = ACGAN_CLASS_NAMES.index(USER_REQUESTED_CLASS)
show_images(acgan_generate(class_idx, n_samples=16), title=f"User requested: '{USER_REQUESTED_CLASS}'", nrow=4)
""")

md(r"""
### 3.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Training the auxiliary classifier on *both* real and fake
  images (not just real, as some simplified tutorials do) is what actually
  gives the generator a useful classification gradient -- classifying only
  reals gives G no direct feedback about its own class-faithfulness.
- ⚠️ **Warning.** AC-GAN is known to have a subtle failure mode: with many
  classes, the auxiliary loss can dominate and push the generator toward
  **low-diversity, highly class-separable but less realistic** samples
  (the discriminator's easiest way to raise classification accuracy is
  often to make classes maximally different-looking, not maximally
  photorealistic).

**This GAN solved the problem of: class-conditional generation with an
explicit, learnable notion of "class identity" baked into the
discriminator -- improving class-faithfulness over plain cGAN on harder,
natural-image data.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 4 -- PROGRESSIVE GAN
# ---------------------------------------------------------------------------
md(r"""
<a id="4"></a>
## 4. Progressive GAN -- Stable high-resolution generation via progressive growing

### 4.1 Problem Statement & Why this GAN

Karras et al. (2018) observed that training a deep GAN directly at high
resolution (1024x1024) is extremely unstable: early in training neither
network has learned coarse structure yet, and the discriminator can
trivially spot fine-detail artifacts, giving the generator no useful
gradient. **Progressive growing** fixes this by starting both networks at
a tiny 4x4 resolution, training to convergence, then **adding a new block**
that doubles resolution (4->8->16->...), fading it in smoothly with a
blend weight `alpha` so the new layers never see a "shocking" untrained
input. This is the **stable high-resolution generation** problem.

We grow 4 -> 8 -> 16 -> 32 on CIFAR-10 (32x32 is already CIFAR-10's native
resolution, so this section demonstrates the *mechanism* end-to-end
quickly; the real ProGAN paper grows to 1024x1024 over ~9 stages and
many GPU-days).

⚠️ **Warning -- Compute.** Even this 4-stage lightweight version trains
4 short sub-schedules; expect the longest single-GPU run in the "easy"
half of this notebook. Reduce `PGAN_EPOCHS_PER_STAGE` first if short on time.

### 4.2 Data Collection & Preparation
""")

code(r"""
PGAN_FINAL_SIZE = 32                    # 4 -> 8 -> 16 -> 32
PGAN_STAGES = [4, 8, 16, 32]
PGAN_NZ = 128
PGAN_BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32}   # shrink batch as resolution (and memory) grows
PGAN_EPOCHS_PER_STAGE = 2                # x2 substages each (fade-in + stabilize) -- see training loop

try:
    pgan_base = torchvision.datasets.CIFAR10(root="data", train=True, download=True,
                                              transform=T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]))
    print(f"Loaded real CIFAR-10: {len(pgan_base)} images")
except Exception as e:
    print(f"CIFAR-10 download failed ({e}); falling back to synthetic textures.")
    rng = np.random.default_rng(SEED)
    imgs = torch.rand(2000, 3, PGAN_FINAL_SIZE, PGAN_FINAL_SIZE) * 2 - 1
    pgan_base = TensorDataset(imgs, torch.zeros(2000, dtype=torch.long))

def pgan_loader_for_size(size):
    '''Build a DataLoader that resizes every image to the CURRENT stage's
    resolution on the fly -- this is what lets one dataset feed every
    growth stage without re-downloading/re-processing.'''
    resize = T.Resize(size)
    class Resized(Dataset):
        def __len__(self_): return len(pgan_base)
        def __getitem__(self_, i):
            x, y = pgan_base[i]
            return resize(x), y
    bs = PGAN_BATCH_SIZES[size]
    return DataLoader(Resized(), batch_size=bs, shuffle=True, drop_last=True)

show_images(next(iter(pgan_loader_for_size(32)))[0][:32], title="Real data (final 32x32 stage)", nrow=8)
""")

md(r"### 4.3 Model Architecture")

code(r"""
def pgan_pixel_norm(x, eps=1e-8):
    '''Pixelwise feature normalization (ProGAN's replacement for BatchNorm
    in the generator): normalize each pixel's channel vector to unit norm.
    Why: prevents signal magnitudes from escalating uncontrollably across
    the many blocks of a progressively-growing generator, without the
    batch-statistics coupling of BatchNorm (which the paper found hurt
    unbounded stochastic growing).'''
    return x / torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


class PGAN_ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = pgan_pixel_norm(self.act(self.c1(x)))
        x = pgan_pixel_norm(self.act(self.c2(x)))
        return x


class PGAN_Generator(nn.Module):
    '''Progressively-growable generator. `to_rgb` layers exist for every
    stage; `forward(z, stage, alpha)` runs the network up to `stage` and,
    if alpha < 1, cross-fades between the *upsampled previous-stage* RGB
    output and the *new stage's* RGB output -- the core trick that avoids
    destabilizing shocks when a new block is introduced mid-training.'''
    def __init__(self, nz=PGAN_NZ, stages=PGAN_STAGES, base_ch=256):
        super().__init__()
        self.stages = stages
        chs = {s: max(32, base_ch // (2 ** i)) for i, s in enumerate(stages)}   # shrink channels as resolution grows (standard ProGAN schedule)
        self.chs = chs
        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, chs[stages[0]], stages[0], 1, 0), nn.LeakyReLU(0.2, True),
        )  # z -> 4x4 base feature map
        self.blocks = nn.ModuleList()   # one growth block per stage AFTER the first
        self.to_rgb = nn.ModuleList([nn.Conv2d(chs[stages[0]], 3, 1)])
        in_c = chs[stages[0]]
        for s in stages[1:]:
            out_c = chs[s]
            self.blocks.append(PGAN_ConvBlock(in_c, out_c))
            self.to_rgb.append(nn.Conv2d(out_c, 3, 1))
            in_c = out_c

    def forward(self, z, stage_idx: int, alpha: float = 1.0):
        x = pgan_pixel_norm(self.init(z.view(z.size(0), -1, 1, 1)))
        if stage_idx == 0:
            return torch.tanh(self.to_rgb[0](x))
        prev_rgb = None
        for i in range(stage_idx):
            if i == stage_idx - 1:
                prev_rgb = F.interpolate(self.to_rgb[i](x), scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.blocks[i](x)
        new_rgb = self.to_rgb[stage_idx](x)
        if alpha >= 1.0 or prev_rgb is None:
            return torch.tanh(new_rgb)
        return torch.tanh(alpha * new_rgb + (1 - alpha) * prev_rgb)   # fade-in blend


class PGAN_Discriminator(nn.Module):
    '''Mirror of the generator: `from_rgb` layers per stage, and the SAME
    fade-in blending on the way IN so the discriminator also sees a smooth
    transition, not a sudden new block.'''
    def __init__(self, stages=PGAN_STAGES, base_ch=256):
        super().__init__()
        self.stages = stages
        chs = {s: max(32, base_ch // (2 ** i)) for i, s in enumerate(stages)}
        self.from_rgb = nn.ModuleList([nn.Conv2d(3, chs[s], 1) for s in stages])
        self.blocks = nn.ModuleList()
        rev = list(reversed(stages))
        for i in range(len(rev) - 1):
            in_c, out_c = chs[rev[i]], chs[rev[i + 1]]
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_c, out_c, 3, 1, 1), nn.LeakyReLU(0.2, True),
            ))
        self.final = nn.Sequential(
            nn.Conv2d(chs[stages[0]], chs[stages[0]], 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chs[stages[0]], 1, stages[0], 1, 0),
        )

    def forward(self, img, stage_idx: int, alpha: float = 1.0):
        x = F.leaky_relu(self.from_rgb[stage_idx](img), 0.2)
        if stage_idx > 0 and alpha < 1.0:
            downsampled = F.avg_pool2d(img, 2)
            skip = F.leaky_relu(self.from_rgb[stage_idx - 1](downsampled), 0.2)
        for i, block_i in enumerate(range(stage_idx - 1, -1, -1)):
            x = self.blocks[len(self.stages) - 2 - block_i](x)
            x = F.avg_pool2d(x, 2)
            if i == 0 and stage_idx > 0 and alpha < 1.0:
                x = alpha * x + (1 - alpha) * skip
        return self.final(x).view(-1)


G_pgan = PGAN_Generator().to(DEVICE).apply(weights_init)
D_pgan = PGAN_Discriminator().to(DEVICE).apply(weights_init)
print(f"ProGAN stages: {PGAN_STAGES}")
""")

md(r"### 4.4 Loss Functions & Optimizers")

code(r"""
# WGAN-GP loss: ProGAN's original paper uses WGAN-GP (Gulrajani et al.)
# rather than the vanilla BCE minimax loss, because the gradient penalty
# keeps the discriminator 1-Lipschitz, which is empirically far more
# stable across the many growth-stage transitions than BCE.
def pgan_gradient_penalty(D, real, fake, stage_idx, alpha, device=DEVICE):
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = D(interp, stage_idx, alpha)
    grad = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    grad = grad.view(b, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

PGAN_LAMBDA_GP = 10.0
PGAN_DRIFT = 1e-3   # small penalty keeping D's outputs near 0 (prevents unbounded drift under WGAN's unconstrained loss)
pgan_opt_G = torch.optim.Adam(G_pgan.parameters(), lr=1e-3, betas=(0.0, 0.99))
pgan_opt_D = torch.optim.Adam(D_pgan.parameters(), lr=1e-3, betas=(0.0, 0.99))
""")

md(r"### 4.5 Training Loop")

code(r"""
# ---------------------------------------------------------------------------
# 4.5a Lightweight from-scratch progressive training
# ---------------------------------------------------------------------------
# Schedule per stage: first PGAN_EPOCHS_PER_STAGE epochs FADE IN (alpha: 0->1),
# then PGAN_EPOCHS_PER_STAGE epochs STABILIZE (alpha=1). Stage 0 (4x4) has no
# fade-in (nothing to fade in from).
pgan_fixed_noise = torch.randn(16, PGAN_NZ, device=DEVICE)

for stage_idx, size in enumerate(PGAN_STAGES):
    loader = pgan_loader_for_size(size)
    n_batches_total = len(loader) * PGAN_EPOCHS_PER_STAGE
    substages = [("stabilize", PGAN_EPOCHS_PER_STAGE)] if stage_idx == 0 else \
                [("fade-in", PGAN_EPOCHS_PER_STAGE), ("stabilize", PGAN_EPOCHS_PER_STAGE)]

    for phase, n_epochs in substages:
        for epoch in range(n_epochs):
            pbar = tqdm(loader, desc=f"[ProGAN] {size}x{size} {phase} epoch {epoch+1}/{n_epochs}")
            for step, (real, _) in enumerate(pbar):
                real = real.to(DEVICE)
                b = real.size(0)
                if phase == "fade-in":
                    alpha = min(1.0, (epoch * len(loader) + step) / max(1, n_batches_total))
                else:
                    alpha = 1.0

                # ---- Discriminator (WGAN-GP critic) ----
                D_pgan.zero_grad(set_to_none=True)
                z = torch.randn(b, PGAN_NZ, device=DEVICE)
                fake = G_pgan(z, stage_idx, alpha)
                d_real = D_pgan(real, stage_idx, alpha)
                d_fake = D_pgan(fake.detach(), stage_idx, alpha)
                gp = pgan_gradient_penalty(D_pgan, real, fake.detach(), stage_idx, alpha)
                loss_d = d_fake.mean() - d_real.mean() + PGAN_LAMBDA_GP * gp + PGAN_DRIFT * (d_real ** 2).mean()
                loss_d.backward()
                pgan_opt_D.step()

                # ---- Generator ----
                G_pgan.zero_grad(set_to_none=True)
                d_fake_g = D_pgan(fake, stage_idx, alpha)
                loss_g = -d_fake_g.mean()
                loss_g.backward()
                pgan_opt_G.step()

                pbar.set_postfix(alpha=f"{alpha:.2f}", loss_D=f"{loss_d.item():.2f}", loss_G=f"{loss_g.item():.2f}")

        with torch.no_grad():
            G_pgan.eval()
            sample = G_pgan(pgan_fixed_noise, stage_idx, alpha=1.0)
            G_pgan.train()
        show_images(sample, title=f"ProGAN samples -- stage {size}x{size} ({phase} done)", nrow=4)

    save_checkpoint({"G": G_pgan.state_dict(), "D": D_pgan.state_dict(), "stage_idx": stage_idx},
                     f"{CHECKPOINT_DIR}/pgan_stage_{size}.pt")

print("Progressive GAN growth complete: 4x4 -> 32x32.")
""")

code(r"""
# ---------------------------------------------------------------------------
# 4.5b Loading strong pretrained weights (inference-only mode)
# ---------------------------------------------------------------------------
# The official NVIDIA ProGAN checkpoints (e.g. CelebA-HQ 1024x1024) are
# distributed as pickled TensorFlow graphs / torch_utils-dependent .pkl
# files via NVlabs' repos, and Facebook Research's pytorch_GAN_zoo package
# offers a torch.hub-loadable PyTorch port. We attempt that here; if there
# is no network access this simply reports so and the notebook continues
# using the from-scratch model trained above.
def try_load_pretrained_progan():
    try:
        pgan_pretrained = torch.hub.load(
            "facebookresearch/pytorch_GAN_zoo:hub", "PGAN",
            model_name="celebAHQ-512", pretrained=True, useGPU=(DEVICE.type == "cuda"),
        )
        print("Loaded official pretrained Progressive GAN (CelebA-HQ 512x512).")
        return pgan_pretrained
    except Exception as e:
        print(f"Pretrained ProGAN unavailable in this environment ({e}). "
              f"Using the from-scratch model trained above instead.")
        return None

pgan_pretrained = try_load_pretrained_progan()
""")

md(r"### 4.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def pgan_generate(n_samples: int = 16, seed: int = None, use_pretrained: bool = False) -> torch.Tensor:
    '''Unified generation function: uses the official pretrained model if it
    was loaded successfully AND the caller asks for it, otherwise falls
    back to our from-scratch model at the final trained stage (32x32).'''
    if seed is not None:
        torch.manual_seed(seed)
    if use_pretrained and pgan_pretrained is not None:
        noise, _ = pgan_pretrained.buildNoiseData(n_samples)
        with torch.no_grad():
            imgs = pgan_pretrained.test(noise)
        return imgs
    G_pgan.eval()
    z = torch.randn(n_samples, PGAN_NZ, device=DEVICE)
    imgs = G_pgan(z, stage_idx=len(PGAN_STAGES) - 1, alpha=1.0)
    G_pgan.train()
    return imgs

show_images(pgan_generate(16), title="ProGAN inference (from-scratch, 32x32)", nrow=4)
""")

md(r"### 4.7 Visualization & Results")

code(r"""
# Growth progression: show the SAME latent vectors decoded at every
# resolution stage side by side -- visualizes exactly what "progressive"
# growing means for a fixed identity in latent space.
with torch.no_grad():
    G_pgan.eval()
    z_demo = torch.randn(4, PGAN_NZ, device=DEVICE)
    fig, axes = plt.subplots(4, len(PGAN_STAGES), figsize=(2.2 * len(PGAN_STAGES), 8))
    for col, size in enumerate(PGAN_STAGES):
        imgs = denorm(G_pgan(z_demo, stage_idx=col, alpha=1.0)).cpu()
        for row in range(4):
            ax = axes[row, col]
            ax.imshow(imgs[row].permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{size}x{size}")
    plt.suptitle("Same latent code decoded at each growth stage")
    plt.tight_layout(); plt.show()
    G_pgan.train()
""")

md(r"### 4.8 User Input → Output Demo")

code(r"""
N_SAMPLES_TO_GENERATE = 9
RANDOM_SEED = 123
USE_PRETRAINED_IF_AVAILABLE = True

demo = pgan_generate(N_SAMPLES_TO_GENERATE, seed=RANDOM_SEED, use_pretrained=USE_PRETRAINED_IF_AVAILABLE)
show_images(demo, title="ProGAN user demo", nrow=3)
""")

md(r"""
### 4.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Progressive growing is fundamentally a **curriculum
  learning** strategy for GANs: solve the easy, low-frequency coarse-shape
  problem first (4x4), then gradually add the harder, high-frequency
  detail problem, instead of asking a randomly initialized network to
  solve both simultaneously.
- 🚀 **Pro Tip.** The `alpha` fade-in schedule is the single most important
  hyperparameter here -- fading too fast reintroduces the instability
  progressive growing was designed to avoid; too slow wastes compute.
- ⚠️ **Warning.** ProGAN has no built-in mechanism for *disentangled,
  controllable* style (e.g. "keep pose, change hair color") -- that
  capability is exactly what StyleGAN (next section) adds on top of this
  same progressive-growing backbone.

**This GAN solved the problem of: stable training at (and generation of)
high resolutions, by growing both networks gradually instead of training
a deep net at full resolution from a random initialization.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 5 -- STYLEGAN / STYLEGAN2
# ---------------------------------------------------------------------------
md(r"""
<a id="5"></a>
## 5. StyleGAN (StyleGAN2) -- Photorealistic controllable face generation

### 5.1 Problem Statement & Why this GAN

Karras et al. (2019, 2020) asked a different question than ProGAN: not
just "how do we generate high resolution images stably" but **"how do we
give the user fine, disentangled control over the generated image"**
(e.g. change hairstyle without changing pose or identity). StyleGAN
replaces the "feed noise directly into the network" idea with a **mapping
network** `z -> w` (an intermediate, more disentangled latent space) whose
`w` then **modulates the statistics of every convolution** in a **synthesis
network** that starts from a *learned constant*, not from `z` itself.
StyleGAN2 additionally replaces AdaIN with **weight demodulation** (removes
"droplet" artifacts) and adds **stochastic noise injection** for
fine, non-structural detail (individual hair strands, skin pores). This is
the **photorealistic, controllable generation** problem.

Note this section intentionally does **not** use progressive-growing
training (that was Section 4) -- StyleGAN2 explicitly replaced progressive
growing with a fixed-architecture, skip-connection generator trained
directly at the target resolution, which is what we implement below.

### 5.2 Data Collection & Preparation
""")

code(r"""
SGAN2_IMG_SIZE = 64          # StyleGAN2 the paper trains up to 1024; we use 64 to fit a single-GPU/CPU demo
SGAN2_Z_DIM = 128
SGAN2_W_DIM = 128
SGAN2_BATCH_SIZE = 32
SGAN2_EPOCHS = 6

sgan2_transform = T.Compose([T.Resize(SGAN2_IMG_SIZE), T.CenterCrop(SGAN2_IMG_SIZE),
                              T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

def make_synthetic_faces(n=2000, size=SGAN2_IMG_SIZE, seed=SEED):
    '''Procedural "face-like" images (oval head + eyes + mouth, randomized
    color/position/size) used ONLY when CelebA cannot be downloaded, so the
    rest of this section (style mixing, truncation, interpolation) still
    has something meaningful and visually checkable to work with offline.'''
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    imgs = np.ones((n, size, size, 3), dtype=np.float32)
    for i in range(n):
        skin = rng.uniform(0.5, 0.95, 3)
        bg = rng.uniform(0.1, 0.6, 3)
        img = np.ones((size, size, 3), dtype=np.float32) * bg
        cx, cy = size / 2 + rng.uniform(-3, 3), size / 2 + rng.uniform(-3, 3)
        rx, ry = size * rng.uniform(0.28, 0.36), size * rng.uniform(0.34, 0.42)
        face = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        img[face] = skin
        eye_dx, eye_dy = rx * 0.45, ry * 0.15
        for sgn in (-1, 1):
            ex, ey = cx + sgn * eye_dx, cy - eye_dy
            eye = (xx - ex) ** 2 + (yy - ey) ** 2 <= (size * 0.035) ** 2
            img[eye] = rng.uniform(0.0, 0.2, 3)
        mouth = (((xx - cx) / (rx * 0.5)) ** 2 + ((yy - (cy + ry * 0.45)) / (ry * 0.12)) ** 2) <= 1
        img[mouth] = rng.uniform(0.3, 0.6, 3) * [1.0, 0.3, 0.3]
        imgs[i] = img
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    return imgs * 2 - 1  # normalize to [-1, 1]

try:
    sgan2_base = torchvision.datasets.CelebA(root="data", split="train", download=True, transform=sgan2_transform)
    sgan2_dataset = torch.utils.data.Subset(sgan2_base, range(min(len(sgan2_base), 20000)))  # cap for fast epochs
    print(f"Loaded real CelebA: using {len(sgan2_dataset)} images")
except Exception as e:
    print(f"CelebA download failed ({e}); falling back to procedurally generated synthetic faces. "
          f"For real face generation, mount/download CelebA-HQ or FFHQ and point sgan2_dataset at it.")
    faces = make_synthetic_faces(2000)
    sgan2_dataset = TensorDataset(faces, torch.zeros(len(faces), dtype=torch.long))

sgan2_loader = DataLoader(sgan2_dataset, batch_size=SGAN2_BATCH_SIZE, shuffle=True, drop_last=True)
real_batch = next(iter(sgan2_loader))[0]
show_images(real_batch[:32], title="StyleGAN2 training data sample", nrow=8)
""")

md(r"### 5.3 Model Architecture")

code(r"""
class ModulatedConv2d(nn.Module):
    '''The core StyleGAN2 operation. Instead of AdaIN (normalize activations,
    then scale/shift by style -- StyleGAN1, which produced "droplet"
    artifacts), StyleGAN2 modulates the CONVOLUTION WEIGHTS themselves by
    the style vector `w`, then demodulates (renormalizes) them so each
    output feature map has unit expected variance. Implemented via the
    standard "grouped convolution" trick: build one modulated weight
    tensor PER SAMPLE in the batch, then run a single grouped conv2d.'''
    def __init__(self, in_c, out_c, kernel_size, style_dim, demodulate=True, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.demodulate = demodulate
        self.k = kernel_size
        self.padding = kernel_size // 2
        self.in_c, self.out_c = in_c, out_c
        self.weight = nn.Parameter(torch.randn(out_c, in_c, kernel_size, kernel_size))
        self.style_fc = nn.Linear(style_dim, in_c)   # maps w -> per-input-channel modulation scale
        nn.init.ones_(self.style_fc.bias)             # start modulation near 1.0 (identity) for training stability
        self.scale = 1 / math.sqrt(in_c * kernel_size ** 2)   # equalized learning rate (StyleGAN trick: keep per-layer gradient scale comparable)

    def forward(self, x, w):
        b, in_c, h, wid = x.shape
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h, wid = h * 2, wid * 2
        style = self.style_fc(w).view(b, 1, in_c, 1, 1)               # (b, 1, in_c, 1, 1)
        weight = self.scale * self.weight.unsqueeze(0) * style        # modulate: (b, out_c, in_c, k, k)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * demod
        weight = weight.view(b * self.out_c, in_c, self.k, self.k)
        x = x.reshape(1, b * in_c, h, wid)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.view(b, self.out_c, out.shape[-2], out.shape[-1])


class NoiseInjection(nn.Module):
    '''Per-pixel Gaussian noise, scaled by a LEARNED per-channel weight.
    Why: gives the generator a cheap, stochastic source of fine detail
    (individual hair strands, skin texture) that need not be encoded in
    `w` at all -- this is what lets StyleGAN2 vary micro-texture between
    samples of the *same* w (see Section 5.7).'''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class StyleConvBlock(nn.Module):
    def __init__(self, in_c, out_c, style_dim, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_c, out_c, 3, style_dim, upsample=upsample)
        self.noise = NoiseInjection(out_c)
        self.bias = nn.Parameter(torch.zeros(1, out_c, 1, 1))
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        x = self.conv(x, w)
        x = self.noise(x)
        return self.act(x + self.bias)


class ToRGB(nn.Module):
    '''Un-demodulated 1x1 modulated conv to 3 channels + additive skip from
    the previous (upsampled) RGB output. Summing RGB contributions from
    EVERY resolution (rather than only the final one) is StyleGAN2's
    "skip generator" design -- it is what let the authors remove
    progressive-growing training while keeping training stable.'''
    def __init__(self, in_c, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_c, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, w, skip=None):
        out = self.conv(x, w) + self.bias
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class StyleGAN2_MappingNetwork(nn.Module):
    '''z -> w. A deep MLP is used (instead of using z directly to modulate
    convs) because the *learned* w-space is empirically far more linear /
    disentangled than the fixed Gaussian z-space -- interpolating in w
    produces much more semantically smooth results (Section 5.7).'''
    def __init__(self, z_dim=SGAN2_Z_DIM, w_dim=SGAN2_W_DIM, n_layers=4):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2, True)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)   # normalize z before mapping (keeps mapping-net input scale fixed)
        return self.net(z)


SGAN2_RESOLUTIONS = [4, 8, 16, 32, 64]
SGAN2_CHANNELS = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32}


class StyleGAN2_Synthesis(nn.Module):
    '''Builds the image from a LEARNED CONSTANT (not from z!) by repeatedly
    applying style-modulated convolutions and upsampling, summing a toRGB
    contribution at every resolution along the way.'''
    def __init__(self, style_dim=SGAN2_W_DIM, resolutions=SGAN2_RESOLUTIONS, channels=SGAN2_CHANNELS):
        super().__init__()
        self.resolutions = resolutions
        self.const = nn.Parameter(torch.randn(1, channels[resolutions[0]], resolutions[0], resolutions[0]))
        self.first_conv = StyleConvBlock(channels[resolutions[0]], channels[resolutions[0]], style_dim)
        self.first_to_rgb = ToRGB(channels[resolutions[0]], style_dim)
        self.up_blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_c = channels[resolutions[0]]
        for res in resolutions[1:]:
            out_c = channels[res]
            self.up_blocks.append(StyleConvBlock(in_c, out_c, style_dim, upsample=True))
            self.to_rgbs.append(ToRGB(out_c, style_dim))
            in_c = out_c
        self.num_layers = 1 + len(self.up_blocks)   # number of distinct style inputs consumed (used for style mixing)

    def forward(self, ws):
        '''`ws` is either a single (B, w_dim) tensor (broadcast to every
        layer) or a list of length `num_layers` of (B, w_dim) tensors, one
        per layer -- the latter is what enables STYLE MIXING (Section 5.7):
        using coarse-layer styles from one w and fine-layer styles from
        another.'''
        if torch.is_tensor(ws):
            ws = [ws] * self.num_layers
        b = ws[0].size(0)
        x = self.const.expand(b, -1, -1, -1)
        x = self.first_conv(x, ws[0])
        rgb = self.first_to_rgb(x, ws[0])
        for i, (block, to_rgb) in enumerate(zip(self.up_blocks, self.to_rgbs)):
            x = block(x, ws[i + 1])
            rgb = to_rgb(x, ws[i + 1], skip=rgb)
        return torch.tanh(rgb)


class StyleGAN2_Generator(nn.Module):
    '''Wraps mapping + synthesis, and keeps a running average `w_avg` of
    mapped latents purely for the TRUNCATION TRICK at inference time
    (Section 5.6): interpolating w toward its mean trades sample diversity
    for sample quality/typicality, with a single scalar knob.'''
    def __init__(self):
        super().__init__()
        self.mapping = StyleGAN2_MappingNetwork()
        self.synthesis = StyleGAN2_Synthesis()
        self.register_buffer("w_avg", torch.zeros(SGAN2_W_DIM))

    def forward(self, z, truncation: float = 1.0, mixing_w2: torch.Tensor = None, mixing_layer: int = None):
        w = self.mapping(z)
        with torch.no_grad():
            self.w_avg.mul_(0.995).add_(w.mean(0).detach(), alpha=0.005)   # cheap running-mean EMA of w
        if truncation < 1.0:
            w = self.w_avg.unsqueeze(0) + truncation * (w - self.w_avg.unsqueeze(0))
        if mixing_w2 is not None:
            ws = [w] * mixing_layer + [mixing_w2] * (self.synthesis.num_layers - mixing_layer)
            return self.synthesis(ws)
        return self.synthesis(w)


class StyleGAN2_DiscBlock(nn.Module):
    '''Residual downsampling block (StyleGAN2's discriminator is a ResNet).
    The 1x1 skip path (scaled by 1/sqrt(2) to preserve variance) lets
    gradients skip the two 3x3 convs, which the paper found meaningfully
    stabilizes training of deeper discriminators.'''
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.skip = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = F.avg_pool2d(y, 2)
        skip = F.avg_pool2d(self.skip(x), 2)
        return (y + skip) / math.sqrt(2)


class StyleGAN2_Discriminator(nn.Module):
    def __init__(self, resolutions=SGAN2_RESOLUTIONS, channels=SGAN2_CHANNELS):
        super().__init__()
        rev = list(reversed(resolutions))
        self.from_rgb = nn.Conv2d(3, channels[rev[0]], 1)
        blocks = []
        for i in range(len(rev) - 1):
            blocks.append(StyleGAN2_DiscBlock(channels[rev[i]], channels[rev[i + 1]]))
        self.blocks = nn.Sequential(*blocks)
        final_c = channels[rev[-1]]
        # Minibatch standard deviation: append one extra feature map holding
        # the std-dev of features ACROSS the batch -- gives D a cheap signal
        # for "does this whole batch look too similar" (a direct anti-mode-
        # collapse mechanism from the ProGAN/StyleGAN family).
        self.final_conv = nn.Conv2d(final_c + 1, final_c, 3, 1, 1)
        self.final_fc = nn.Sequential(nn.Flatten(), nn.Linear(final_c * rev[-1] * rev[-1], final_c), nn.LeakyReLU(0.2, True), nn.Linear(final_c, 1))

    def minibatch_stddev(self, x):
        std = x.std(dim=0, unbiased=False).mean().expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)

    def forward(self, x):
        x = F.leaky_relu(self.from_rgb(x), 0.2)
        x = self.blocks(x)
        x = self.minibatch_stddev(x)
        x = F.leaky_relu(self.final_conv(x), 0.2)
        return self.final_fc(x).view(-1)


G_sgan2 = StyleGAN2_Generator().to(DEVICE).apply(weights_init)
D_sgan2 = StyleGAN2_Discriminator().to(DEVICE).apply(weights_init)
print(f"StyleGAN2-lite: {sum(p.numel() for p in G_sgan2.parameters())/1e6:.1f}M generator params, "
      f"{sum(p.numel() for p in D_sgan2.parameters())/1e6:.1f}M discriminator params")
""")

md(r"### 5.4 Loss Functions & Optimizers")

code(r"""
sgan2_criterion = nn.BCEWithLogitsLoss()
sgan2_opt_G = torch.optim.Adam(G_sgan2.parameters(), lr=2e-4, betas=(0.0, 0.99))
sgan2_opt_D = torch.optim.Adam(D_sgan2.parameters(), lr=2e-4, betas=(0.0, 0.99))

SGAN2_R1_GAMMA = 5.0
SGAN2_R1_EVERY = 16   # "lazy" regularization: StyleGAN2 only applies R1 every N steps to save compute, scaling its weight by N to compensate

def sgan2_r1_penalty(D, real):
    '''R1 gradient penalty (Mescheder et al. / StyleGAN2): penalize the
    NORM of D's gradient w.r.t. REAL images only (unlike WGAN-GP's penalty
    on an interpolated point). This directly discourages D from having a
    sharp decision boundary right at the real-data manifold, which is
    empirically what actually causes non-convergent oscillation in
    practice -- and it is cheaper than WGAN-GP since it needs no fake pass.'''
    real = real.requires_grad_(True)
    logits = D(real)
    grad = torch.autograd.grad(outputs=logits.sum(), inputs=real, create_graph=True)[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()
""")

md(r"### 5.5 Training Loop")

code(r"""
# ---------------------------------------------------------------------------
# 5.5a Lightweight from-scratch training
# ---------------------------------------------------------------------------
sgan2_fixed_z = torch.randn(16, SGAN2_Z_DIM, device=DEVICE)
sgan2_step = 0
sgan2_history = []

for epoch in range(SGAN2_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter()}
    pbar = tqdm(sgan2_loader, desc=f"[StyleGAN2] epoch {epoch+1}/{SGAN2_EPOCHS}")
    for real, *_ in pbar:
        real = real.to(DEVICE)
        b = real.size(0)
        sgan2_step += 1

        # ---- Discriminator ----
        D_sgan2.zero_grad(set_to_none=True)
        z = torch.randn(b, SGAN2_Z_DIM, device=DEVICE)
        fake = G_sgan2(z)
        logits_real = D_sgan2(real)
        logits_fake = D_sgan2(fake.detach())
        loss_d = sgan2_criterion(logits_real, torch.ones(b, device=DEVICE)) + \
                 sgan2_criterion(logits_fake, torch.zeros(b, device=DEVICE))
        if sgan2_step % SGAN2_R1_EVERY == 0:
            r1 = sgan2_r1_penalty(D_sgan2, real)
            loss_d = loss_d + SGAN2_R1_GAMMA / 2 * r1 * SGAN2_R1_EVERY
        loss_d.backward()
        sgan2_opt_D.step()

        # ---- Generator ----
        G_sgan2.zero_grad(set_to_none=True)
        logits_fake_g = D_sgan2(fake)
        loss_g = sgan2_criterion(logits_fake_g, torch.ones(b, device=DEVICE))
        loss_g.backward()
        sgan2_opt_G.step()

        meters["D"].update(loss_d.item()); meters["G"].update(loss_g.item())
        pbar.set_postfix(loss_D=f"{loss_d.item():.3f}", loss_G=f"{loss_g.item():.3f}")

    sgan2_history.append((meters["G"].avg, meters["D"].avg))
    with torch.no_grad():
        G_sgan2.eval()
        sample = G_sgan2(sgan2_fixed_z, truncation=0.7)
        G_sgan2.train()
    show_images(sample, title=f"StyleGAN2-lite samples after epoch {epoch+1}", nrow=4)
    save_checkpoint({"G": G_sgan2.state_dict(), "D": D_sgan2.state_dict()}, f"{CHECKPOINT_DIR}/sgan2_epoch{epoch+1}.pt")

print("StyleGAN2-lite training complete.")
""")

code(r"""
# ---------------------------------------------------------------------------
# 5.5b Loading strong pretrained weights (inference-only mode)
# ---------------------------------------------------------------------------
# Official StyleGAN2/StyleGAN2-ADA checkpoints are distributed as .pkl files
# whose loading code (legacy.py / dnnlib) lives in NVIDIA's own repo, which
# also ships custom CUDA ops (upfirdn2d, fused bias-act) for full speed.
# We attempt to clone the repo and load an official FFHQ checkpoint;
# without network access this cleanly falls back to the from-scratch model.
def try_load_pretrained_stylegan2(pkl_url="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"):
    try:
        import subprocess, sys
        if not os.path.isdir("stylegan2-ada-pytorch"):
            subprocess.run(["git", "clone", "--depth", "1",
                             "https://github.com/NVlabs/stylegan2-ada-pytorch.git"], check=True)
        sys.path.insert(0, "stylegan2-ada-pytorch")
        import legacy, dnnlib  # provided by the cloned repo
        with dnnlib.util.open_url(pkl_url) as f:
            G_ema = legacy.load_network_pkl(f)["G_ema"].to(DEVICE).eval()
        print("Loaded official pretrained StyleGAN2-ADA generator (FFHQ).")
        return G_ema
    except Exception as e:
        print(f"Pretrained StyleGAN2 unavailable in this environment ({e}). "
              f"Falling back to the from-scratch StyleGAN2-lite model trained above.")
        return None

sgan2_pretrained = try_load_pretrained_stylegan2()
""")

md(r"### 5.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def sgan2_generate(n_samples: int = 16, truncation: float = 0.7, seed: int = None, use_pretrained: bool = False) -> torch.Tensor:
    '''Truncation trick: truncation=1.0 -> full diversity (may include odd
    samples); truncation<1.0 pulls w toward the running average w_avg,
    trading diversity for consistently "typical", higher-quality faces.'''
    if seed is not None:
        torch.manual_seed(seed)
    if use_pretrained and sgan2_pretrained is not None:
        z = torch.randn(n_samples, sgan2_pretrained.z_dim, device=DEVICE)
        return sgan2_pretrained(z, None, truncation_psi=truncation)
    G_sgan2.eval()
    z = torch.randn(n_samples, SGAN2_Z_DIM, device=DEVICE)
    imgs = G_sgan2(z, truncation=truncation)
    G_sgan2.train()
    return imgs

show_images(sgan2_generate(16, truncation=0.7), title="StyleGAN2-lite inference (truncation=0.7)", nrow=4)
""")

md(r"### 5.7 Visualization & Results")

code(r"""
gs, ds = zip(*sgan2_history)
plt.figure(figsize=(6, 4)); plt.plot(gs, label="G"); plt.plot(ds, label="D")
plt.legend(); plt.title("StyleGAN2-lite training losses"); plt.show()

# Truncation sweep: same z's, increasing truncation -- shows the
# diversity/quality trade-off directly.
with torch.no_grad():
    G_sgan2.eval()
    z_fixed = torch.randn(8, SGAN2_Z_DIM, device=DEVICE)
    for psi in [0.2, 0.5, 1.0]:
        imgs = G_sgan2(z_fixed, truncation=psi)
        show_images(imgs, title=f"Truncation psi={psi}", nrow=8, figsize=(12, 2))

    # Style mixing: coarse layers from w1 (pose/shape), fine layers from w2
    # (color/texture) -- the signature StyleGAN capability that vanilla
    # GAN latent space does not offer, because early layers control
    # low-resolution structure and late layers control high-resolution
    # detail almost independently in w-space.
    z1, z2 = torch.randn(4, SGAN2_Z_DIM, device=DEVICE), torch.randn(4, SGAN2_Z_DIM, device=DEVICE)
    w2 = G_sgan2.mapping(z2)
    mixed = G_sgan2(z1, mixing_w2=w2, mixing_layer=2)   # layers 0-1 from z1, layers 2+ from z2
    show_images(mixed, title="Style mixing: coarse=z1, fine=z2", nrow=4, figsize=(8, 3))
    G_sgan2.train()

# Latent interpolation in W-space (smoother than Z-space -- see 5.1).
@torch.no_grad()
def sgan2_interpolate_w(n_steps=8):
    G_sgan2.eval()
    z1, z2 = torch.randn(1, SGAN2_Z_DIM, device=DEVICE), torch.randn(1, SGAN2_Z_DIM, device=DEVICE)
    w1, w2 = G_sgan2.mapping(z1), G_sgan2.mapping(z2)
    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1)
    w_interp = (1 - alphas) * w1 + alphas * w2
    imgs = G_sgan2.synthesis(w_interp)
    G_sgan2.train()
    return imgs

show_images(sgan2_interpolate_w(8), title="W-space interpolation", nrow=8, figsize=(12, 2))
""")

md(r"### 5.8 User Input → Output Demo")

code(r"""
N_SAMPLES_TO_GENERATE = 9
TRUNCATION_PSI = 0.6   # lower = more "average"/safe faces, higher = more diverse/unusual
RANDOM_SEED = 99

demo = sgan2_generate(N_SAMPLES_TO_GENERATE, truncation=TRUNCATION_PSI, seed=RANDOM_SEED)
show_images(demo, title=f"User demo: truncation={TRUNCATION_PSI}", nrow=3)
""")

md(r"""
### 5.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Weight demodulation is a *deterministic, statistics-based*
  approximation of "normalize then modulate" (AdaIN) that never actually
  looks at the real activation statistics -- this decoupling from
  instance-specific statistics is precisely what removes StyleGAN1's
  characteristic droplet artifacts.
- 💡 **Insight.** Style mixing regularization (training with mixed w's, not
  just evaluating with them) is what makes the coarse/fine layer split
  semantically meaningful in the first place -- without it, mixing tends
  to just look broken.
- ⚠️ **Warning.** Our 64x64 lightweight model captures the *mechanism*
  (mapping network, modulated convs, noise injection, truncation, style
  mixing) but not the photorealism of the 1024x1024 FFHQ-trained original,
  which needs orders of magnitude more data/compute -- use
  `try_load_pretrained_stylegan2()` above for real photorealistic results.

**This GAN solved the problem of: photorealistic generation with
disentangled, user-controllable style -- separating "what identity/pose"
from "what fine texture/detail" in a way plain latent-space GANs cannot.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 6 -- BIGGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="6"></a>
## 6. BigGAN -- Large-scale class-conditional high-quality generation

### 6.1 Problem Statement & Why this GAN

Brock et al. (2018) asked: what happens if you take class-conditional
GANs (Section 2/3) and simply **scale everything up** -- batch size (up
to 2048), model width, and dataset diversity (all 1000 ImageNet classes) --
while fixing the specific instabilities that scale exposes? Their answer
combines **self-attention** (global, not just local receptive fields),
**spectral normalization** on every weight (Lipschitz-constrained layers
for stability), a **class-conditional projection discriminator**, and a
**hierarchical latent** (different chunks of `z` fed to different
resolutions). This is the **large-scale, high-fidelity class-conditional
generation** problem -- more classes, more diversity, sharper detail than
cGAN/AC-GAN can achieve.

We implement every one of these architectural ideas at CIFAR-10 scale
(all 10 classes, 32x32) -- full BigGAN needs ImageNet at 128-512px and
enormous batch sizes to show its real advantage, which is why we also
provide official pretrained-weight loading below.

### 6.2 Data Collection & Preparation
""")

code(r"""
BIGGAN_IMG_SIZE = 32
BIGGAN_BATCH_SIZE = 64
BIGGAN_NZ = 120                 # divisible by BIGGAN_N_CHUNKS below (hierarchical latent)
BIGGAN_N_CLASSES = 10
BIGGAN_EPOCHS = 8
BIGGAN_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

biggan_transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

try:
    biggan_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=biggan_transform)
    print(f"Loaded real CIFAR-10 (all classes): {len(biggan_dataset)} images")
except Exception as e:
    print(f"CIFAR-10 download failed ({e}); falling back to synthetic per-class textures.")
    rng = np.random.default_rng(SEED)
    imgs, labels = [], []
    for c in range(BIGGAN_N_CLASSES):
        base_color = torch.tensor(rng.uniform(-1, 1, 3), dtype=torch.float32)   # dtype pinned: rng.uniform is float64, and float32+float64 silently promotes to float64 (breaks Conv2d's float32 weights downstream)
        for _ in range(300):
            img = (torch.randn(3, BIGGAN_IMG_SIZE, BIGGAN_IMG_SIZE) * 0.1 + base_color.view(3, 1, 1)).clamp(-1, 1)
            imgs.append(img); labels.append(c)
    biggan_dataset = TensorDataset(torch.stack(imgs), torch.tensor(labels))

biggan_loader = DataLoader(biggan_dataset, batch_size=BIGGAN_BATCH_SIZE, shuffle=True, drop_last=True)
real_batch, real_labels = next(iter(biggan_loader))
show_images(real_batch[:32], title="BigGAN training data (10 classes)", nrow=8)
""")

md(r"### 6.3 Model Architecture")

code(r"""
def sn_conv2d(*args, **kwargs):
    '''Spectral normalization rescales a layer's weight matrix by (an
    estimate of) its largest singular value every forward pass, so the
    layer is provably 1-Lipschitz. BigGAN applies this to EVERY weight in
    BOTH networks (not just D, as in earlier spectral-norm GANs) -- the
    paper found this is what lets you push batch size/width up without
    training collapsing.'''
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


def sn_linear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))


class SelfAttention(nn.Module):
    '''Non-local self-attention (SAGAN, used inside BigGAN). Convolutions
    only ever see a local receptive field; attention lets every spatial
    location attend to every other one, which matters for images with
    long-range structure (e.g. a dog's two ears must match, even though
    they may be far apart at low resolution).'''
    def __init__(self, channels):
        super().__init__()
        self.query = sn_conv2d(channels, max(1, channels // 8), 1)
        self.key = sn_conv2d(channels, max(1, channels // 8), 1)
        self.value = sn_conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))   # starts at 0 -> block starts as identity, attention "phased in" as gamma learns to grow

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, -1, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


class ConditionalBatchNorm2d(nn.Module):
    '''BatchNorm whose per-channel gain/bias are PREDICTED from the
    conditioning vector (class embedding concatenated with a z-chunk) via
    small linear layers, instead of being plain learned parameters. This
    is how the class label (and, uniquely to BigGAN, part of the noise
    vector too) is injected at every single resolution of the generator.'''
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gain = sn_linear(cond_dim, num_features)
        self.bias = sn_linear(cond_dim, num_features)

    def forward(self, x, cond):
        out = self.bn(x)
        gain = (1 + self.gain(cond)).view(-1, self.num_features, 1, 1)
        bias = self.bias(cond).view(-1, self.num_features, 1, 1)
        return out * gain + bias


class GResBlockUp(nn.Module):
    '''Residual upsampling block: BN-ReLU-Upsample-Conv twice, with a
    matching upsample+1x1-conv skip path (pre-activation ResNet style).'''
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        self.bn1 = ConditionalBatchNorm2d(in_c, cond_dim)
        self.conv1 = sn_conv2d(in_c, out_c, 3, 1, 1)
        self.bn2 = ConditionalBatchNorm2d(out_c, cond_dim)
        self.conv2 = sn_conv2d(out_c, out_c, 3, 1, 1)
        self.skip_conv = sn_conv2d(in_c, out_c, 1, bias=False)
        self.act = nn.ReLU(True)

    def forward(self, x, cond):
        h = self.act(self.bn1(x, cond))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.act(self.bn2(h, cond))
        h = self.conv2(h)
        skip = self.skip_conv(F.interpolate(x, scale_factor=2, mode="nearest"))
        return h + skip


class BigGAN_Generator(nn.Module):
    '''Hierarchical latent: z is split into N_CHUNKS pieces, one consumed by
    the initial projection and one by EACH residual block's conditional
    BatchNorm (concatenated with the shared class embedding). This gives
    the generator direct, per-resolution control over the noise -- BigGAN's
    ablations show this measurably improves sample quality over feeding
    the same z everywhere.'''
    N_CHUNKS = 4

    def __init__(self, nz=BIGGAN_NZ, n_classes=BIGGAN_N_CLASSES, embed_dim=128, base_ch=96, bottom=4):
        super().__init__()
        assert nz % self.N_CHUNKS == 0
        self.chunk_size = nz // self.N_CHUNKS
        self.base_ch, self.bottom = base_ch, bottom
        self.class_embed = nn.Embedding(n_classes, embed_dim)
        self.fc = sn_linear(self.chunk_size, base_ch * 8 * bottom * bottom)
        cond_dim = embed_dim + self.chunk_size
        self.block1 = GResBlockUp(base_ch * 8, base_ch * 4, cond_dim)   # 4 -> 8
        self.block2 = GResBlockUp(base_ch * 4, base_ch * 2, cond_dim)   # 8 -> 16
        self.attn = SelfAttention(base_ch * 2)
        self.block3 = GResBlockUp(base_ch * 2, base_ch, cond_dim)       # 16 -> 32
        self.final_bn = nn.BatchNorm2d(base_ch)
        self.final_conv = sn_conv2d(base_ch, 3, 3, 1, 1)

    def forward(self, z, y):
        chunks = z.split(self.chunk_size, dim=1)
        y_embed = self.class_embed(y)
        x = self.fc(chunks[0]).view(-1, self.base_ch * 8, self.bottom, self.bottom)
        x = self.block1(x, torch.cat([chunks[1], y_embed], dim=1))
        x = self.block2(x, torch.cat([chunks[2], y_embed], dim=1))
        x = self.attn(x)
        x = self.block3(x, torch.cat([chunks[3], y_embed], dim=1))
        x = F.relu(self.final_bn(x))
        return torch.tanh(self.final_conv(x))


class DResBlockDown(nn.Module):
    '''`activate_first=False` matches BigGAN/SNGAN's "OptimizedBlock": the
    very FIRST discriminator block skips its leading activation, since
    there are no learned features yet to activate -- applying ReLU
    directly to the raw image would needlessly clip away every negative
    pixel value (our data is normalized to [-1, 1], so that's half of it).

    Note the activation is explicitly `inplace=False`: an in-place ReLU
    applied to this block's input `x` would, when `x` IS the generator's
    Tanh output (as happens when the generator is updated), silently
    corrupt the exact tensor autograd needs to compute Tanh's local
    gradient during `loss_g.backward()` -- PyTorch detects this and raises
    a "modified by an inplace operation" RuntimeError.'''
    def __init__(self, in_c, out_c, downsample=True, activate_first=True):
        super().__init__()
        self.conv1 = sn_conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = sn_conv2d(out_c, out_c, 3, 1, 1)
        self.skip_conv = sn_conv2d(in_c, out_c, 1, bias=False)
        self.downsample = downsample
        self.activate_first = activate_first
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.act(x) if self.activate_first else x
        h = self.conv1(h)
        h = self.conv2(self.act(h))
        skip = self.skip_conv(x)
        if self.downsample:
            h, skip = F.avg_pool2d(h, 2), F.avg_pool2d(skip, 2)
        return h + skip


class BigGAN_Discriminator(nn.Module):
    '''Projection discriminator (Miyato & Koyama, 2018): instead of
    concatenating the label as an extra channel (cGAN-style), the class
    embedding is combined with the image FEATURES via an inner product
    added to the unconditional real/fake score. This is a much stronger,
    more scalable conditioning signal for discriminators with many classes.'''
    def __init__(self, n_classes=BIGGAN_N_CLASSES, base_ch=96):
        super().__init__()
        self.block1 = DResBlockDown(3, base_ch, activate_first=False)   # 32 -> 16 (no leading activation -- see DResBlockDown docstring)
        self.attn = SelfAttention(base_ch)
        self.block2 = DResBlockDown(base_ch, base_ch * 2)     # 16 -> 8
        self.block3 = DResBlockDown(base_ch * 2, base_ch * 4)  # 8 -> 4
        self.block4 = DResBlockDown(base_ch * 4, base_ch * 8, downsample=False)  # 4 -> 4
        self.act = nn.ReLU(inplace=False)
        self.fc = sn_linear(base_ch * 8, 1)
        self.class_embed = nn.Embedding(n_classes, base_ch * 8)

    def forward(self, x, y):
        h = self.block1(x)
        h = self.attn(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.act(h).sum(dim=[2, 3])          # global sum pooling (BigGAN uses sum, not mean, pooling)
        out = self.fc(h).view(-1)
        proj = (self.class_embed(y) * h).sum(dim=1)   # projection term
        return out + proj


G_biggan = BigGAN_Generator().to(DEVICE)
D_biggan = BigGAN_Discriminator().to(DEVICE)
print(f"BigGAN-lite: {sum(p.numel() for p in G_biggan.parameters())/1e6:.1f}M generator params, "
      f"{sum(p.numel() for p in D_biggan.parameters())/1e6:.1f}M discriminator params")
""")

md(r"### 6.4 Loss Functions & Optimizers")

code(r"""
# Hinge loss (Lim & Ye / Tran et al.), the loss BigGAN actually uses: it
# only penalizes D while its margin is violated (D(real) < 1 or D(fake) >
# -1), which pairs well with spectral normalization's Lipschitz bound and
# empirically gives cleaner gradients than BCE at this scale.
def biggan_d_hinge_loss(logits_real, logits_fake):
    return F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()

def biggan_g_hinge_loss(logits_fake):
    return -logits_fake.mean()

# BigGAN uses a LOWER learning rate + more D steps per G step than most
# GANs in this notebook, and, crucially, DIFFERENT learning rates for G/D
# (Two-Timescale Update Rule, TTUR) -- D is allowed to adapt faster.
biggan_opt_G = torch.optim.Adam(G_biggan.parameters(), lr=1e-4, betas=(0.0, 0.999))
biggan_opt_D = torch.optim.Adam(D_biggan.parameters(), lr=4e-4, betas=(0.0, 0.999))
BIGGAN_D_STEPS_PER_G_STEP = 2

biggan_ema = EMA(G_biggan, decay=0.999)  # BigGAN inference always uses the EMA generator, never raw weights
""")

md(r"### 6.5 Training Loop")

code(r"""
# ---------------------------------------------------------------------------
# 6.5a Lightweight from-scratch training
# ---------------------------------------------------------------------------
biggan_fixed_z = torch.randn(BIGGAN_N_CLASSES * 4, BIGGAN_NZ, device=DEVICE)
biggan_fixed_y = torch.arange(BIGGAN_N_CLASSES, device=DEVICE).repeat_interleave(4)
biggan_history = []

for epoch in range(BIGGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter()}
    pbar = tqdm(biggan_loader, desc=f"[BigGAN] epoch {epoch+1}/{BIGGAN_EPOCHS}")
    d_step_counter = 0
    for real, y in pbar:
        real, y = real.to(DEVICE), y.to(DEVICE)
        b = real.size(0)

        # ---- Discriminator (D_STEPS_PER_G_STEP updates per G update) ----
        D_biggan.zero_grad(set_to_none=True)
        z = torch.randn(b, BIGGAN_NZ, device=DEVICE)
        fake_y = torch.randint(0, BIGGAN_N_CLASSES, (b,), device=DEVICE)
        with torch.no_grad():
            fake = G_biggan(z, fake_y)
        logits_real = D_biggan(real, y)
        logits_fake = D_biggan(fake, fake_y)
        loss_d = biggan_d_hinge_loss(logits_real, logits_fake)
        loss_d.backward()
        biggan_opt_D.step()
        d_step_counter += 1

        if d_step_counter % BIGGAN_D_STEPS_PER_G_STEP == 0:
            # ---- Generator ----
            G_biggan.zero_grad(set_to_none=True)
            z = torch.randn(b, BIGGAN_NZ, device=DEVICE)
            fake_y = torch.randint(0, BIGGAN_N_CLASSES, (b,), device=DEVICE)
            fake = G_biggan(z, fake_y)
            logits_fake_g = D_biggan(fake, fake_y)
            loss_g = biggan_g_hinge_loss(logits_fake_g)
            loss_g.backward()
            biggan_opt_G.step()
            biggan_ema.update(G_biggan)
            meters["G"].update(loss_g.item())

        meters["D"].update(loss_d.item())
        pbar.set_postfix(loss_D=f"{loss_d.item():.3f}", loss_G=f"{meters['G'].avg:.3f}")

    biggan_history.append((meters["G"].avg, meters["D"].avg))
    with torch.no_grad():
        eval_G = BigGAN_Generator().to(DEVICE)
        biggan_ema.copy_to(eval_G)
        eval_G.eval()
        sample = eval_G(biggan_fixed_z, biggan_fixed_y)
    show_images(sample, title=f"BigGAN-lite (EMA) samples after epoch {epoch+1}", nrow=BIGGAN_N_CLASSES)
    save_checkpoint({"G": G_biggan.state_dict(), "D": D_biggan.state_dict(), "ema": biggan_ema.shadow},
                     f"{CHECKPOINT_DIR}/biggan_epoch{epoch+1}.pt")

print("BigGAN-lite training complete.")
""")

code(r"""
# ---------------------------------------------------------------------------
# 6.5b Loading strong pretrained weights (inference-only mode)
# ---------------------------------------------------------------------------
# `pytorch_pretrained_biggan` (Hugging Face) wraps official BigGAN-deep
# checkpoints trained on all 1000 ImageNet classes at 128/256/512px --
# genuinely large-scale, unlike our CIFAR-10-scale model above.
def try_load_pretrained_biggan(model_name="biggan-deep-128"):
    try:
        from pytorch_pretrained_biggan import BigGAN
        model = BigGAN.from_pretrained(model_name).to(DEVICE).eval()
        print(f"Loaded official pretrained {model_name} (1000 ImageNet classes).")
        return model
    except Exception as e:
        print(f"Pretrained BigGAN unavailable ({e}). Install with "
              f"'pip install pytorch-pretrained-biggan' and ensure network access. "
              f"Falling back to the from-scratch BigGAN-lite model trained above.")
        return None

biggan_pretrained = try_load_pretrained_biggan()
""")

md(r"### 6.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def biggan_generate(class_id: int, n_samples: int = 8, truncation: float = 0.5, seed: int = None, use_pretrained: bool = False) -> torch.Tensor:
    '''BigGAN's truncation trick: sample z from a NORMAL, then resample any
    component with |z| > truncation (truncated normal) -- lower truncation
    means safer/sharper/less diverse samples, exactly analogous to
    StyleGAN2's w-space truncation but applied directly to z.'''
    if seed is not None:
        torch.manual_seed(seed)
    if use_pretrained and biggan_pretrained is not None:
        from pytorch_pretrained_biggan import truncated_noise_sample, one_hot_from_int
        noise = torch.from_numpy(truncated_noise_sample(truncation=truncation, batch_size=n_samples)).to(DEVICE)
        one_hot = torch.from_numpy(one_hot_from_int([class_id] * n_samples, batch_size=n_samples)).to(DEVICE)
        return biggan_pretrained(noise, one_hot, truncation)

    eval_G = BigGAN_Generator().to(DEVICE)
    biggan_ema.copy_to(eval_G)
    eval_G.eval()
    z = torch.randn(n_samples, BIGGAN_NZ, device=DEVICE).clamp(-truncation * 2, truncation * 2)
    y = torch.full((n_samples,), class_id, dtype=torch.long, device=DEVICE)
    return eval_G(z, y)

show_images(biggan_generate(class_id=3, n_samples=8), title=f"BigGAN inference: class='{BIGGAN_CLASS_NAMES[3]}'", nrow=8, figsize=(10, 2))
""")

md(r"### 6.7 Visualization & Results")

code(r"""
gs, ds = zip(*biggan_history)
plt.figure(figsize=(6, 4)); plt.plot(gs, label="G"); plt.plot(ds, label="D")
plt.legend(); plt.title("BigGAN-lite training losses"); plt.show()

# All 10 classes, EMA weights -- the core deliverable of large-scale
# class-conditional generation.
eval_G = BigGAN_Generator().to(DEVICE)
biggan_ema.copy_to(eval_G)
eval_G.eval()
with torch.no_grad():
    z = torch.randn(BIGGAN_N_CLASSES * 4, BIGGAN_NZ, device=DEVICE)
    y = torch.arange(BIGGAN_N_CLASSES, device=DEVICE).repeat_interleave(4)
    grid = eval_G(z, y)
show_images(grid, title="One row per class: " + ", ".join(BIGGAN_CLASS_NAMES), nrow=4, figsize=(6, 15))

# Truncation sweep for a single class.
with torch.no_grad():
    z_fixed = torch.randn(8, BIGGAN_NZ, device=DEVICE)
    y_fixed = torch.full((8,), 7, dtype=torch.long, device=DEVICE)  # "horse"
    for trunc in [0.3, 0.7, 1.2]:
        imgs = eval_G(z_fixed.clamp(-trunc * 2, trunc * 2), y_fixed)
        show_images(imgs, title=f"Truncation={trunc}, class='horse'", nrow=8, figsize=(12, 2))
""")

md(r"### 6.8 User Input → Output Demo")

code(r"""
USER_REQUESTED_CLASS = "ship"
TRUNCATION = 0.5
N_SAMPLES_TO_GENERATE = 8

class_idx = BIGGAN_CLASS_NAMES.index(USER_REQUESTED_CLASS)
demo = biggan_generate(class_idx, n_samples=N_SAMPLES_TO_GENERATE, truncation=TRUNCATION, seed=11)
show_images(demo, title=f"User requested: '{USER_REQUESTED_CLASS}', truncation={TRUNCATION}", nrow=4)
""")

md(r"""
### 6.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Almost none of BigGAN's individual components are new
  (self-attention: SAGAN; spectral norm: Miyato et al.; projection
  discriminator: Miyato & Koyama; hinge loss: geometric GAN) -- its
  contribution was rigorously combining them AND showing they keep working
  as batch size and model width scale up by 10-50x, which is a genuinely
  hard empirical/engineering result, not just an architecture.
- 🚀 **Pro Tip.** Always evaluate/deploy the **EMA** generator, never the
  raw training weights -- the gap in sample quality is large and free.
- ⚠️ **Warning.** BigGAN is notoriously prone to **late-training collapse**
  even after many stable epochs, which is why the original paper
  checkpoints frequently and picks the best checkpoint by FID rather than
  training to a fixed final step.

**This GAN solved the problem of: large-scale, high-fidelity,
class-conditional generation across many diverse classes -- pushing
conditional-GAN sample quality far beyond what cGAN/AC-GAN achieve by
combining attention, spectral normalization, and a stronger conditioning
mechanism at scale.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 7 -- STACKGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="7"></a>
## 7. StackGAN -- Text-to-image generation (two-stage)

### 7.1 Problem Statement & Why this GAN

Zhang et al. (2017) tackled **text-to-image synthesis**: given a natural
-language description, generate a matching photo -- a problem plain
class-conditional GANs cannot address at all, since a sentence is a far
richer, more variable condition than a fixed class id. StackGAN's key
ideas are (1) **two stages**: Stage-I sketches low-resolution shape/color
from text, Stage-II takes that sketch *and* the text again and refines it
to a higher-resolution, more detailed image; and (2) **Conditioning
Augmentation (CA)**: instead of feeding the raw text embedding directly,
sample a condition vector `c ~ N(mu(t), sigma(t))` from a small Gaussian
predicted from the text embedding `t`, regularized toward `N(0, I)` with a
KL term. This smooths the conditioning manifold, so nearby (but distinct)
sentences produce smoothly varying images instead of a sparse, easily
-overfit set of exact-caption reproductions.

We build genuine template-generated captions ("a photo of a airplane",
"a small dog", ...) for CIFAR-10 images and a real trainable
Embedding+LSTM text encoder, since paired caption datasets (CUB-200,
Oxford-102) are large downloads inappropriate for a guaranteed-offline
demo. **Swap `stackgan_dataset` for CUB-200-2011 (with its 10
human-written captions per image) for a real text-to-image result.**

### 7.2 Data Collection & Preparation
""")

code(r"""
STACKGAN_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
STACKGAN_TEMPLATES = ["a photo of a {}", "a small {}", "an image showing a {}", "a blurry photo of a {}"]
STAGE1_SIZE, STAGE2_SIZE = 32, 64
STACKGAN_MAX_LEN = 8
STACKGAN_NZ, STACKGAN_C_DIM = 100, 64
STACKGAN_BATCH_SIZE = 64
STACKGAN_STAGE1_EPOCHS, STACKGAN_STAGE2_EPOCHS = 6, 6

# ---- Build a tiny vocabulary from the templates themselves ----
_all_words = set()
for t in STACKGAN_TEMPLATES:
    _all_words.update(t.replace("{}", "").split())
_all_words.update(STACKGAN_CLASS_NAMES)
STACKGAN_VOCAB = {"<pad>": 0, "<unk>": 1}
for w in sorted(_all_words):
    STACKGAN_VOCAB[w] = len(STACKGAN_VOCAB)

def tokenize_caption(caption: str) -> torch.Tensor:
    ids = [STACKGAN_VOCAB.get(w, STACKGAN_VOCAB["<unk>"]) for w in caption.split()]
    ids = ids[:STACKGAN_MAX_LEN] + [0] * max(0, STACKGAN_MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)

try:
    stackgan_base = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True,
        transform=T.Compose([T.Resize(STAGE2_SIZE), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]))
    print(f"Loaded real CIFAR-10 (as image side of synthetic captions): {len(stackgan_base)} images")
except Exception as e:
    print(f"CIFAR-10 download failed ({e}); falling back to synthetic per-class textures.")
    rng = np.random.default_rng(SEED)
    imgs, labels = [], []
    for c in range(10):
        color = torch.tensor(rng.uniform(-1, 1, 3), dtype=torch.float32)   # dtype pinned -- see BigGAN section 6.2 for why
        for _ in range(200):
            img = (torch.randn(3, STAGE2_SIZE, STAGE2_SIZE) * 0.1 + color.view(3, 1, 1)).clamp(-1, 1)
            imgs.append(img); labels.append(c)
    stackgan_base = TensorDataset(torch.stack(imgs), torch.tensor(labels))

class StackGANTextDataset(Dataset):
    '''Wraps an image+label dataset with a randomly-templated CAPTION per
    __getitem__ call, tokenized to fixed length -- the "text" half of the
    text-to-image pair. Real StackGAN training data ships several
    human-written captions per image instead of a template.'''
    def __init__(self, base, class_names, rng_seed=SEED):
        self.base = base
        self.class_names = class_names
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        template = STACKGAN_TEMPLATES[self.rng.integers(len(STACKGAN_TEMPLATES))]
        caption = template.format(self.class_names[int(y)])
        return img, tokenize_caption(caption), int(y)

stackgan_dataset = StackGANTextDataset(stackgan_base, STACKGAN_CLASS_NAMES)
stackgan_loader = DataLoader(stackgan_dataset, batch_size=STACKGAN_BATCH_SIZE, shuffle=True, drop_last=True)

imgs64, tokens, ys = next(iter(stackgan_loader))
sample_captions = [" ".join(k for k, v in STACKGAN_VOCAB.items() if v in tok.tolist() and v > 1) for tok in tokens[:4]]
print("Example (image, caption) pairs:", list(zip([STACKGAN_CLASS_NAMES[y] for y in ys[:4].tolist()], sample_captions)))
show_images(imgs64[:16], title="StackGAN training images (64x64 target)", nrow=4)
""")

md(r"### 7.3 Model Architecture")

code(r"""
class StackGAN_TextEncoder(nn.Module):
    '''A small trainable Embedding + LSTM that reads the tokenized caption
    and produces a single fixed-size sentence embedding (its final hidden
    state). The real StackGAN paper uses a pretrained char-CNN-RNN text
    encoder trained jointly with a matching image encoder (Reed et al.);
    we train ours end-to-end with the GANs instead, which is simpler but
    otherwise plays the identical architectural role.'''
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, tokens):
        emb = self.embed(tokens)
        _, (h_n, _) = self.lstm(emb)
        return h_n[-1]   # (B, hidden_dim)


class ConditioningAugmentation(nn.Module):
    '''Reparameterization-trick sampling: c = mu(t) + sigma(t) * eps. The
    KL(N(mu,sigma) || N(0,I)) term returned alongside `c` is added to the
    generator loss -- it keeps the learned (mu, sigma) close to a standard
    Gaussian, which is what makes the conditioning manifold smooth and
    prevents the network from encoding each caption as an isolated point.'''
    def __init__(self, text_dim=128, c_dim=STACKGAN_C_DIM):
        super().__init__()
        self.fc = nn.Linear(text_dim, c_dim * 2)

    def forward(self, text_embed):
        mu, logvar = self.fc(text_embed).chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        c = mu + std * torch.randn_like(std)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return c, kl


class StackGAN_ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class StackGAN_Stage1_Generator(nn.Module):
    '''z (randomness) + c (text-derived condition) -> a coarse 32x32 image
    that captures rough shape and color implied by the caption.'''
    def __init__(self, nz=STACKGAN_NZ, c_dim=STACKGAN_C_DIM, ngf=128):
        super().__init__()
        self.ngf = ngf
        self.fc = nn.Sequential(nn.Linear(nz + c_dim, ngf * 4 * 4 * 4), nn.BatchNorm1d(ngf * 4 * 4 * 4), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),  # 4 -> 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU(True),          # 8 -> 16
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1), nn.BatchNorm2d(ngf // 2), nn.ReLU(True),    # 16 -> 32
            nn.Conv2d(ngf // 2, 3, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z, c):
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, self.ngf * 4, 4, 4)
        return self.net(x)


class StackGAN_Stage1_Discriminator(nn.Module):
    def __init__(self, c_dim=STACKGAN_C_DIM, ndf=64):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(ndf * 4 + c_dim, ndf * 4, 1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4),
        )

    def forward(self, img, c):
        feat = self.img_encoder(img)
        c_map = c.view(c.size(0), -1, 1, 1).expand(-1, -1, feat.size(2), feat.size(3))
        return self.joint(torch.cat([feat, c_map], dim=1)).view(-1)


class StackGAN_Stage2_Generator(nn.Module):
    '''Takes the STAGE-I image (not noise!) plus the text condition c again,
    encodes it down, fuses in c a second time, refines through residual
    blocks, then upsamples to 64x64 -- adding detail/resolution while
    staying faithful to Stage-I's coarse layout and the original caption.'''
    def __init__(self, c_dim=STACKGAN_C_DIM, ngf=128, n_res_blocks=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),   # 32 -> 16
        )
        self.fuse = nn.Sequential(nn.Conv2d(ngf * 2 + c_dim, ngf * 2, 3, 1, 1), nn.BatchNorm2d(ngf * 2), nn.ReLU(True))
        self.res_blocks = nn.Sequential(*[StackGAN_ResBlock(ngf * 2) for _ in range(n_res_blocks)])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU(True),        # 16 -> 32
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1), nn.BatchNorm2d(ngf // 2), nn.ReLU(True),  # 32 -> 64
            nn.Conv2d(ngf // 2, 3, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, stage1_img, c):
        feat = self.down(stage1_img)
        c_map = c.view(c.size(0), -1, 1, 1).expand(-1, -1, feat.size(2), feat.size(3))
        feat = self.fuse(torch.cat([feat, c_map], dim=1))
        feat = self.res_blocks(feat)
        return self.up(feat)


class StackGAN_Stage2_Discriminator(nn.Module):
    def __init__(self, c_dim=STACKGAN_C_DIM, ndf=64):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(ndf * 8 + c_dim, ndf * 8, 1), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4),
        )

    def forward(self, img, c):
        feat = self.img_encoder(img)
        c_map = c.view(c.size(0), -1, 1, 1).expand(-1, -1, feat.size(2), feat.size(3))
        return self.joint(torch.cat([feat, c_map], dim=1)).view(-1)


stackgan_text_encoder = StackGAN_TextEncoder(len(STACKGAN_VOCAB)).to(DEVICE)
stackgan_ca = ConditioningAugmentation().to(DEVICE)
G1_stackgan = StackGAN_Stage1_Generator().to(DEVICE).apply(weights_init)
D1_stackgan = StackGAN_Stage1_Discriminator().to(DEVICE).apply(weights_init)
G2_stackgan = StackGAN_Stage2_Generator().to(DEVICE).apply(weights_init)
D2_stackgan = StackGAN_Stage2_Discriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 7.4 Loss Functions & Optimizers")

code(r"""
stackgan_adv_criterion = nn.BCEWithLogitsLoss()
STACKGAN_LAMBDA_KL = 2.0   # weight of the CA KL term relative to the adversarial loss

# Stage-I optimizes G1+text_encoder+CA jointly against D1.
stackgan_opt_G1 = torch.optim.Adam(
    list(G1_stackgan.parameters()) + list(stackgan_text_encoder.parameters()) + list(stackgan_ca.parameters()),
    lr=2e-4, betas=(0.5, 0.999))
stackgan_opt_D1 = torch.optim.Adam(D1_stackgan.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Stage-II optimizes only G2 (Stage-I is frozen; text pipeline reused frozen too -- see 7.5).
stackgan_opt_G2 = torch.optim.Adam(G2_stackgan.parameters(), lr=2e-4, betas=(0.5, 0.999))
stackgan_opt_D2 = torch.optim.Adam(D2_stackgan.parameters(), lr=2e-4, betas=(0.5, 0.999))
""")

md(r"### 7.5 Training Loop")

code(r"""
# ---------------------------------------------------------------------------
# 7.5a Stage-I: text -> coarse 32x32 image
# ---------------------------------------------------------------------------
stackgan_stage1_history = []
for epoch in range(STACKGAN_STAGE1_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter(), "KL": AvgMeter()}
    pbar = tqdm(stackgan_loader, desc=f"[StackGAN Stage-I] epoch {epoch+1}/{STACKGAN_STAGE1_EPOCHS}")
    for imgs64, tokens, _ in pbar:
        real64, tokens = imgs64.to(DEVICE), tokens.to(DEVICE)
        real32 = F.interpolate(real64, size=STAGE1_SIZE, mode="bilinear", align_corners=False)
        b = real32.size(0)

        text_embed = stackgan_text_encoder(tokens)
        c, kl = stackgan_ca(text_embed)
        z = torch.randn(b, STACKGAN_NZ, device=DEVICE)
        fake32 = G1_stackgan(z, c)

        # ---- D1 ----
        D1_stackgan.zero_grad(set_to_none=True)
        loss_d1 = stackgan_adv_criterion(D1_stackgan(real32, c.detach()), torch.ones(b, device=DEVICE)) + \
                  stackgan_adv_criterion(D1_stackgan(fake32.detach(), c.detach()), torch.zeros(b, device=DEVICE))
        loss_d1.backward()
        stackgan_opt_D1.step()

        # ---- G1 + text encoder + CA ----
        stackgan_opt_G1.zero_grad(set_to_none=True)
        loss_g1_adv = stackgan_adv_criterion(D1_stackgan(fake32, c), torch.ones(b, device=DEVICE))
        loss_g1 = loss_g1_adv + STACKGAN_LAMBDA_KL * kl
        loss_g1.backward()
        stackgan_opt_G1.step()

        meters["D"].update(loss_d1.item()); meters["G"].update(loss_g1.item()); meters["KL"].update(kl.item())
        pbar.set_postfix(loss_D=f"{loss_d1.item():.3f}", loss_G=f"{loss_g1.item():.3f}", KL=f"{kl.item():.3f}")

    stackgan_stage1_history.append((meters["G"].avg, meters["D"].avg))
    with torch.no_grad():
        G1_stackgan.eval()
        te = stackgan_text_encoder(tokens[:16]); c_demo, _ = stackgan_ca(te)
        sample = G1_stackgan(torch.randn(16, STACKGAN_NZ, device=DEVICE), c_demo)
        G1_stackgan.train()
    show_images(sample, title=f"StackGAN Stage-I samples after epoch {epoch+1}", nrow=4)

save_checkpoint({"G1": G1_stackgan.state_dict(), "D1": D1_stackgan.state_dict(),
                  "text_encoder": stackgan_text_encoder.state_dict(), "ca": stackgan_ca.state_dict()},
                 f"{CHECKPOINT_DIR}/stackgan_stage1.pt")
print("StackGAN Stage-I complete.")
""")

code(r"""
# ---------------------------------------------------------------------------
# 7.5b Stage-II: (Stage-I image + text) -> refined 64x64 image
# ---------------------------------------------------------------------------
# Stage-I generator and the text pipeline are FROZEN here (requires_grad_(False))
# -- Stage-II's only job is to add detail/resolution conditioned on what
# Stage-I already committed to, exactly as in the original paper.
G1_stackgan.eval()
for p in G1_stackgan.parameters():
    p.requires_grad_(False)
for p in stackgan_text_encoder.parameters():
    p.requires_grad_(False)
for p in stackgan_ca.parameters():
    p.requires_grad_(False)

stackgan_stage2_history = []
for epoch in range(STACKGAN_STAGE2_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter()}
    pbar = tqdm(stackgan_loader, desc=f"[StackGAN Stage-II] epoch {epoch+1}/{STACKGAN_STAGE2_EPOCHS}")
    for imgs64, tokens, _ in pbar:
        real64, tokens = imgs64.to(DEVICE), tokens.to(DEVICE)
        b = real64.size(0)

        with torch.no_grad():
            text_embed = stackgan_text_encoder(tokens)
            c, _ = stackgan_ca(text_embed)
            z = torch.randn(b, STACKGAN_NZ, device=DEVICE)
            stage1_img = G1_stackgan(z, c)
        fake64 = G2_stackgan(stage1_img, c)

        # ---- D2 ----
        D2_stackgan.zero_grad(set_to_none=True)
        loss_d2 = stackgan_adv_criterion(D2_stackgan(real64, c), torch.ones(b, device=DEVICE)) + \
                  stackgan_adv_criterion(D2_stackgan(fake64.detach(), c), torch.zeros(b, device=DEVICE))
        loss_d2.backward()
        stackgan_opt_D2.step()

        # ---- G2 ----
        stackgan_opt_G2.zero_grad(set_to_none=True)
        loss_g2 = stackgan_adv_criterion(D2_stackgan(fake64, c), torch.ones(b, device=DEVICE))
        loss_g2.backward()
        stackgan_opt_G2.step()

        meters["D"].update(loss_d2.item()); meters["G"].update(loss_g2.item())
        pbar.set_postfix(loss_D=f"{loss_d2.item():.3f}", loss_G=f"{loss_g2.item():.3f}")

    stackgan_stage2_history.append((meters["G"].avg, meters["D"].avg))
    with torch.no_grad():
        G2_stackgan.eval()
        sample = G2_stackgan(stage1_img[:16], c[:16])
        G2_stackgan.train()
    show_images(sample, title=f"StackGAN Stage-II samples after epoch {epoch+1}", nrow=4)

save_checkpoint({"G2": G2_stackgan.state_dict(), "D2": D2_stackgan.state_dict()}, f"{CHECKPOINT_DIR}/stackgan_stage2.pt")
print("StackGAN Stage-II complete.")
""")

md(r"### 7.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def stackgan_generate(caption: str, n_samples: int = 4, seed: int = None):
    '''End-to-end text -> (stage1_image, stage2_image). This is the function
    a real product would call: user types a sentence, gets an image back.'''
    if seed is not None:
        torch.manual_seed(seed)
    tokens = tokenize_caption(caption).unsqueeze(0).repeat(n_samples, 1).to(DEVICE)
    G1_stackgan.eval(); G2_stackgan.eval()
    text_embed = stackgan_text_encoder(tokens)
    c, _ = stackgan_ca(text_embed)
    z = torch.randn(n_samples, STACKGAN_NZ, device=DEVICE)
    stage1_img = G1_stackgan(z, c)
    stage2_img = G2_stackgan(stage1_img, c)
    G1_stackgan.train(); G2_stackgan.train()
    return stage1_img, stage2_img

s1, s2 = stackgan_generate("a photo of a horse", n_samples=8)
show_images(F.interpolate(s1, size=STAGE2_SIZE), title="Stage-I (upsampled for display)", nrow=8, figsize=(12, 2))
show_images(s2, title="Stage-II (final)", nrow=8, figsize=(12, 2))
""")

md(r"### 7.7 Visualization & Results")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
g1s, d1s = zip(*stackgan_stage1_history); g2s, d2s = zip(*stackgan_stage2_history)
axes[0].plot(g1s, label="G1"); axes[0].plot(d1s, label="D1"); axes[0].legend(); axes[0].set_title("Stage-I losses")
axes[1].plot(g2s, label="G2"); axes[1].plot(d2s, label="D2"); axes[1].legend(); axes[1].set_title("Stage-II losses")
plt.show()

# Side-by-side: for several captions, show Stage-I vs Stage-II output --
# directly visualizes what the refinement stage adds.
fig, axes = plt.subplots(3, 2, figsize=(5, 8))
for row, cap in enumerate(["a photo of a bird", "a small dog", "an image showing a ship"]):
    s1, s2 = stackgan_generate(cap, n_samples=1, seed=row)
    axes[row, 0].imshow(denorm(F.interpolate(s1, size=STAGE2_SIZE))[0].permute(1, 2, 0).cpu()); axes[row, 0].set_title(f"Stage-I: '{cap}'", fontsize=8); axes[row, 0].axis("off")
    axes[row, 1].imshow(denorm(s2)[0].permute(1, 2, 0).cpu()); axes[row, 1].set_title("Stage-II", fontsize=8); axes[row, 1].axis("off")
plt.tight_layout(); plt.show()

# Conditioning-augmentation smoothness check: same caption, resampled c
# each time -- outputs should vary smoothly, not jump between unrelated
# images, if the CA/KL regularization is doing its job.
s1_multi, s2_multi = stackgan_generate("a photo of a cat", n_samples=8)
show_images(s2_multi, title="Same caption, 8 resampled conditioning vectors c", nrow=8, figsize=(12, 2))
""")

md(r"### 7.8 User Input → Output Demo")

code(r"""
USER_CAPTION = "a photo of a truck"
N_SAMPLES_TO_GENERATE = 4

_, s2_demo = stackgan_generate(USER_CAPTION, n_samples=N_SAMPLES_TO_GENERATE, seed=42)
show_images(s2_demo, title=f"User caption: '{USER_CAPTION}'", nrow=4)
""")

md(r"""
### 7.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Conditioning Augmentation is a general trick, not
  StackGAN-specific -- any GAN conditioned on a continuous, high
  -dimensional, sparsely-sampled embedding (text, audio, even
  fine-grained attribute vectors) benefits from smoothing that embedding
  with a small VAE-style reparameterized sample before it reaches the
  generator.
- ⚠️ **Warning.** With only template-generated captions ("a photo of a
  {class}"), the model can only ever really recover *class identity* from
  text -- true free-form text-to-image (arbitrary attributes, colors,
  counts, relations) requires a real captioned dataset (CUB-200,
  Oxford-102 Flowers, or COCO) with genuine linguistic variation.
- ⚠️ **Warning.** Because Stage-II is trained conditioned on Stage-I's
  *own* outputs (not ground truth), any systematic bias Stage-I develops
  (e.g. always centering the subject) propagates into Stage-II -- this
  coupling is why later text-to-image work (AttnGAN, diffusion models)
  moved toward single-stage, attention-based refinement instead of a
  two-stage pipeline.

**This GAN solved the problem of: text-to-image generation -- turning a
free-form natural-language description into a matching image, first
sketched coarsely then refined, rather than picking from a fixed set of
classes.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 8 -- CYCLEGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="8"></a>
## 8. CycleGAN -- Unpaired image-to-image translation

### 8.1 Problem Statement & Why this GAN

Zhu et al. (2017) addressed image-to-image translation (horses <-> zebras,
photos <-> paintings) in the setting where you have **two unpaired
collections** of images (a bunch of horse photos, a bunch of zebra photos,
*no* horse/zebra pairs of the exact same pose) -- pix2pix-style
paired translation is inapplicable, since no ground-truth target exists
for any given input. CycleGAN's solution is **cycle-consistency**: learn
two generators `G: A->B` and `F: B->A` such that translating there and
back recovers the original image, `F(G(a)) ~ a` and `G(F(b)) ~ b`. This
constraint alone, with no paired supervision at all, is enough to force
`G`/`F` to preserve content while changing only style/domain-specific
appearance. This is the **unpaired image-to-image translation** problem.

### 8.2 Data Collection & Preparation
""")

code(r"""
CYCLEGAN_IMG_SIZE = 64
CYCLEGAN_BATCH_SIZE = 8
CYCLEGAN_EPOCHS = 6

cyclegan_transform = T.Compose([T.Resize(CYCLEGAN_IMG_SIZE), T.CenterCrop(CYCLEGAN_IMG_SIZE),
                                 T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

class UnpairedFolderDataset(Dataset):
    '''Loads all images directly under `root` (any format Pillow supports);
    CRUCIALLY, this dataset is indexed independently for domain A and
    domain B -- we never assume image i in A corresponds to image i in B.'''
    def __init__(self, root, transform):
        from PIL import Image
        self.Image = Image
        self.paths = sorted(Path(root).glob("*.jpg")) + sorted(Path(root).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)

def add_stripes(img: torch.Tensor, rng) -> torch.Tensor:
    '''Procedurally overlay a zebra-like diagonal stripe pattern -- used
    ONLY to build an offline, guaranteed-available "second domain" when the
    real horse2zebra dataset cannot be downloaded, so cycle-consistency
    still has a genuine (if synthetic) style-transfer task to learn.'''
    c, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    phase = rng.uniform(0, 2 * math.pi)
    freq = rng.uniform(0.4, 0.7)
    stripes = (torch.sin(freq * (xx + yy).float() + phase) > 0).float()
    stripe_color = torch.tensor([-1.0, -1.0, -1.0]).view(3, 1, 1)
    return img * (1 - 0.5 * stripes) + stripe_color * (0.5 * stripes)

try:
    import urllib.request, zipfile
    zip_path = "data/horse2zebra.zip"
    if not os.path.exists("data/horse2zebra"):
        os.makedirs("data", exist_ok=True)
        url = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("data")
    domain_A = UnpairedFolderDataset("data/horse2zebra/trainA", cyclegan_transform)
    domain_B = UnpairedFolderDataset("data/horse2zebra/trainB", cyclegan_transform)
    assert len(domain_A) > 0 and len(domain_B) > 0
    print(f"Loaded real horse2zebra: {len(domain_A)} horses, {len(domain_B)} zebras")
except Exception as e:
    print(f"horse2zebra download failed ({e}); falling back to synthetic "
          f"'plain photo' <-> 'striped photo' domains built from CIFAR-10.")
    try:
        cifar_src = torchvision.datasets.CIFAR10(root="data", train=True, download=True,
                                                   transform=T.Compose([T.Resize(CYCLEGAN_IMG_SIZE), T.ToTensor(),
                                                                         T.Normalize((0.5,) * 3, (0.5,) * 3)]))
        horse_idx = [i for i, (_, y) in enumerate(cifar_src) if y == 7]  # CIFAR-10 class 7 = horse
    except Exception:
        cifar_src = None
        horse_idx = []
    rng = np.random.default_rng(SEED)
    if cifar_src is not None and len(horse_idx) >= 200:
        half = len(horse_idx) // 2
        imgs_A = torch.stack([cifar_src[i][0] for i in horse_idx[:half]])
        imgs_B = torch.stack([add_stripes(cifar_src[i][0], rng) for i in horse_idx[half:]])
    else:
        imgs_A = torch.rand(200, 3, CYCLEGAN_IMG_SIZE, CYCLEGAN_IMG_SIZE) * 2 - 1
        imgs_B = torch.stack([add_stripes(img, rng) for img in imgs_A])
    domain_A = TensorDataset(imgs_A)
    domain_B = TensorDataset(imgs_B)

def _unwrap(batch):
    return batch[0] if isinstance(batch, (list, tuple)) else batch

loader_A = DataLoader(domain_A, batch_size=CYCLEGAN_BATCH_SIZE, shuffle=True, drop_last=True)
loader_B = DataLoader(domain_B, batch_size=CYCLEGAN_BATCH_SIZE, shuffle=True, drop_last=True)

show_images(torch.stack([_unwrap(domain_A[i]) for i in range(8)]), title="Domain A samples", nrow=4)
show_images(torch.stack([_unwrap(domain_B[i]) for i in range(8)]), title="Domain B samples", nrow=4)
""")

md(r"### 8.3 Model Architecture")

code(r"""
class CycleGAN_ResBlock(nn.Module):
    '''Reflection-padded residual block -- CycleGAN's generator backbone.
    Reflection padding (rather than zero padding) avoids the border
    artifacts a translation network would otherwise learn near image
    edges, since edges get "seen" less often during training.'''
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGAN_Generator(nn.Module):
    '''Encoder (downsample x2) -> N residual blocks -> Decoder (upsample x2).
    InstanceNorm (not BatchNorm) throughout is the CycleGAN/pix2pix
    standard: normalizing PER IMAGE rather than per batch matters for
    style-transfer tasks, where each image's own contrast/brightness
    should be treated independently, not tied to other images in the batch.'''
    def __init__(self, nc=3, ngf=64, n_res_blocks=6):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(nc, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):   # downsample 64 -> 32 -> 16
            layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        for _ in range(n_res_blocks):
            layers += [CycleGAN_ResBlock(ch)]
        for _ in range(2):   # upsample back 16 -> 32 -> 64
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, nc, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CycleGAN_PatchDiscriminator(nn.Module):
    '''70x70 PatchGAN: instead of one real/fake logit for the whole image,
    output a GRID of logits, each judging a local patch. This focuses the
    discriminator (and hence the generator's gradient) on local texture
    realism rather than global layout -- exactly the right inductive bias
    for style/texture translation tasks.'''
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        self.net = nn.Sequential(
            *block(nc, ndf, norm=False),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            *block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)   # (B, 1, H', W') patch map of logits


G_AB = CycleGAN_Generator().to(DEVICE).apply(weights_init)   # horse -> zebra
G_BA = CycleGAN_Generator().to(DEVICE).apply(weights_init)   # zebra -> horse
D_A = CycleGAN_PatchDiscriminator().to(DEVICE).apply(weights_init)
D_B = CycleGAN_PatchDiscriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 8.4 Loss Functions & Optimizers")

code(r"""
# LSGAN (least-squares) adversarial loss: CycleGAN's actual choice --
# MSE against 1/0 targets instead of BCE. Empirically gives more stable
# gradients for image-translation tasks than the vanilla log-loss.
cyclegan_adv_criterion = nn.MSELoss()
cyclegan_cycle_criterion = nn.L1Loss()
cyclegan_identity_criterion = nn.L1Loss()

CYCLEGAN_LAMBDA_CYCLE = 10.0
CYCLEGAN_LAMBDA_IDENTITY = 5.0   # weight on identity loss relative to cycle loss (paper: 0.5x lambda_cycle)

cyclegan_opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=2e-4, betas=(0.5, 0.999))
cyclegan_opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=2e-4, betas=(0.5, 0.999))

class ImagePool:
    '''History buffer of previously generated fakes (Shrivastava et al.,
    2017, adopted by CycleGAN): update discriminators using a MIX of the
    latest fakes and older ones sampled from this pool, rather than only
    the latest batch. This damps the oscillation that can occur when D is
    trained exclusively against G's *current* (fast-moving) outputs.'''
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randrange(self.pool_size)
                result.append(self.images[idx].clone())
                self.images[idx] = img
            else:
                result.append(img)
        return torch.cat(result, dim=0)

pool_A, pool_B = ImagePool(), ImagePool()
""")

md(r"### 8.5 Training Loop")

code(r"""
cyclegan_history = []
for epoch in range(CYCLEGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter(), "cycle": AvgMeter()}
    pbar = tqdm(zip(loader_A, loader_B), total=min(len(loader_A), len(loader_B)),
                desc=f"[CycleGAN] epoch {epoch+1}/{CYCLEGAN_EPOCHS}")
    for real_A, real_B in pbar:
        real_A, real_B = _unwrap(real_A).to(DEVICE), _unwrap(real_B).to(DEVICE)
        b = real_A.size(0)

        # ---- Generators: adversarial + cycle-consistency + identity ----
        cyclegan_opt_G.zero_grad(set_to_none=True)
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        loss_adv_AB = cyclegan_adv_criterion(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        loss_adv_BA = cyclegan_adv_criterion(D_A(fake_A), torch.ones_like(D_A(fake_A)))

        recon_A = G_BA(fake_B)   # A -> B -> A
        recon_B = G_AB(fake_A)   # B -> A -> B
        loss_cycle = cyclegan_cycle_criterion(recon_A, real_A) + cyclegan_cycle_criterion(recon_B, real_B)

        # Identity loss: G_AB(real_B) should be ~real_B (already a zebra,
        # zebra->zebra should be near-identity) -- empirically helps
        # preserve color composition, especially for photo<->painting tasks.
        loss_idt = cyclegan_identity_criterion(G_AB(real_B), real_B) + cyclegan_identity_criterion(G_BA(real_A), real_A)

        loss_g = loss_adv_AB + loss_adv_BA + CYCLEGAN_LAMBDA_CYCLE * loss_cycle + CYCLEGAN_LAMBDA_IDENTITY * loss_idt
        loss_g.backward()
        cyclegan_opt_G.step()

        # ---- Discriminators (using the image pool) ----
        cyclegan_opt_D.zero_grad(set_to_none=True)
        fake_B_pooled = pool_B.query(fake_B.detach())
        fake_A_pooled = pool_A.query(fake_A.detach())
        loss_d_A = 0.5 * (cyclegan_adv_criterion(D_A(real_A), torch.ones_like(D_A(real_A))) +
                           cyclegan_adv_criterion(D_A(fake_A_pooled), torch.zeros_like(D_A(fake_A_pooled))))
        loss_d_B = 0.5 * (cyclegan_adv_criterion(D_B(real_B), torch.ones_like(D_B(real_B))) +
                           cyclegan_adv_criterion(D_B(fake_B_pooled), torch.zeros_like(D_B(fake_B_pooled))))
        loss_d = loss_d_A + loss_d_B
        loss_d.backward()
        cyclegan_opt_D.step()

        meters["G"].update(loss_g.item()); meters["D"].update(loss_d.item()); meters["cycle"].update(loss_cycle.item())
        pbar.set_postfix(loss_G=f"{loss_g.item():.3f}", loss_D=f"{loss_d.item():.3f}", cycle=f"{loss_cycle.item():.3f}")

    cyclegan_history.append((meters["G"].avg, meters["D"].avg, meters["cycle"].avg))
    with torch.no_grad():
        G_AB.eval()
        demo_fake_B = G_AB(real_A[:8])
        G_AB.train()
    show_images(torch.cat([real_A[:8], demo_fake_B]), title=f"CycleGAN A->B after epoch {epoch+1} (top=real A, bottom=fake B)", nrow=8)
    save_checkpoint({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict()},
                     f"{CHECKPOINT_DIR}/cyclegan_epoch{epoch+1}.pt")

print("CycleGAN training complete.")
""")

md(r"### 8.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def cyclegan_translate(images: torch.Tensor, direction: str = "A2B") -> torch.Tensor:
    '''Clean translation entry point. direction: "A2B" or "B2A".'''
    images = images.to(DEVICE)
    if direction == "A2B":
        G_AB.eval(); out = G_AB(images); G_AB.train()
    else:
        G_BA.eval(); out = G_BA(images); G_BA.train()
    return out

sample_A = torch.stack([_unwrap(domain_A[i]) for i in range(8)])
show_images(torch.cat([sample_A, cyclegan_translate(sample_A, "A2B")]),
            title="CycleGAN inference: top=input (A), bottom=translated (B)", nrow=8)
""")

md(r"### 8.7 Visualization & Results")

code(r"""
gs, ds, cs = zip(*cyclegan_history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(gs, label="G"); axes[0].plot(ds, label="D"); axes[0].legend(); axes[0].set_title("Adversarial losses")
axes[1].plot(cs); axes[1].set_title("Cycle-consistency L1 loss (should trend down)")
plt.show()

# Full round trip: A -> B -> A, to visually verify cycle-consistency held.
sample_A8 = torch.stack([_unwrap(domain_A[i]) for i in range(4)])
fake_B = cyclegan_translate(sample_A8, "A2B")
recon_A = cyclegan_translate(fake_B, "B2A")
show_images(torch.cat([sample_A8, fake_B, recon_A]),
            title="Round trip: real A (top) -> fake B (mid) -> reconstructed A (bottom)", nrow=4, figsize=(6, 5))
""")

md(r"### 8.8 User Input → Output Demo")

code(r"""
IMAGE_INDEX = 3          # which image from domain A to translate
DIRECTION = "A2B"        # "A2B" or "B2A"

user_img = _unwrap(domain_A[IMAGE_INDEX]).unsqueeze(0)
translated = cyclegan_translate(user_img, DIRECTION)
show_images(torch.cat([user_img, translated]), title=f"User demo: direction={DIRECTION}", nrow=2, figsize=(4, 2))
""")

md(r"""
### 8.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Cycle-consistency is a form of **self-supervision**: it
  manufactures a training signal (the reconstruction target *is* the
  input) in a setting where no direct supervision (paired data) exists at
  all -- a pattern reused far beyond GANs (e.g. in unsupervised
  representation learning).
- ⚠️ **Warning.** Cycle-consistency only constrains *content preservation*,
  not *correctness of the translation* -- CycleGAN can and does learn
  degenerate solutions that hide information imperceptibly in the "fake"
  image purely to make the cycle reconstruct well (documented as steganographic
  artifacts in follow-up work), rather than truly discarding
  domain-A-specific content.
- ⚠️ **Warning.** Because there is no pairing constraint, CycleGAN cannot
  guarantee *geometric* consistency (e.g. it may translate texture well
  but distort shape) -- this is a known limitation the PatchGAN's local
  focus can actually make worse, since it never "sees" global structure.

**This GAN solved the problem of: translating images between two visual
domains using only unpaired example collections from each domain, with no
example ever telling the model "this A corresponds to this B".**
""")

\
# ---------------------------------------------------------------------------
# SECTION 9 -- SRGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="9"></a>
## 9. SRGAN -- Single-image super-resolution

### 9.1 Problem Statement & Why this GAN

Ledig et al. (2017) tackled **single-image super-resolution**: given only
a low-resolution image, hallucinate a plausible 4x higher-resolution
version. Pure pixel-MSE-trained super-resolution networks (pre-SRGAN)
produce blurry outputs, because MSE is minimized by the *average* of all
plausible high-frequency completions -- exactly the wrong objective when
"plausible-looking detail" matters more than "literally correct per
-pixel value" (a genuinely different problem from every earlier section:
here the input is a full image, not noise, and there is a well-defined
"correct-ish" target). SRGAN's fix: train with a **perceptual loss**
(distance in a pretrained VGG's feature space, not pixel space) *plus* an
**adversarial loss**, so the generator is rewarded for producing detail
that *looks* real, not detail that minimizes pixel-wise error.

### 9.2 Data Collection & Preparation
""")

code(r"""
SRGAN_HR_SIZE = 96          # STL-10's native resolution -- convenient, no cropping needed
SRGAN_SCALE = 4
SRGAN_LR_SIZE = SRGAN_HR_SIZE // SRGAN_SCALE   # 24x24 input -> 96x96 output
SRGAN_BATCH_SIZE = 16
SRGAN_EPOCHS = 6

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

srgan_dataset = SRDataset(srgan_hr_base)
srgan_loader = DataLoader(srgan_dataset, batch_size=SRGAN_BATCH_SIZE, shuffle=True, drop_last=True)
lr_batch, hr_batch = next(iter(srgan_loader))
show_images(F.interpolate(lr_batch[:8], size=SRGAN_HR_SIZE, mode="nearest"), title="LR inputs (nearest-upsampled for display)", nrow=8, figsize=(12, 2))
show_images(hr_batch[:8], title="HR targets", nrow=8, figsize=(12, 2))
""")

md(r"### 9.3 Model Architecture")

code(r"""
class SRGAN_ResBlock(nn.Module):
    '''BN + PReLU residual block operating entirely at LOW resolution
    (before upsampling) -- keeps the expensive deep processing cheap, with
    all detail-adding upsampling deferred to the end via PixelShuffle.'''
    def __init__(self, ch=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class PixelShuffleUpsample(nn.Module):
    '''Sub-pixel convolution (Shi et al., 2016): convolve to 4x the target
    channels, then rearrange (PixelShuffle) into 2x spatial resolution.
    This produces far fewer checkerboard artifacts than transposed
    convolution, which is why SRGAN (and most later SR GANs) use it.'''
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class SRGAN_Generator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_res_blocks=8, scale=SRGAN_SCALE):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(in_ch, base_ch, 9, 1, 4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[SRGAN_ResBlock(base_ch) for _ in range(n_res_blocks)])
        self.mid = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.BatchNorm2d(base_ch))
        n_upsamples = int(math.log2(scale))
        self.upsample = nn.Sequential(*[PixelShuffleUpsample(base_ch) for _ in range(n_upsamples)])
        self.exit = nn.Conv2d(base_ch, 3, 9, 1, 4)

    def forward(self, x):
        entry_feat = self.entry(x)
        res_feat = self.mid(self.res_blocks(entry_feat))
        feat = entry_feat + res_feat   # global residual skip AROUND all residual blocks (paper's key stabilizer for deep SR nets)
        feat = self.upsample(feat)
        return torch.tanh(self.exit(feat))


class SRGAN_Discriminator(nn.Module):
    '''VGG-style deep classifier at the HR resolution -- alternating
    stride-1/stride-2 conv blocks with growing channel count, judging
    "is this a real HR photo or an upsampled fake" globally (unlike
    CycleGAN's PatchGAN, a single global decision suits the SR task,
    since realism here is a property of the whole image's sharpness).'''
    def __init__(self, base_ch=64):
        super().__init__()
        def conv_bn_lrelu(in_c, out_c, stride, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        layers = []
        layers += conv_bn_lrelu(3, base_ch, 1, bn=False)
        layers += conv_bn_lrelu(base_ch, base_ch, 2)
        layers += conv_bn_lrelu(base_ch, base_ch * 2, 1)
        layers += conv_bn_lrelu(base_ch * 2, base_ch * 2, 2)
        layers += conv_bn_lrelu(base_ch * 2, base_ch * 4, 1)
        layers += conv_bn_lrelu(base_ch * 4, base_ch * 4, 2)
        layers += conv_bn_lrelu(base_ch * 4, base_ch * 8, 1)
        layers += conv_bn_lrelu(base_ch * 8, base_ch * 8, 2)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(6), nn.Flatten(),
            nn.Linear(base_ch * 8 * 6 * 6, 1024), nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).view(-1)


G_srgan = SRGAN_Generator().to(DEVICE)
D_srgan = SRGAN_Discriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 9.4 Loss Functions & Optimizers")

code(r"""
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

def srgan_content_loss(sr, hr):
    if srgan_vgg is None:
        return F.mse_loss(sr, hr)
    sr_n = (denorm(sr) - srgan_vgg_mean) / srgan_vgg_std
    hr_n = (denorm(hr) - srgan_vgg_mean) / srgan_vgg_std
    return F.mse_loss(srgan_vgg(sr_n), srgan_vgg(hr_n))

srgan_adv_criterion = nn.BCEWithLogitsLoss()
SRGAN_LAMBDA_ADV = 1e-3   # the paper's own weighting: adversarial term is intentionally tiny relative to content loss
srgan_opt_G = torch.optim.Adam(G_srgan.parameters(), lr=1e-4, betas=(0.9, 0.999))
srgan_opt_D = torch.optim.Adam(D_srgan.parameters(), lr=1e-4, betas=(0.9, 0.999))
""")

md(r"### 9.5 Training Loop")

code(r"""
def psnr(sr, hr):
    '''Peak Signal-to-Noise Ratio in dB -- the standard quantitative metric
    for super-resolution (higher = closer to the ground-truth HR image).'''
    mse = F.mse_loss(denorm(sr), denorm(hr)).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))

srgan_history = []
for epoch in range(SRGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter(), "PSNR": AvgMeter()}
    pbar = tqdm(srgan_loader, desc=f"[SRGAN] epoch {epoch+1}/{SRGAN_EPOCHS}")
    for lr, hr in pbar:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        b = lr.size(0)

        sr = G_srgan(lr)

        # ---- Discriminator ----
        D_srgan.zero_grad(set_to_none=True)
        loss_d = srgan_adv_criterion(D_srgan(hr), torch.ones(b, device=DEVICE)) + \
                 srgan_adv_criterion(D_srgan(sr.detach()), torch.zeros(b, device=DEVICE))
        loss_d.backward()
        srgan_opt_D.step()

        # ---- Generator: content (perceptual) + tiny adversarial term ----
        G_srgan.zero_grad(set_to_none=True)
        content = srgan_content_loss(sr, hr)
        adv = srgan_adv_criterion(D_srgan(sr), torch.ones(b, device=DEVICE))
        loss_g = content + SRGAN_LAMBDA_ADV * adv
        loss_g.backward()
        srgan_opt_G.step()

        meters["G"].update(loss_g.item()); meters["D"].update(loss_d.item()); meters["PSNR"].update(psnr(sr, hr))
        pbar.set_postfix(loss_G=f"{loss_g.item():.3f}", loss_D=f"{loss_d.item():.3f}", PSNR=f"{meters['PSNR'].avg:.1f}dB")

    srgan_history.append((meters["G"].avg, meters["D"].avg, meters["PSNR"].avg))
    with torch.no_grad():
        G_srgan.eval()
        sample_sr = G_srgan(lr[:4])
        G_srgan.train()
    show_images(torch.cat([F.interpolate(lr[:4], size=SRGAN_HR_SIZE, mode="nearest"), sample_sr, hr[:4]]),
                title=f"SRGAN epoch {epoch+1}: LR (nearest) / SR / HR", nrow=4)
    save_checkpoint({"G": G_srgan.state_dict(), "D": D_srgan.state_dict()}, f"{CHECKPOINT_DIR}/srgan_epoch{epoch+1}.pt")

print("SRGAN training complete.")
""")

md(r"### 9.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def srgan_upscale(lr_images: torch.Tensor) -> torch.Tensor:
    '''Clean inference entry point: LR image(s) in ([-1,1], any small size),
    4x super-resolved image(s) out.'''
    G_srgan.eval()
    sr = G_srgan(lr_images.to(DEVICE))
    G_srgan.train()
    return sr

demo_lr = lr_batch[:4].to(DEVICE)
show_images(srgan_upscale(demo_lr), title="SRGAN inference: 4x super-resolved", nrow=4)
""")

md(r"### 9.7 Visualization & Results")

code(r"""
gs, ds, psnrs = zip(*srgan_history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(gs, label="G"); axes[0].plot(ds, label="D"); axes[0].legend(); axes[0].set_title("Losses")
axes[1].plot(psnrs); axes[1].set_title("PSNR (dB) -- higher is better")
plt.show()

# Side-by-side: bicubic upsampling (a naive, non-learned baseline) vs SRGAN vs ground truth.
with torch.no_grad():
    lr4, hr4 = lr_batch[:4].to(DEVICE), hr_batch[:4].to(DEVICE)
    bicubic = F.interpolate(lr4, size=SRGAN_HR_SIZE, mode="bicubic", align_corners=False)
    sr4 = srgan_upscale(lr4)
fig, axes = plt.subplots(4, 3, figsize=(6, 8))
for row in range(4):
    for col, (img, name) in enumerate(zip([bicubic[row], sr4[row], hr4[row]], ["Bicubic", "SRGAN", "Ground truth"])):
        axes[row, col].imshow(denorm(img).permute(1, 2, 0).cpu()); axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(name)
plt.tight_layout(); plt.show()
print(f"Bicubic PSNR: {psnr(bicubic, hr4):.2f}dB | SRGAN PSNR: {psnr(sr4, hr4):.2f}dB")
""")

md(r"### 9.8 User Input → Output Demo")

code(r"""
IMAGE_INDEX = 2   # which validation image to super-resolve

user_lr = lr_batch[IMAGE_INDEX:IMAGE_INDEX + 1]
user_sr = srgan_upscale(user_lr)
show_images(torch.cat([F.interpolate(user_lr, size=SRGAN_HR_SIZE, mode="nearest"), user_sr]),
            title="User demo: LR input (nearest-upsampled) vs SRGAN output", nrow=2, figsize=(4, 2))
""")

md(r"""
### 9.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** "Lower pixel MSE" and "looks more realistic" are
  genuinely different objectives for super-resolution -- SRGAN explicitly
  trades a bit of the former for a lot of the latter, which is why SRGAN
  images often score *worse* on PSNR than a purely MSE-trained baseline
  while looking clearly sharper/more detailed to a human. Always report
  both a perceptual comparison AND PSNR/SSIM, never PSNR alone.
- ⚠️ **Warning.** SRGAN can hallucinate texture that is plausible but
  factually wrong (invented detail not present in the true HR image) --
  unacceptable for domains needing faithful reconstruction (medical
  imaging, forensics), fine for domains where "looks good" is the goal.

**This GAN solved the problem of: single-image super-resolution --
recovering plausible high-frequency detail from a low-resolution input,
optimized for perceptual realism rather than pixel-wise accuracy alone.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 10 -- ESRGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="10"></a>
## 10. ESRGAN -- Enhanced perceptual super-resolution

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

We reuse the exact same (LR, HR) STL-10-derived pairing pipeline built in
Section 9.2 (`srgan_dataset` / `srgan_loader`) -- ESRGAN and SRGAN solve
the identical super-resolution task, so there is no reason to redefine or
re-download the data. **Swap in DIV2K (`eugenesiow/Div2k` on Hugging Face
Datasets, or the official DIV2K download) for a production-quality run.**
""")

code(r"""
esrgan_loader = srgan_loader   # identical (LR, HR) pipeline as Section 9 -- see 10.2 markdown
lr_batch_e, hr_batch_e = next(iter(esrgan_loader))
print(f"ESRGAN reusing SRGAN's data pipeline: {len(esrgan_loader.dataset)} (LR, HR) pairs, "
      f"LR={tuple(lr_batch_e.shape[-2:])} -> HR={tuple(hr_batch_e.shape[-2:])}")
""")

md(r"### 10.3 Model Architecture")

code(r"""
class ESRGAN_DenseBlock(nn.Module):
    '''DenseNet-style block: each conv sees the CONCATENATION of the block
    input and every previous conv's output, maximizing feature reuse and
    gradient flow. Output is added back with a small residual scale
    (`beta`, not 1.0) -- ESRGAN's "residual scaling" trick, which the
    paper found necessary to keep very deep RRDB stacks stable without
    any normalization layer at all.'''
    def __init__(self, ch=64, growth=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(ch, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(ch + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(ch + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(ch + 4 * growth, ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + self.beta * x5


class RRDB(nn.Module):
    '''Residual-in-Residual Dense Block: three ESRGAN_DenseBlocks chained,
    with an OUTER residual-scaled skip connection around all three -- the
    "residual in residual" in the name. No BatchNorm anywhere in this block.'''
    def __init__(self, ch=64, growth=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.blocks = nn.Sequential(ESRGAN_DenseBlock(ch, growth, beta),
                                     ESRGAN_DenseBlock(ch, growth, beta),
                                     ESRGAN_DenseBlock(ch, growth, beta))

    def forward(self, x):
        return x + self.beta * self.blocks(x)


class ESRGAN_UpsampleBlock(nn.Module):
    '''Nearest-neighbor upsample + conv (NOT PixelShuffle) -- the ESRGAN
    paper's choice, which it found produces fewer checkerboard artifacts
    than SRGAN's sub-pixel convolution at the cost of slightly more
    compute.'''
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.act(self.conv(x))


class ESRGAN_Generator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_rrdb=6, growth=32, scale=SRGAN_SCALE):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(base_ch, growth) for _ in range(n_rrdb)])
        self.body_conv = nn.Conv2d(base_ch, base_ch, 3, 1, 1)
        n_ups = int(math.log2(scale))
        self.upsample = nn.Sequential(*[ESRGAN_UpsampleBlock(base_ch) for _ in range(n_ups)])
        self.hr_conv = nn.Conv2d(base_ch, base_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self.exit = nn.Conv2d(base_ch, 3, 3, 1, 1)

    def forward(self, x):
        entry_feat = self.entry(x)
        body_feat = self.body_conv(self.body(entry_feat))
        feat = entry_feat + body_feat   # global residual around the entire RRDB stack
        feat = self.upsample(feat)
        feat = self.act(self.hr_conv(feat))
        return torch.tanh(self.exit(feat))


class ESRGAN_Discriminator(nn.Module):
    '''Same VGG-style architecture as SRGAN's discriminator (Section 9.3) --
    what differs is not the network but the LOSS computed from its output
    (see 10.4: relativistic average, not absolute, real/fake).'''
    def __init__(self, base_ch=64):
        super().__init__()
        def conv_bn_lrelu(in_c, out_c, stride, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        layers = []
        layers += conv_bn_lrelu(3, base_ch, 1, bn=False)
        layers += conv_bn_lrelu(base_ch, base_ch, 2)
        layers += conv_bn_lrelu(base_ch, base_ch * 2, 1)
        layers += conv_bn_lrelu(base_ch * 2, base_ch * 2, 2)
        layers += conv_bn_lrelu(base_ch * 2, base_ch * 4, 1)
        layers += conv_bn_lrelu(base_ch * 4, base_ch * 4, 2)
        layers += conv_bn_lrelu(base_ch * 4, base_ch * 8, 1)
        layers += conv_bn_lrelu(base_ch * 8, base_ch * 8, 2)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(6), nn.Flatten(),
            nn.Linear(base_ch * 8 * 6 * 6, 1024), nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).view(-1)


G_esrgan = ESRGAN_Generator().to(DEVICE)
D_esrgan = ESRGAN_Discriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 10.4 Loss Functions & Optimizers")

code(r"""
# Reuse the VGG perceptual-loss network built in Section 9.4 if available.
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

md(r"### 10.5 Training Loop")

code(r"""
esrgan_history = []
for epoch in range(ESRGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter(), "PSNR": AvgMeter()}
    pbar = tqdm(esrgan_loader, desc=f"[ESRGAN] epoch {epoch+1}/{ESRGAN_EPOCHS}")
    for lr, hr in pbar:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        sr = G_esrgan(lr)

        # ---- Discriminator (relativistic) ----
        D_esrgan.zero_grad(set_to_none=True)
        logits_real = D_esrgan(hr)
        logits_fake = D_esrgan(sr.detach())
        loss_d, _ = esrgan_relativistic_loss(logits_real, logits_fake)
        loss_d.backward()
        esrgan_opt_D.step()

        # ---- Generator: pixel + perceptual + relativistic-adversarial ----
        G_esrgan.zero_grad(set_to_none=True)
        logits_real_for_g = D_esrgan(hr).detach()
        logits_fake_for_g = D_esrgan(sr)
        _, loss_adv_g = esrgan_relativistic_loss(logits_real_for_g, logits_fake_for_g)
        loss_pixel = F.l1_loss(sr, hr)
        loss_content = esrgan_content_loss(sr, hr)
        loss_g = ESRGAN_LAMBDA_PIXEL * loss_pixel + loss_content + ESRGAN_LAMBDA_ADV * loss_adv_g
        loss_g.backward()
        esrgan_opt_G.step()

        meters["G"].update(loss_g.item()); meters["D"].update(loss_d.item()); meters["PSNR"].update(psnr(sr, hr))
        pbar.set_postfix(loss_G=f"{loss_g.item():.3f}", loss_D=f"{loss_d.item():.3f}", PSNR=f"{meters['PSNR'].avg:.1f}dB")

    esrgan_history.append((meters["G"].avg, meters["D"].avg, meters["PSNR"].avg))
    with torch.no_grad():
        G_esrgan.eval()
        sample_sr = G_esrgan(lr[:4])
        G_esrgan.train()
    show_images(torch.cat([F.interpolate(lr[:4], size=SRGAN_HR_SIZE, mode="nearest"), sample_sr, hr[:4]]),
                title=f"ESRGAN epoch {epoch+1}: LR (nearest) / SR / HR", nrow=4)
    save_checkpoint({"G": G_esrgan.state_dict(), "D": D_esrgan.state_dict()}, f"{CHECKPOINT_DIR}/esrgan_epoch{epoch+1}.pt")

print("ESRGAN training complete.")
""")

md(r"### 10.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def esrgan_upscale(lr_images: torch.Tensor) -> torch.Tensor:
    G_esrgan.eval()
    sr = G_esrgan(lr_images.to(DEVICE))
    G_esrgan.train()
    return sr

demo_lr_e = lr_batch_e[:4].to(DEVICE)
show_images(esrgan_upscale(demo_lr_e), title="ESRGAN inference: 4x super-resolved", nrow=4)
""")

md(r"### 10.7 Visualization & Results")

code(r"""
gs, ds, psnrs = zip(*esrgan_history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(gs, label="G"); axes[0].plot(ds, label="D"); axes[0].legend(); axes[0].set_title("Losses")
axes[1].plot(psnrs); axes[1].set_title("PSNR (dB)")
plt.show()

# Direct SRGAN vs ESRGAN comparison on the SAME inputs -- the core
# deliverable of this section (does ESRGAN actually look better?).
with torch.no_grad():
    lr4, hr4 = lr_batch_e[:4].to(DEVICE), hr_batch_e[:4].to(DEVICE)
    sr_srgan = srgan_upscale(lr4)
    sr_esrgan = esrgan_upscale(lr4)
fig, axes = plt.subplots(4, 3, figsize=(6, 8))
for row in range(4):
    for col, (img, name) in enumerate(zip([sr_srgan[row], sr_esrgan[row], hr4[row]], ["SRGAN", "ESRGAN", "Ground truth"])):
        axes[row, col].imshow(denorm(img).permute(1, 2, 0).cpu()); axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(name)
plt.tight_layout(); plt.show()
print(f"SRGAN PSNR: {psnr(sr_srgan, hr4):.2f}dB | ESRGAN PSNR: {psnr(sr_esrgan, hr4):.2f}dB "
      f"(remember: higher PSNR != more realistic, see 9.9)")
""")

md(r"### 10.8 User Input → Output Demo")

code(r"""
IMAGE_INDEX = 1

user_lr_e = lr_batch_e[IMAGE_INDEX:IMAGE_INDEX + 1]
user_sr_e = esrgan_upscale(user_lr_e)
show_images(torch.cat([F.interpolate(user_lr_e, size=SRGAN_HR_SIZE, mode="nearest"), user_sr_e]),
            title="User demo: LR input (nearest-upsampled) vs ESRGAN output", nrow=2, figsize=(4, 2))
""")

md(r"""
### 10.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Every ESRGAN change (RRDB, no-BN, relativistic
  discriminator, L1-not-L2 perceptual loss) targets a *specific, named
  artifact* SRGAN exhibited -- this is a good general lesson for iterating
  on GANs: identify one concrete failure mode at a time rather than
  redesigning the whole system.
- 🚀 **Pro Tip.** The official ESRGAN paper also releases a **PSNR
  -oriented** (no adversarial loss, pure L1) checkpoint and linearly
  interpolates its weights with the GAN-trained checkpoint at inference
  time -- a cheap way to trade sharpness for fewer hallucinated artifacts
  without retraining.
- ⚠️ **Warning.** RRDB's dense concatenations use more memory per layer
  than plain residual blocks -- our lightweight 6-RRDB model is already
  near the practical CPU-training limit; scale `n_rrdb` up only with a GPU.

**This GAN solved the problem of: single-image super-resolution with
fewer artifacts and more natural texture than SRGAN, by removing
BatchNorm, deepening the residual structure (RRDB), and switching to a
relative (not absolute) adversarial judgment.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 11 -- GANSYNTH
# ---------------------------------------------------------------------------
md(r"""
<a id="11"></a>
## 11. GANSynth -- Audio / music generation

### 11.1 Problem Statement & Why this GAN

Engel et al. (2019) asked whether GANs -- until then almost exclusively an
image tool -- could generate **musical audio** faster and at higher
fidelity than the autoregressive models (WaveNet, SampleRNN) dominant at
the time. Their key insight: don't make the GAN output a raw waveform
(a very long, temporally-fine-grained 1-D signal that convolutional GANs
handle poorly); instead output a **time-frequency spectrogram image**
(so all the image-GAN machinery from Sections 1-10 applies directly), then
convert that spectrogram back to audio. GANSynth's full pipeline also
predicts **instantaneous frequency** (the phase derivative) alongside
magnitude for higher-fidelity phase reconstruction; we simplify to
**log-magnitude spectrograms + Griffin-Lim phase reconstruction** to keep
the demo dependency-free and fast, while keeping the core idea --
*generate audio by generating an image of audio* -- fully intact. This is
the **audio/music generation** problem, conditioned on musical pitch.

### 11.2 Data Collection & Preparation

Real NSynth is a ~30GB download of individual instrument notes --
inappropriate for a guaranteed-offline demo. We instead **synthesize** a
dataset of musical notes via **additive synthesis** (random-weighted sine
harmonics = random timbre, at a controlled MIDI pitch) using
`torchaudio`'s standard signal-processing transforms for the actual
spectrogram pipeline -- a real, working small mel/spectrogram dataset, as
suggested for this section. **Swap this for real NSynth
(`tensorflow-datasets`'s `nsynth` config, or a Hugging Face NSynth mirror)
for musically realistic timbres.**
""")

code(r"""
import torchaudio

GANSYNTH_SR = 16000
GANSYNTH_DURATION = 0.5                       # seconds per note
GANSYNTH_N_FFT = 126
GANSYNTH_HOP = 125                            # must be <= n_fft for istft/Griffin-Lim; chosen with n_fft to land near 64 time frames
GANSYNTH_N_FREQ = GANSYNTH_N_FFT // 2 + 1     # 64 -- chosen so the spectrogram is a clean square "image"
GANSYNTH_N_FRAMES = 64
GANSYNTH_PITCH_MIN, GANSYNTH_PITCH_MAX = 48, 72   # 2 octaves of MIDI pitches (C3 - C5)
GANSYNTH_N_PITCHES = GANSYNTH_PITCH_MAX - GANSYNTH_PITCH_MIN + 1
GANSYNTH_BATCH_SIZE = 32
GANSYNTH_EPOCHS = 8
GANSYNTH_NZ = 64

spec_transform = torchaudio.transforms.Spectrogram(n_fft=GANSYNTH_N_FFT, hop_length=GANSYNTH_HOP, power=2.0)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
griffinlim = torchaudio.transforms.GriffinLim(n_fft=GANSYNTH_N_FFT, hop_length=GANSYNTH_HOP, power=2.0, n_iter=32)

def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def synth_note(midi_pitch: int, rng, n_harmonics: int = 6) -> torch.Tensor:
    '''Additive synthesis: sum of sine harmonics at random relative
    amplitudes (= a random "timbre") at the requested pitch's fundamental
    frequency, with a simple exponential decay envelope (like a plucked
    string). This is what makes each pitch's training examples diverse in
    timbre, giving the GAN something nontrivial to model beyond a single
    fixed waveform per pitch.'''
    freq = midi_to_freq(midi_pitch)
    t = np.linspace(0, GANSYNTH_DURATION, int(GANSYNTH_SR * GANSYNTH_DURATION), endpoint=False)
    harmonic_amps = rng.dirichlet(np.ones(n_harmonics))   # random mixture summing to 1 -> random timbre
    wave = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        wave += harmonic_amps[h - 1] * np.sin(2 * np.pi * freq * h * t)
    envelope = np.exp(-3.0 * t / GANSYNTH_DURATION)
    wave = wave * envelope
    wave = wave / (np.max(np.abs(wave)) + 1e-8)
    wave = wave + rng.normal(0, 0.005, size=wave.shape)   # light noise floor, like real recordings
    return torch.tensor(wave, dtype=torch.float32)

def wave_to_spec_db(wave: torch.Tensor) -> torch.Tensor:
    spec_db = amp_to_db(spec_transform(wave.unsqueeze(0))).squeeze(0)   # (freq, time), roughly in [-80, 0] dB
    if spec_db.size(1) < GANSYNTH_N_FRAMES:
        spec_db = F.pad(spec_db, (0, GANSYNTH_N_FRAMES - spec_db.size(1)), value=-80.0)
    return spec_db[:, :GANSYNTH_N_FRAMES]

def spec_db_to_wave(spec_db: torch.Tensor) -> torch.Tensor:
    '''Approximate inverse: dB -> power spectrogram -> Griffin-Lim phase
    reconstruction. This is a classical DSP algorithm (no learned phase
    model), which is why real GANSynth's instantaneous-frequency approach
    sounds meaningfully cleaner -- but Griffin-Lim needs no extra network
    or training and is "good enough" to hear pitch/timbre clearly.'''
    spec_power = torch.pow(10.0, spec_db.clamp(-80, 0) / 10.0)
    return griffinlim(spec_power.unsqueeze(0)).squeeze(0)

class GANSynthDataset(Dataset):
    def __init__(self, n_per_pitch=24, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.pitch_idx_list = [p for p in range(GANSYNTH_N_PITCHES) for _ in range(n_per_pitch)]

    def __len__(self):
        return len(self.pitch_idx_list)

    def __getitem__(self, i):
        pitch_idx = self.pitch_idx_list[i]
        wave = synth_note(GANSYNTH_PITCH_MIN + pitch_idx, self.rng)
        spec_db = wave_to_spec_db(wave)
        spec_norm = (spec_db.clamp(-80, 0) + 40) / 40   # map [-80, 0] dB -> [-1, 1] (matches Tanh generator output)
        return spec_norm.unsqueeze(0), pitch_idx

gansynth_dataset = GANSynthDataset()
gansynth_loader = DataLoader(gansynth_dataset, batch_size=GANSYNTH_BATCH_SIZE, shuffle=True, drop_last=True)
spec_batch, pitch_batch = next(iter(gansynth_loader))
print(f"Synthetic GANSynth dataset: {len(gansynth_dataset)} notes across {GANSYNTH_N_PITCHES} pitches, "
      f"spectrogram shape {tuple(spec_batch.shape[-2:])}")
show_images(spec_batch[:16], title="Real training spectrograms (as images)", nrow=4, normalize=True)

try:
    from IPython.display import Audio, display
    display(Audio(spec_db_to_wave(spec_batch[0, 0] * 40 - 40).numpy(), rate=GANSYNTH_SR))
except Exception as e:
    print(f"Inline audio playback unavailable in this environment ({e}); "
          f"spec_db_to_wave(...) still returns a valid waveform tensor you can save with torchaudio.save(...).")
""")

md(r"### 11.3 Model Architecture")

code(r"""
class GANSynth_Generator(nn.Module):
    '''Identical mechanism to Section 2's cGAN generator -- pitch-conditional
    image generation -- pointed at a spectrogram "image" instead of a
    photo. This is the central lesson of GANSynth: once audio is
    represented as a 2-D time-frequency image, it IS an image-generation
    problem.'''
    def __init__(self, nz=GANSYNTH_NZ, n_pitches=GANSYNTH_N_PITCHES, ngf=64, embed_dim=32):
        super().__init__()
        self.pitch_embed = nn.Embedding(n_pitches, embed_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),                                    # 1 -> 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),                                    # 4 -> 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),                                    # 8 -> 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),                                        # 16 -> 32
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False), nn.Tanh(),                # 32 -> 64
        )

    def forward(self, z, pitch):
        pe = self.pitch_embed(pitch)
        zc = torch.cat([z, pe], dim=1)
        return self.net(zc.view(zc.size(0), -1, 1, 1))


class GANSynth_Discriminator(nn.Module):
    def __init__(self, n_pitches=GANSYNTH_N_PITCHES, ndf=64, spec_size=GANSYNTH_N_FRAMES):
        super().__init__()
        self.spec_size = spec_size
        self.pitch_embed = nn.Embedding(n_pitches, spec_size * spec_size)
        self.net = nn.Sequential(
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, spec, pitch):
        pitch_map = self.pitch_embed(pitch).view(-1, 1, self.spec_size, self.spec_size)
        return self.net(torch.cat([spec, pitch_map], dim=1)).view(-1)


G_gansynth = GANSynth_Generator().to(DEVICE).apply(weights_init)
D_gansynth = GANSynth_Discriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 11.4 Loss Functions & Optimizers")

code(r"""
gansynth_criterion = nn.BCEWithLogitsLoss()
gansynth_opt_G = torch.optim.Adam(G_gansynth.parameters(), lr=2e-4, betas=(0.5, 0.999))
gansynth_opt_D = torch.optim.Adam(D_gansynth.parameters(), lr=2e-4, betas=(0.5, 0.999))
""")

md(r"### 11.5 Training Loop")

code(r"""
gansynth_history = []
for epoch in range(GANSYNTH_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter()}
    pbar = tqdm(gansynth_loader, desc=f"[GANSynth] epoch {epoch+1}/{GANSYNTH_EPOCHS}")
    for spec, pitch in pbar:
        spec, pitch = spec.to(DEVICE), pitch.to(DEVICE)
        b = spec.size(0)

        D_gansynth.zero_grad(set_to_none=True)
        logits_real = D_gansynth(spec, pitch)
        loss_d_real = gansynth_criterion(logits_real, torch.ones(b, device=DEVICE))
        z = torch.randn(b, GANSYNTH_NZ, device=DEVICE)
        fake_pitch = torch.randint(0, GANSYNTH_N_PITCHES, (b,), device=DEVICE)
        fake_spec = G_gansynth(z, fake_pitch)
        logits_fake = D_gansynth(fake_spec.detach(), fake_pitch)
        loss_d_fake = gansynth_criterion(logits_fake, torch.zeros(b, device=DEVICE))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        gansynth_opt_D.step()

        G_gansynth.zero_grad(set_to_none=True)
        logits_fake_g = D_gansynth(fake_spec, fake_pitch)
        loss_g = gansynth_criterion(logits_fake_g, torch.ones(b, device=DEVICE))
        loss_g.backward()
        gansynth_opt_G.step()

        meters["D"].update(loss_d.item()); meters["G"].update(loss_g.item())
        pbar.set_postfix(loss_D=f"{loss_d.item():.3f}", loss_G=f"{loss_g.item():.3f}")

    gansynth_history.append((meters["G"].avg, meters["D"].avg))
    with torch.no_grad():
        G_gansynth.eval()
        demo_pitch = torch.arange(GANSYNTH_N_PITCHES, device=DEVICE)[:16] % GANSYNTH_N_PITCHES
        demo_z = torch.randn(16, GANSYNTH_NZ, device=DEVICE)
        sample = G_gansynth(demo_z, demo_pitch)
        G_gansynth.train()
    show_images(sample, title=f"GANSynth generated spectrograms after epoch {epoch+1}", nrow=4)
    save_checkpoint({"G": G_gansynth.state_dict(), "D": D_gansynth.state_dict()}, f"{CHECKPOINT_DIR}/gansynth_epoch{epoch+1}.pt")

print("GANSynth training complete.")
""")

md(r"### 11.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def gansynth_generate(midi_pitch: int, seed: int = None) -> torch.Tensor:
    '''Text/number-in (a MIDI pitch), audio waveform out -- the clean
    end-to-end user-facing entry point for this section.'''
    if seed is not None:
        torch.manual_seed(seed)
    pitch_idx = max(0, min(GANSYNTH_N_PITCHES - 1, midi_pitch - GANSYNTH_PITCH_MIN))
    G_gansynth.eval()
    z = torch.randn(1, GANSYNTH_NZ, device=DEVICE)
    pitch_t = torch.tensor([pitch_idx], device=DEVICE)
    spec_norm = G_gansynth(z, pitch_t)[0, 0].cpu()
    G_gansynth.train()
    spec_db = spec_norm * 40 - 40
    return spec_db_to_wave(spec_db)

wave = gansynth_generate(midi_pitch=60, seed=0)   # 60 = middle C
print(f"Generated {wave.numel()/GANSYNTH_SR:.1f}s of audio at {GANSYNTH_SR}Hz for MIDI pitch 60 (middle C).")
try:
    from IPython.display import Audio, display
    display(Audio(wave.numpy(), rate=GANSYNTH_SR))
except Exception:
    pass
""")

md(r"### 11.7 Visualization & Results")

code(r"""
gs, ds = zip(*gansynth_history)
plt.figure(figsize=(6, 4)); plt.plot(gs, label="G"); plt.plot(ds, label="D")
plt.legend(); plt.title("GANSynth training losses"); plt.show()

# Pitch sweep: same z (same "timbre"), varying requested pitch -- checks
# that pitch conditioning actually shifts the spectrogram's energy to a
# higher fundamental frequency band, not just an arbitrary different image.
with torch.no_grad():
    G_gansynth.eval()
    z_fixed = torch.randn(1, GANSYNTH_NZ, device=DEVICE).repeat(6, 1)
    pitches_to_show = torch.linspace(0, GANSYNTH_N_PITCHES - 1, 6).long().to(DEVICE)
    specs = G_gansynth(z_fixed, pitches_to_show)
    G_gansynth.train()
show_images(specs, title="Same latent, increasing pitch (low -> high, left to right)", nrow=6, figsize=(12, 2))

# Latent interpolation in timbre space at a FIXED pitch -- audible/visible
# smooth timbre morphing is the audio analogue of Section 1's image
# interpolation sanity check.
with torch.no_grad():
    G_gansynth.eval()
    z1, z2 = torch.randn(1, GANSYNTH_NZ, device=DEVICE), torch.randn(1, GANSYNTH_NZ, device=DEVICE)
    alphas = torch.linspace(0, 1, 6, device=DEVICE).view(-1, 1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    pitch_fixed = torch.full((6,), GANSYNTH_N_PITCHES // 2, dtype=torch.long, device=DEVICE)
    specs_interp = G_gansynth(z_interp, pitch_fixed)
    G_gansynth.train()
show_images(specs_interp, title="Timbre interpolation at fixed pitch", nrow=6, figsize=(12, 2))
""")

md(r"### 11.8 User Input → Output Demo")

code(r"""
USER_MIDI_PITCH = 67    # 67 = G4
RANDOM_SEED = 3

user_wave = gansynth_generate(USER_MIDI_PITCH, seed=RANDOM_SEED)
print(f"User requested MIDI pitch {USER_MIDI_PITCH} (~{midi_to_freq(USER_MIDI_PITCH):.1f} Hz)")
try:
    from IPython.display import Audio, display
    display(Audio(user_wave.numpy(), rate=GANSYNTH_SR))
except Exception:
    pass
plt.figure(figsize=(6, 2)); plt.plot(user_wave.numpy()); plt.title("Generated waveform"); plt.show()
""")

md(r"""
### 11.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** "Turn X into an image, then use image-GAN machinery" is
  a broadly reusable trick beyond audio (e.g. time-series-as-image,
  graph-layouts-as-image) whenever X has a natural, roughly
  translation-invariant 2-D structure.
- ⚠️ **Warning.** Griffin-Lim phase reconstruction is the main fidelity
  bottleneck here, not the GAN -- it produces audible "phasiness"/metallic
  artifacts that a learned phase model (real GANSynth's instantaneous
  -frequency channel, or a neural vocoder like WaveGlow/HiFi-GAN) would
  remove.
- ⚠️ **Warning.** Our additive-synthesis training data has much less
  timbral realism than real recorded instruments (NSynth) -- pitch
  conditioning transfers directly to real data, but expect the model to
  need more capacity/data to capture real instrument timbre variation.

**This GAN solved the problem of: audio/music generation, conditioned on
a requested pitch, by reframing "generate a sound" as "generate a
spectrogram image" -- reusing conditional image-GAN machinery for a
non-image modality.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 12 -- GAMEGAN
# ---------------------------------------------------------------------------
md(r"""
<a id="12"></a>
## 12. GameGAN -- Game environment / next-frame simulation

### 12.1 Problem Statement & Why this GAN

Kim et al. (2020, NVIDIA) asked whether a neural network could **learn to
simulate a video game's engine purely by watching gameplay** -- given the
current frame and the player's action, predict the next frame, without
ever being given the game's actual rules/source code. This is a
fundamentally different problem from every earlier section: the generator
is not conditioned on noise-plus-a-label but on **the previous visual
state plus a discrete control input**, and must implicitly learn
game-specific dynamics (walls block movement, the agent moves in the
requested direction) purely from pixel-level supervision. Full GameGAN
additionally has a **memory module** (an external, addressable memory
bank for long-term spatial consistency -- "remember what's off-screen")
and separates a **dynamics engine** (predicts what changes) from a
**rendering engine** (turns that into pixels).

⚠️ **Warning -- Simplification.** Real GameGAN was trained on thousands of
hours of actual Pac-Man gameplay with a full external memory module. We
implement its **core mechanism** -- action-conditioned next-frame
prediction with a small recurrent "dynamics" state standing in for the
external memory -- on a **custom GridWorld** we can simulate instantly and
offline, which is why no pretrained-weight-loading subsection appears
here (no broadly available, license-clear pretrained GameGAN checkpoint
exists to load).

### 12.2 Data Collection & Preparation
""")

code(r"""
GAMEGAN_GRID_SIZE = 8
GAMEGAN_IMG_SIZE = 32
GAMEGAN_N_ACTIONS = 5
GAMEGAN_ACTION_NAMES = ["up", "down", "left", "right", "stay"]
GAMEGAN_BATCH_SIZE = 64
GAMEGAN_EPOCHS = 10

class GridWorld:
    '''A tiny, fully-deterministic "game": an agent (red square) moves on an
    8x8 grid with a fixed wall pattern (dark squares), rendered as a 32x32
    RGB image. This IS the "game engine" GameGAN has to learn implicitly
    from pixels alone -- we only use it here to generate training/ground
    -truth rollouts and to compare predictions against.'''
    ACTION_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}  # up, down, left, right, stay

    def __init__(self, grid_size=GAMEGAN_GRID_SIZE, img_size=GAMEGAN_IMG_SIZE, rng=None):
        self.grid_size, self.img_size = grid_size, img_size
        self.rng = rng or np.random.default_rng()
        self.walls = {(i, 4) for i in range(2, 6)}   # a fixed obstacle wall -- the "rule" the model must implicitly discover
        self.reset()

    def reset(self):
        while True:
            pos = (int(self.rng.integers(0, self.grid_size)), int(self.rng.integers(0, self.grid_size)))
            if pos not in self.walls:
                self.agent_pos = pos
                return self.render()

    def step(self, action: int):
        dx, dy = self.ACTION_DELTAS[action]
        x, y = self.agent_pos
        nx = int(np.clip(x + dx, 0, self.grid_size - 1))
        ny = int(np.clip(y + dy, 0, self.grid_size - 1))
        if (nx, ny) not in self.walls:   # walls block movement -- this is the "physics" GameGAN must learn purely from pixels
            self.agent_pos = (nx, ny)
        return self.render()

    def render(self) -> torch.Tensor:
        cell = self.img_size // self.grid_size
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32) * 0.9
        for wx, wy in self.walls:
            img[wy * cell:(wy + 1) * cell, wx * cell:(wx + 1) * cell] = [0.2, 0.2, 0.2]
        ax, ay = self.agent_pos
        img[ay * cell:(ay + 1) * cell, ax * cell:(ax + 1) * cell] = [0.9, 0.15, 0.15]
        return torch.from_numpy(img).permute(2, 0, 1) * 2 - 1

class GameGANDataset(Dataset):
    '''Random-action rollouts through GridWorld, cut into independent
    (frame_t, action_t, frame_t+1) TRAINING TRIPLES. Unlike CycleGAN
    (Section 8), this IS paired data -- the simulator gives us exact
    ground-truth next-frames, so we can (and do) use a direct pixel
    reconstruction loss alongside the adversarial one.'''
    def __init__(self, n_episodes=250, episode_len=16, seed=SEED):
        rng = np.random.default_rng(seed)
        self.triples = []
        for _ in range(n_episodes):
            env = GridWorld(rng=rng)
            frame = env.reset()
            for _ in range(episode_len):
                action = int(rng.integers(0, GAMEGAN_N_ACTIONS))
                next_frame = env.step(action)
                self.triples.append((frame, action, next_frame))
                frame = next_frame

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        return self.triples[i]

gamegan_dataset = GameGANDataset()
gamegan_loader = DataLoader(gamegan_dataset, batch_size=GAMEGAN_BATCH_SIZE, shuffle=True, drop_last=True)
frame_t, action_t, frame_t1 = next(iter(gamegan_loader))
print(f"GameGAN training triples: {len(gamegan_dataset)}, example actions: "
      f"{[GAMEGAN_ACTION_NAMES[a] for a in action_t[:5].tolist()]}")
show_images(torch.cat([frame_t[:8], frame_t1[:8]]), title="Real transitions: frame_t (top) -> frame_t+1 (bottom)", nrow=8)
""")

md(r"### 12.3 Model Architecture")

code(r"""
class GameGAN_Encoder(nn.Module):
    def __init__(self, nc=3, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1), nn.LeakyReLU(0.2, True),                              # 32 -> 16
            nn.Conv2d(nf, nf * 2, 4, 2, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),   # 16 -> 8
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),  # 8 -> 4
        )

    def forward(self, x):
        return self.net(x)


class GameGAN_Decoder(nn.Module):
    def __init__(self, nf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1), nn.BatchNorm2d(nf * 2), nn.ReLU(True),  # 4 -> 8
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1), nn.BatchNorm2d(nf), nn.ReLU(True),          # 8 -> 16
            nn.ConvTranspose2d(nf, nc, 4, 2, 1), nn.Tanh(),                                       # 16 -> 32
        )

    def forward(self, x):
        return self.net(x)


class GameGAN_Generator(nn.Module):
    '''Action-conditioned next-frame predictor with a GRUCell standing in
    for GameGAN's external memory module: `hidden` is threaded through
    calls across a rollout (see 12.6), so predictions can in principle
    depend on more history than just the single previous frame -- the
    minimal recurrent analogue of "remembering game state over time".'''
    def __init__(self, n_actions=GAMEGAN_N_ACTIONS, nf=64, action_embed_dim=16):
        super().__init__()
        self.nf = nf
        self.encoder = GameGAN_Encoder(nf=nf)
        self.action_embed = nn.Embedding(n_actions, action_embed_dim)
        bottleneck_dim = nf * 4 * 4 * 4
        self.hidden_dim = bottleneck_dim
        self.fuse = nn.Linear(bottleneck_dim + action_embed_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.to_bottleneck = nn.Linear(self.hidden_dim, bottleneck_dim)
        self.decoder = GameGAN_Decoder(nf=nf)

    def forward(self, frame, action, hidden=None):
        b = frame.size(0)
        feat = self.encoder(frame).view(b, -1)
        a_emb = self.action_embed(action)
        fused = torch.tanh(self.fuse(torch.cat([feat, a_emb], dim=1)))
        if hidden is None:
            hidden = torch.zeros(b, self.hidden_dim, device=frame.device)
        new_hidden = self.gru(fused, hidden)
        bottleneck = self.to_bottleneck(new_hidden).view(b, self.nf * 4, 4, 4)
        next_frame = self.decoder(bottleneck)
        return next_frame, new_hidden


class GameGAN_Discriminator(nn.Module):
    '''Conditional discriminator judging the TRIPLE (prev_frame, action,
    next_frame) jointly -- exactly the pix2pix/conditional-GAN pattern
    from Section 2, applied to "is this a physically plausible transition"
    instead of "is this the requested class".'''
    def __init__(self, n_actions=GAMEGAN_N_ACTIONS, nf=64, img_size=GAMEGAN_IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.action_embed = nn.Embedding(n_actions, img_size * img_size)
        self.net = nn.Sequential(
            nn.Conv2d(7, nf, 4, 2, 1), nn.LeakyReLU(0.2, True),                               # 3 (prev) + 3 (next) + 1 (action map)
            nn.Conv2d(nf, nf * 2, 4, 2, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, 1, 4, 1, 0),
        )

    def forward(self, prev_frame, action, next_frame):
        a_map = self.action_embed(action).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([prev_frame, next_frame, a_map], dim=1)
        return self.net(x).view(-1)


G_gamegan = GameGAN_Generator().to(DEVICE)
D_gamegan = GameGAN_Discriminator().to(DEVICE).apply(weights_init)
""")

md(r"### 12.4 Loss Functions & Optimizers")

code(r"""
gamegan_adv_criterion = nn.BCEWithLogitsLoss()
gamegan_recon_criterion = nn.L1Loss()
GAMEGAN_LAMBDA_RECON = 10.0   # since we HAVE ground truth next-frames (unlike CycleGAN), a strong direct pixel loss anchors predictions to the true game physics

gamegan_opt_G = torch.optim.Adam(G_gamegan.parameters(), lr=2e-4, betas=(0.5, 0.999))
gamegan_opt_D = torch.optim.Adam(D_gamegan.parameters(), lr=2e-4, betas=(0.5, 0.999))
""")

md(r"### 12.5 Training Loop")

code(r"""
gamegan_history = []
for epoch in range(GAMEGAN_EPOCHS):
    meters = {"G": AvgMeter(), "D": AvgMeter(), "L1": AvgMeter()}
    pbar = tqdm(gamegan_loader, desc=f"[GameGAN] epoch {epoch+1}/{GAMEGAN_EPOCHS}")
    for frame_t, action_t, frame_t1 in pbar:
        frame_t, action_t, frame_t1 = frame_t.to(DEVICE), action_t.to(DEVICE), frame_t1.to(DEVICE)
        b = frame_t.size(0)

        pred_t1, _ = G_gamegan(frame_t, action_t)   # single-step training: hidden starts fresh each batch (i.i.d. triples)

        # ---- Discriminator ----
        D_gamegan.zero_grad(set_to_none=True)
        logits_real = D_gamegan(frame_t, action_t, frame_t1)
        logits_fake = D_gamegan(frame_t, action_t, pred_t1.detach())
        loss_d = gamegan_adv_criterion(logits_real, torch.ones(b, device=DEVICE)) + \
                 gamegan_adv_criterion(logits_fake, torch.zeros(b, device=DEVICE))
        loss_d.backward()
        gamegan_opt_D.step()

        # ---- Generator: adversarial + strong pixel reconstruction ----
        G_gamegan.zero_grad(set_to_none=True)
        logits_fake_g = D_gamegan(frame_t, action_t, pred_t1)
        loss_g_adv = gamegan_adv_criterion(logits_fake_g, torch.ones(b, device=DEVICE))
        loss_g_recon = gamegan_recon_criterion(pred_t1, frame_t1)
        loss_g = loss_g_adv + GAMEGAN_LAMBDA_RECON * loss_g_recon
        loss_g.backward()
        gamegan_opt_G.step()

        meters["G"].update(loss_g.item()); meters["D"].update(loss_d.item()); meters["L1"].update(loss_g_recon.item())
        pbar.set_postfix(loss_G=f"{loss_g.item():.3f}", loss_D=f"{loss_d.item():.3f}", L1=f"{loss_g_recon.item():.3f}")

    gamegan_history.append((meters["G"].avg, meters["D"].avg, meters["L1"].avg))
    with torch.no_grad():
        G_gamegan.eval()
        pred_demo, _ = G_gamegan(frame_t[:8], action_t[:8])
        G_gamegan.train()
    show_images(torch.cat([frame_t1[:8], pred_demo]), title=f"GameGAN epoch {epoch+1}: real next-frame (top) vs predicted (bottom)", nrow=8)
    save_checkpoint({"G": G_gamegan.state_dict(), "D": D_gamegan.state_dict()}, f"{CHECKPOINT_DIR}/gamegan_epoch{epoch+1}.pt")

print("GameGAN training complete.")
""")

md(r"### 12.6 Inference / Generation Function")

code(r"""
@torch.no_grad()
def gamegan_rollout(start_frame: torch.Tensor, actions: list) -> list:
    '''The core product function: given a starting frame and a SEQUENCE of
    actions, autoregressively simulate the game -- each predicted frame
    becomes the next step's input, exactly like actually playing the
    (learned, implicit) game engine. `hidden` is threaded across steps.'''
    G_gamegan.eval()
    frame = start_frame.unsqueeze(0).to(DEVICE)
    hidden = None
    frames = [frame.squeeze(0).cpu()]
    for action in actions:
        action_t = torch.tensor([action], device=DEVICE)
        frame, hidden = G_gamegan(frame, action_t, hidden)
        frames.append(frame.squeeze(0).cpu())
    G_gamegan.train()
    return frames

demo_env = GridWorld(rng=np.random.default_rng(0))
start_frame = demo_env.reset()
demo_actions = [0, 0, 3, 3, 1]   # up, up, right, right, down
predicted_frames = gamegan_rollout(start_frame, demo_actions)
show_images(torch.stack(predicted_frames), title=f"GameGAN rollout: {[GAMEGAN_ACTION_NAMES[a] for a in demo_actions]}", nrow=len(predicted_frames))
""")

md(r"### 12.7 Visualization & Results")

code(r"""
gs, ds, l1s = zip(*gamegan_history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(gs, label="G"); axes[0].plot(ds, label="D"); axes[0].legend(); axes[0].set_title("Adversarial losses")
axes[1].plot(l1s); axes[1].set_title("Pixel L1 reconstruction loss")
plt.show()

# Ground-truth rollout vs GENERATED rollout for the SAME action sequence
# and the SAME starting frame -- directly visualizes prediction drift.
eval_env = GridWorld(rng=np.random.default_rng(1))
gt_start = eval_env.reset()
eval_actions = [3, 3, 3, 1, 1, 2, 2, 0]
gt_frames = [gt_start]
for a in eval_actions:
    gt_frames.append(eval_env.step(a))
pred_frames = gamegan_rollout(gt_start, eval_actions)

fig, axes = plt.subplots(2, len(gt_frames), figsize=(1.6 * len(gt_frames), 3.2))
for t in range(len(gt_frames)):
    axes[0, t].imshow(denorm(gt_frames[t]).permute(1, 2, 0).numpy()); axes[0, t].axis("off")
    axes[1, t].imshow(denorm(pred_frames[t]).permute(1, 2, 0).numpy()); axes[1, t].axis("off")
axes[0, 0].set_ylabel("Ground truth", fontsize=9)
axes[1, 0].set_ylabel("GameGAN", fontsize=9)
plt.suptitle("Ground-truth simulator (top) vs learned GameGAN rollout (bottom) -- watch drift accumulate")
plt.tight_layout(); plt.show()

# Quantitative drift: per-step L1 pixel error between ground truth and
# prediction -- error should trend UP with rollout length, illustrating
# the compounding-error problem inherent to autoregressive video prediction.
errors = [F.l1_loss(denorm(pred_frames[t]), denorm(gt_frames[t])).item() for t in range(len(gt_frames))]
plt.figure(figsize=(6, 3)); plt.plot(errors, marker="o")
plt.xlabel("rollout step"); plt.ylabel("L1 error vs ground truth"); plt.title("Compounding prediction drift over a rollout")
plt.show()
""")

md(r"### 12.8 User Input → Output Demo")

code(r"""
USER_ACTION_SEQUENCE = ["right", "right", "right", "down", "down"]
RANDOM_SEED = 5

user_env = GridWorld(rng=np.random.default_rng(RANDOM_SEED))
user_start = user_env.reset()
action_ids = [GAMEGAN_ACTION_NAMES.index(a) for a in USER_ACTION_SEQUENCE]
user_rollout = gamegan_rollout(user_start, action_ids)
show_images(torch.stack(user_rollout), title=f"User action sequence: {USER_ACTION_SEQUENCE}", nrow=len(user_rollout))
""")

md(r"""
### 12.9 Key Insights & Limitations of this GAN

- 💡 **Insight.** Framing "learn the game" as "predict the next frame
  conditioned on the previous frame and the action" turns environment
  simulation into a conditional-GAN problem almost identical in shape to
  Section 2's cGAN -- the interesting research content is entirely in
  *what* additional structure (memory, disentangled dynamics/rendering)
  makes that prediction remain accurate over long rollouts.
- ⚠️ **Warning.** The compounding-drift plot in 12.7 is the central
  limitation of ALL autoregressive video-prediction models, GameGAN
  included: small per-step errors compound, and after enough steps the
  simulated trajectory can diverge from anything the true environment
  would produce (e.g. "phasing through" a wall the model didn't perfectly
  learn is solid).
- ⚠️ **Warning.** Our GRUCell is a toy stand-in for GameGAN's actual
  external memory module (which explicitly reads/writes a spatial memory
  bank keyed by camera/agent position, enabling correct recall of
  off-screen content) -- do not expect this notebook's model to remember
  anything beyond very recent frames.

**This GAN solved the problem of: learning an implicit simulator of a
(simplified) game environment directly from observed (frame, action,
next-frame) transitions -- predicting future visual game state without
ever being given the game's actual rules.**
""")

\
# ---------------------------------------------------------------------------
# SECTION 13 -- UNIFIED INFERENCE HUB
# ---------------------------------------------------------------------------
md(r"""
<a id="13"></a>
## 13. Unified Inference Hub (Bonus)

Every section above defined its own clean `*_generate` / `*_translate` /
`*_upscale` / `*_rollout` function. This section wraps all twelve behind a
**single, consistent dictionary-based API**: `GAN_HUB[name]["run"](**kwargs)`.
This is the shape a real product backend would use -- one dispatcher, many
interchangeable model backends -- and is a natural place to plug in a
Streamlit/Gradio frontend (see Section 14).
""")

code(r"""
# Each entry: a human-readable description, the expected kwargs, and a
# `run(**kwargs)` callable that adapts that model's specific inference
# function to a uniform calling convention.
GAN_HUB = {
    "gan": {
        "description": "Original GAN -- unconditional MNIST-style digit generation",
        "kwargs": "n_samples:int, seed:int=None",
        "run": lambda n_samples=16, seed=None: gan_generate(n_samples, seed=seed),
    },
    "cgan": {
        "description": "cGAN -- class-conditional FashionMNIST generation",
        "kwargs": f"class_name:str in {CGAN_CLASS_NAMES}, n_samples:int",
        "run": lambda class_name, n_samples=8, seed=None: cgan_generate(CGAN_CLASS_NAMES.index(class_name), n_samples, seed=seed),
    },
    "acgan": {
        "description": "AC-GAN -- conditional CIFAR-10-subset generation + auxiliary classification",
        "kwargs": f"class_name:str in {ACGAN_CLASS_NAMES}, n_samples:int",
        "run": lambda class_name, n_samples=8, seed=None: acgan_generate(ACGAN_CLASS_NAMES.index(class_name), n_samples, seed=seed),
    },
    "progan": {
        "description": "Progressive GAN -- 32x32 CIFAR-10-style generation via progressive growing",
        "kwargs": "n_samples:int, seed:int=None",
        "run": lambda n_samples=16, seed=None: pgan_generate(n_samples, seed=seed),
    },
    "stylegan2": {
        "description": "StyleGAN2 -- controllable 64x64 face-style generation",
        "kwargs": "n_samples:int, truncation:float=0.7, seed:int=None",
        "run": lambda n_samples=16, truncation=0.7, seed=None: sgan2_generate(n_samples, truncation=truncation, seed=seed),
    },
    "biggan": {
        "description": "BigGAN -- large-scale class-conditional CIFAR-10 generation",
        "kwargs": f"class_name:str in {BIGGAN_CLASS_NAMES}, n_samples:int, truncation:float=0.5",
        "run": lambda class_name, n_samples=8, truncation=0.5, seed=None: biggan_generate(BIGGAN_CLASS_NAMES.index(class_name), n_samples, truncation=truncation, seed=seed),
    },
    "stackgan": {
        "description": "StackGAN -- two-stage text-to-image generation",
        "kwargs": "caption:str, n_samples:int",
        "run": lambda caption, n_samples=4, seed=None: stackgan_generate(caption, n_samples, seed=seed)[1],   # return only the final Stage-II image
    },
    "cyclegan": {
        "description": "CycleGAN -- unpaired A<->B image translation",
        "kwargs": "images:Tensor, direction:str in ['A2B','B2A']",
        "run": lambda images, direction="A2B": cyclegan_translate(images, direction),
    },
    "srgan": {
        "description": "SRGAN -- 4x single-image super-resolution",
        "kwargs": "lr_images:Tensor",
        "run": lambda lr_images: srgan_upscale(lr_images),
    },
    "esrgan": {
        "description": "ESRGAN -- enhanced 4x single-image super-resolution",
        "kwargs": "lr_images:Tensor",
        "run": lambda lr_images: esrgan_upscale(lr_images),
    },
    "gansynth": {
        "description": "GANSynth -- pitch-conditional audio note generation",
        "kwargs": "midi_pitch:int, seed:int=None",
        "run": lambda midi_pitch=60, seed=None: gansynth_generate(midi_pitch, seed=seed),
    },
    "gamegan": {
        "description": "GameGAN -- action-conditioned GridWorld next-frame rollout",
        "kwargs": "start_frame:Tensor, actions:list[int]",
        "run": lambda start_frame, actions: gamegan_rollout(start_frame, actions),
    },
}

def generate(model_name: str, **kwargs):
    '''The single entry point a frontend (CLI, Streamlit, Gradio, REST API)
    would call: generate("cgan", class_name="Sneaker", n_samples=8).'''
    if model_name not in GAN_HUB:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(GAN_HUB.keys())}")
    return GAN_HUB[model_name]["run"](**kwargs)

print("Unified GAN_HUB ready. Available models:")
for name, meta in GAN_HUB.items():
    print(f"  - {name:10s} | {meta['description']}")
    print(f"    {' ' * 12}kwargs: {meta['kwargs']}")
""")

code(r"""
# A few example calls through the unified API, exercising several
# different modalities (image / conditional image / text-to-image) through
# the exact same `generate(...)` function signature shape.
show_images(generate("gan", n_samples=8, seed=1), title="Hub: gan", nrow=4)
show_images(generate("cgan", class_name="Bag", n_samples=8, seed=1), title="Hub: cgan(class_name='Bag')", nrow=4)
_, hub_stackgan_img = stackgan_generate("a photo of a deer", n_samples=4, seed=1)
show_images(hub_stackgan_img, title="Hub: stackgan(caption='a photo of a deer')", nrow=4)
""")

\
# ---------------------------------------------------------------------------
# SECTION 14 -- FINAL NOTES & RECOMMENDATIONS
# ---------------------------------------------------------------------------
md(r"""
<a id="14"></a>
## 14. Final Notes & Recommendations

### Which models to train from scratch vs. load pretrained

| # | Model | Train from scratch when... | Load pretrained when... |
|---|-------|------------------------------|---------------------------|
| 1 | GAN | Always -- it's small, fast, and the point is to see the mechanism | -- |
| 2 | cGAN | Always -- same as above | -- |
| 3 | AC-GAN | Always -- same as above | -- |
| 4 | Progressive GAN | Learning/demonstrating the growth mechanism | You need real high-res (512-1024px) faces -- use `pytorch_GAN_zoo` |
| 5 | StyleGAN2 | Learning style-modulation/mixing mechanics | You need photorealistic faces -- use official NVlabs checkpoints |
| 6 | BigGAN | Learning attention/spectral-norm/projection-D mechanics | You need diverse, high-fidelity ImageNet-class images -- use `pytorch_pretrained_biggan` |
| 7 | StackGAN | Always for now -- no broadly available pretrained checkpoint at this scale | -- |
| 8 | CycleGAN | Your own custom unpaired domains | Standard pairs (horse2zebra, etc.) -- many pretrained checkpoints exist |
| 9 | SRGAN | Learning the perceptual-loss mechanism | -- |
| 10 | ESRGAN | Learning the RRDB/relativistic-loss mechanism | Real photo restoration -- official Real-ESRGAN checkpoints are excellent and widely available |
| 11 | GANSynth | Always -- our synthetic-audio setup is inherently a from-scratch demo | -- |
| 12 | GameGAN | Always -- no broadly available pretrained checkpoint exists | -- |

### Approximate GPU-hours for a *real* (not lightweight-demo) training run, on a single modern GPU

- **Cheap (< 1 GPU-hour):** GAN, cGAN, AC-GAN, GANSynth, GameGAN (small custom envs), SRGAN/ESRGAN on a small crop set.
- **Moderate (a few GPU-hours to ~1 GPU-day):** Progressive GAN to 256px, CycleGAN on a full horse2zebra-sized dataset, StackGAN on CUB-200.
- **Expensive (many GPU-days to GPU-weeks):** StyleGAN2 to 1024px on FFHQ, BigGAN on full ImageNet at 256-512px, ESRGAN/SRGAN on full DIV2K at high resolution with long schedules.

⚠️ **Warning.** These are order-of-magnitude guides, not benchmarks --
actual cost depends heavily on batch size, resolution, and how many
samples you need before FID/visual quality plateaus.

### How to extend this notebook into a Streamlit / Gradio app

The [Unified Inference Hub](#13) (`GAN_HUB` / `generate(...)`) was built
specifically to make this step mechanical:

1. **Persist trained weights.** Every training loop above already calls
   `save_checkpoint(...)` each epoch -- point your app at the final
   checkpoint files in `checkpoints/` instead of retraining at app
   startup.
2. **Move model construction + `load_checkpoint` calls into a single
   `load_all_models()` function**, called once at app startup (not per
   request) -- GAN inference is cheap, but re-building/re-loading a dozen
   networks per request is not.
3. **Wrap `generate(model_name, **kwargs)` directly as your backend
   call.** A minimal Gradio app is genuinely this short:
   ```python
   import gradio as gr

   def run_cgan(class_name, n_samples):
       imgs = generate("cgan", class_name=class_name, n_samples=n_samples)
       return [T.ToPILImage()(denorm(img)) for img in imgs]

   gr.Interface(
       fn=run_cgan,
       inputs=[gr.Dropdown(CGAN_CLASS_NAMES), gr.Slider(1, 16, value=8, step=1)],
       outputs=gr.Gallery(),
   ).launch()
   ```
   Repeat per-model (a `gr.Tab` per `GAN_HUB` entry) for a full 12-model
   demo app, or a Streamlit `st.selectbox` over `GAN_HUB.keys()` for the
   Streamlit equivalent.
4. **For audio (GANSynth) and video-like rollouts (GameGAN),** use
   `gr.Audio` / `gr.Gallery(...).style(columns=...)` respectively --
   `spec_db_to_wave(...)` and `gamegan_rollout(...)` already return
   exactly the tensors those components expect.
5. **Deploy** the Gradio app directly via Hugging Face Spaces (zero extra
   infra) or the Streamlit app via Streamlit Community Cloud -- both
   accept a `requirements.txt` (torch, torchvision, torchaudio,
   gradio/streamlit) and the app script with no further changes needed.

### Closing summary

Twelve GANs, twelve genuinely different problems: unconditional
generation, class-conditional generation, conditional generation with
auxiliary classification, stable high-resolution growth, disentangled
style control, large-scale class-conditional fidelity, text-to-image,
unpaired translation, super-resolution (two ways), audio synthesis, and
learned game simulation. The adversarial min-max game from Section 1
never changes -- everything else in this notebook is a variation on *what*
the generator is conditioned on, *what* representation it outputs, and
*what* structural prior (attention, style modulation, dense connections,
recurrence) makes that specific problem tractable.
""")

# ===APPEND_MARKER===

def build():
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
            "colab": {
                "name": "GANForge_Master_Notebook.ipynb",
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open("GANForge_Master_Notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote notebook with {len(cells)} cells")


if __name__ == "__main__":
    build()
