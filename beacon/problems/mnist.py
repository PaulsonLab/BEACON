"""MNIST digit discovery (paper Sec. 5.5).

Candidates are 2,000 random points of an 8-dimensional VAE latent space
(scaled to ``[-1, 1]^8``); each decodes to a 14 x 14 image, the 196-pixel
outcome. A pretrained CNN classifies images; a digit class counts as reached
when an image is classified with probability >= 0.99. Reachability is the
fraction of the 10 classes reached (the paper's figure plots the count).

BEACON models the image with a higher-order GP (HOGP). UG-BEACON uses the
digit classes (classifier probability >= 0.5) as guidance regions.

Only the bundled weights in ``models/mnist`` are required (no dataset
download).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from beacon.device import DTYPE
from beacon.metrics import ClassifierPartition, Reachability
from beacon.paths import MNIST_MODELS_DIR
from beacon.problems.base import DiscreteProblem

LATENT_DIM = 8
IMAGE_SHAPE = (14, 14)


class CNN(nn.Module):
    """Digit classifier for 14 x 14 images (architecture of ``CNN2.pth``)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.out = nn.Linear(288, 10)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(x.size(0), -1)
        return F.softmax(self.out(x), dim=1), x


class VAE(nn.Module):
    """VAE for 14 x 14 digits with an 8-dimensional latent space (``VAE1.pth``)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(196, 128)
        self.fc21 = nn.Linear(128, LATENT_DIM)
        self.fc22 = nn.Linear(128, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, 128)
        self.fc4 = nn.Linear(128, 196)

    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def decoder(self, z):
        return torch.sigmoid(self.fc4(torch.tanh(self.fc3(z))))


def _load(module: nn.Module, filename: str) -> nn.Module:
    path = MNIST_MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained weights {path}.")
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return module.eval()


def make_mnist(n_candidates: int = 2000, candidate_seed: int = 0, threshold: float = 0.99, guidance_threshold: float = 0.5, **_):
    vae = _load(VAE(), "VAE1.pth")
    cnn = _load(CNN(), "CNN2.pth")

    generator = torch.Generator().manual_seed(candidate_seed)
    X = torch.rand(n_candidates, LATENT_DIM, generator=generator)
    with torch.no_grad():
        Y = vae.decoder(-1 + 2 * X).reshape(n_candidates, -1)

    def classify(images: torch.Tensor):
        with torch.no_grad():
            probs, _ = cnn(images.detach().float().cpu().reshape(-1, 1, *IMAGE_SHAPE))
        confidence, labels = probs.max(dim=1)
        return labels, confidence

    partition = ClassifierPartition(classify, 10, threshold=threshold)
    problem = DiscreteProblem(
        "mnist",
        X.to(DTYPE),
        Y.to(DTYPE),
        metric=Reachability(partition),
        partition=partition,
        output_shape=IMAGE_SHAPE,
        defaults=dict(
            n_init=10,
            iterations=40,
            replicates=10,
            k=10,
            surrogate="hogp",
            archive="observed",
            discrete_sampler="exact",
        ),
        description="MNIST digits decoded from an 8D VAE latent space; 196-pixel outcomes, 10 digit classes.",
    )
    problem.guidance_partition = ClassifierPartition(classify, 10, threshold=guidance_threshold)
    return problem
