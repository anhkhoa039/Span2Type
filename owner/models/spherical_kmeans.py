"""Spherical K-Means clustering utilities.

Spherical k-means is k-means on the unit hypersphere:
- L2-normalize all input vectors.
- Assign points to the centroid with maximum cosine similarity (i.e., smallest angle).
- Update centroids by averaging assigned points and re-normalizing to unit length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


@dataclass
class SphericalKMeansResult:
    centroids: torch.Tensor  # [k, dim] unit-norm
    labels: torch.Tensor  # [n] long
    inertia: float  # sum(1 - max_cosine) over samples
    n_iter: int


class SphericalKMeans:
    """A minimal spherical k-means implementation for embedding clustering."""

    def __init__(
        self,
        n_clusters: int,
        *,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: Optional[int] = None,
        eps: float = 1e-12,
    ):
        self.n_clusters = int(n_clusters)
        self.init = str(init)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = seed
        self.eps = float(eps)

        self.cluster_centers_: Optional[torch.Tensor] = None
        self.labels_: Optional[torch.Tensor] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def _make_generator(self, device: torch.device, offset: int = 0) -> torch.Generator:
        g = torch.Generator(device=device)
        if self.seed is not None:
            g.manual_seed(int(self.seed) + int(offset))
        return g

    def _init_centroids_random(self, x_unit: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        n = x_unit.shape[0]
        perm = torch.randperm(n, generator=g, device=x_unit.device)
        centroids = x_unit[perm[: self.n_clusters]].clone()
        return _l2_normalize(centroids, eps=self.eps)

    def _init_centroids_kmeans_pp(self, x_unit: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """Spherical k-means++ initialization on unit vectors.

        Uses cosine distance d(x,c) = 1 - cos(x,c) and samples next centroid with prob ∝ d^2.
        """
        n, d = x_unit.shape
        device = x_unit.device

        # Pick first center uniformly at random.
        first_idx = torch.randint(0, n, (1,), generator=g, device=device).item()
        centers = torch.empty((self.n_clusters, d), device=device, dtype=x_unit.dtype)
        centers[0] = x_unit[first_idx]

        # Track closest distance to any chosen center so far.
        closest_sim = (x_unit @ centers[0].view(-1, 1)).squeeze(1)  # [n]
        closest_dist = (1.0 - closest_sim).clamp_min(0.0)  # [n]

        for i in range(1, self.n_clusters):
            probs = closest_dist.pow(2)
            s = probs.sum()
            if float(s.item()) <= 0.0:
                # All points identical to chosen centers; fall back to random.
                idx = torch.randint(0, n, (1,), generator=g, device=device).item()
            else:
                probs = probs / s
                idx = int(torch.multinomial(probs, num_samples=1, replacement=False, generator=g).item())
            centers[i] = x_unit[idx]

            sims_i = (x_unit @ centers[i].view(-1, 1)).squeeze(1)
            closest_sim = torch.maximum(closest_sim, sims_i)
            closest_dist = (1.0 - closest_sim).clamp_min(0.0)

        return _l2_normalize(centers, eps=self.eps)

    def _init_centroids(self, x_unit: torch.Tensor, *, g: torch.Generator) -> torch.Tensor:
        n = x_unit.shape[0]
        if self.n_clusters > n:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be > n_samples ({n})."
            )
        if self.init == "random":
            return self._init_centroids_random(x_unit, g)
        if self.init in {"k-means++", "kmeans++"}:
            return self._init_centroids_kmeans_pp(x_unit, g)
        raise ValueError('init must be "k-means++" or "random".')

    def _run_once(self, x_unit: torch.Tensor, *, g: torch.Generator) -> SphericalKMeansResult:
        centroids = self._init_centroids(x_unit, g=g)

        labels = torch.zeros((x_unit.shape[0],), dtype=torch.long, device=x_unit.device)
        n_iter = 0

        for it in range(1, self.max_iter + 1):
            n_iter = it
            sims = x_unit @ centroids.T  # [n, k]
            labels = sims.argmax(dim=1)  # [n]

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = x_unit[mask].mean(dim=0)
                else:
                    # Empty cluster: pick the point currently farthest (smallest max sim)
                    # from all centroids to encourage coverage.
                    max_sims = sims.max(dim=1).values
                    farthest_idx = int(max_sims.argmin().item())
                    new_centroids[k] = x_unit[farthest_idx]

            new_centroids = _l2_normalize(new_centroids, eps=self.eps)

            shift = (new_centroids - centroids).norm(p=2, dim=1).max().item()
            centroids = new_centroids
            if shift < self.tol:
                break

        sims = x_unit @ centroids.T
        labels = sims.argmax(dim=1)
        max_sims = sims.max(dim=1).values
        inertia = float((1.0 - max_sims).clamp_min(0.0).sum().item())

        return SphericalKMeansResult(
            centroids=centroids,
            labels=labels.to(dtype=torch.long),
            inertia=inertia,
            n_iter=n_iter,
        )

    def fit(self, x: torch.Tensor) -> "SphericalKMeans":
        """Fit spherical k-means.

        Args:
            x: [n, dim] embedding tensor (float). Will be moved to CPU.
        """
        x = x.detach().to(device="cpu")
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D [n, dim], got shape {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("Expected at least 1 sample.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1.")

        x_unit = _l2_normalize(x, eps=self.eps)

        best: Optional[SphericalKMeansResult] = None
        for run in range(self.n_init):
            g = self._make_generator(device=x_unit.device, offset=run)
            res = self._run_once(x_unit, g=g)
            if best is None or res.inertia < best.inertia:
                best = res

        if best is None:
            raise RuntimeError("Unexpected: no runs executed.")

        self.cluster_centers_ = best.centroids
        self.labels_ = best.labels
        self.inertia_ = best.inertia
        self.n_iter_ = best.n_iter
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for x.

        Args:
            x: [n, dim] embedding tensor.
        Returns:
            [n] long tensor of predicted cluster ids.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")

        x = x.detach().to(device="cpu")
        x_unit = _l2_normalize(x, eps=self.eps)
        centroids = self.cluster_centers_
        sims = x_unit @ centroids.T
        return sims.argmax(dim=1).to(dtype=torch.long)


def bic_spherical_kmeans(inertia: float, n_samples: int, n_clusters: int) -> float:
    """A simple BIC-like criterion analogous to the Euclidean k-means variant used in repo.

    This mirrors the structure in `owner/models/entity_typing.py`:
        BIC = n * log(inertia/n) + log(n) * k

    Here inertia is the spherical objective sum(1 - cos), lower is better.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0.")

    # Guard against log(0) when inertia is extremely small.
    inertia_per = max(inertia / n_samples, 1e-12)
    n = float(n_samples)
    k = float(n_clusters)
    return n * torch.log(torch.tensor(inertia_per)).item() + torch.log(torch.tensor(n)).item() * k


class AutoSphericalKMeans:
    """Spherical k-means that estimates k by a BIC-like score."""

    def __init__(
        self,
        k_min: int,
        k_max: int,
        k_step: int = 1,
        *,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: Optional[int] = None,
    ):
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_step = int(k_step)
        self.init = str(init)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = seed

        self._best: Optional[SphericalKMeans] = None
        self._scores: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (ks, bics)

    @property
    def k(self) -> int:
        if self._best is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")
        return int(self._best.n_clusters)

    def get_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return tried ks and corresponding BIC-like scores (lower is better)."""
        if self._scores is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")
        return self._scores

    def fit(self, entity_embeddings: torch.Tensor) -> "AutoSphericalKMeans":
        x = entity_embeddings.detach().to(device="cpu")
        n = x.shape[0]
        ks = list(range(self.k_min, self.k_max + 1, self.k_step))
        if len(ks) == 0:
            raise ValueError("Empty k range. Check k_min/k_max/k_step.")

        bics = []
        best_bic = None
        best_model = None

        for k in ks:
            model = SphericalKMeans(
                n_clusters=k,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                seed=self.seed,
            ).fit(x)
            bic = bic_spherical_kmeans(model.inertia_, n_samples=n, n_clusters=k)
            bics.append(bic)
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_model = model

        self._best = best_model
        self._scores = (torch.tensor(ks, dtype=torch.long), torch.tensor(bics, dtype=torch.float))
        return self

    def predict(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        if self._best is None:
            raise RuntimeError("Model is not fit yet. Call fit() first.")
        return self._best.predict(entity_embeddings)

