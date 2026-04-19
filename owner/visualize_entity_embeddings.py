"""Visualize entity embeddings (CrossNER + saved EntityEncodingModel checkpoint).

Example (from repo root):
  python -m owner.visualize_entity_embeddings \\
    --checkpoint checkpoints/owner/conll2003/new_model/entity_typing.pt \\
    --output outputs/soft_prompt.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from owner.data.datasets.entity_typing import EntityTypingDataset
from owner.data.serialization import parse_owner_dataset
from owner.models.entity_typing import EntityEncodingModel
from owner.models.soft_prompt_entity_encoding import SoftPromptEntityEncodingModel
from owner.utils.pytorch import get_num_workers


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(repo: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    return (repo / p).resolve()


def _collect_embeddings(
    *,
    dataset_json: Path,
    checkpoint: Path,
    plm_name: str,
    template: str,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    owner_ds = parse_owner_dataset(str(dataset_json))
    et_ds = EntityTypingDataset(owner_ds, plm_name, max_len, template)
    loader = DataLoader(
        et_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_num_workers(),
    )
    model = EntityEncodingModel(plm_name)
    model = SoftPromptEntityEncodingModel(plm_name)
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    out_vecs: list[np.ndarray] = []
    out_types: list[str] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_index = batch["mask_index"].to(device)
            labels = batch["entity_type_label"]

            emb = model(input_ids, attention_mask, mask_index)
            out_vecs.append(emb.cpu().numpy())
            for lid in labels.tolist():
                out_types.append(str(et_ds.id_to_entity_type[int(lid)]))

    return np.concatenate(out_vecs, axis=0), out_types


def _subsample(
    emb: np.ndarray,
    types: list[str],
    domains: list[str],
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    n = emb.shape[0]
    if max_samples is None or n <= max_samples:
        return emb, types, domains
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return emb[idx], [types[i] for i in idx], [domains[i] for i in idx]


def _types_to_colors(types: list[str], seed: int) -> np.ndarray:
    uniq = sorted(set(types))
    rng = np.random.RandomState(seed)
    palette = rng.uniform(0.15, 0.95, size=(len(uniq), 3))
    t2i = {t: i for i, t in enumerate(uniq)}
    return palette[np.array([t2i[t] for t in types], dtype=np.int64)]


def main(argv: list[str] | None = None) -> int:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description="t-SNE plot of entity embeddings (CrossNER).")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/crossner"),
        help="Root folder with ai/, literature/, ... (relative to repo if not absolute).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["ai", "literature", "music", "science", "politics"],
        help="CrossNER subfolders under data-root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test.json",
        help="Filename inside each domain folder (e.g. test.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/owner/conll2003/new_model/entity_typing.pt"),
        help="Path to entity_typing.pt (encoder state dict).",
    )
    parser.add_argument("--plm-name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--template",
        type=str,
        default="{sentence} {entity} is a [MASK].",
    )
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=2500,
        help="Cap entities per domain before pooling. Use -1 for no cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/crossner_entity_tsne.png"),
        help="PNG path (relative to repo if not absolute).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    data_root = _resolve_path(repo, args.data_root)
    checkpoint = _resolve_path(repo, args.checkpoint)
    output_path = _resolve_path(repo, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    cap = None if args.max_samples_per_domain < 0 else args.max_samples_per_domain

    ncols = min(3, len(args.domains))
    nrows = (len(args.domains) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for ax_i, domain in enumerate(args.domains):
        json_path = data_root / domain / args.split
        if not json_path.is_file():
            print(f"Missing dataset: {json_path}", file=sys.stderr)
            return 1

        emb, types = _collect_embeddings(
            dataset_json=json_path,
            checkpoint=checkpoint,
            plm_name=args.plm_name,
            template=args.template,
            max_len=args.max_len,
            batch_size=args.batch_size,
            device=device,
        )
        domains = [domain] * len(types)
        emb, types, _ = _subsample(emb, types, domains, cap, args.seed)

        n = emb.shape[0]
        r, c_ = divmod(ax_i, ncols)
        ax = axes[r][c_]
        if n < 4:
            ax.text(
                0.5,
                0.5,
                f"Too few points for t-SNE (n={n})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        perplexity = min(30.0, float(n - 1) * 0.9)
        perplexity = max(2.0, perplexity)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=args.seed,
        )
        z = tsne.fit_transform(emb)
        colors = _types_to_colors(types, seed=args.seed)

        ax.scatter(z[:, 0], z[:, 1], c=colors, s=6, alpha=0.75, linewidths=0)
        ax.set_title(f"{domain} (n={n}, types={len(set(types))})")
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(len(args.domains), nrows * ncols):
        r, c_ = divmod(k, ncols)
        axes[r][c_].axis("off")

    fig.suptitle(
        f"Entity embeddings (t-SNE) — {args.plm_name}\n{checkpoint.name}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



