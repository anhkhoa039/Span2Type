"""Cluster naming with MMR exemplar selection + MLM/Ollama backends."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency
    ollama = None

_BASIC_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "we", "you", "i", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "as",
    "by", "at", "from", "and", "or", "but", "if", "then", "than", "so",
    "pun", "etc",
}


def _is_good_label_token(token: str) -> bool:
    """Heuristic token filter to avoid degenerate MLM labels."""
    if token.startswith("##"):
        return False
    if token in {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}:
        return False
    if not re.fullmatch(r"[a-z]+", token):
        return False
    if len(token) < 3:
        return False
    if token in _BASIC_STOPWORDS:
        return False
    return True


def _clean_ollama_label(raw_text: str, max_words: int = 3) -> str:
    """Strip common filler and keep a short human-readable label."""
    text = raw_text.strip()
    text = re.sub(
        r"^(the\s+category\s+is|category|label|answer)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.splitlines()[0].strip().strip("`\"' ")
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    text = re.sub(r"[^A-Za-z0-9\-\s]", "", text).strip()
    words = text.split()
    if not words:
        return "unknown"
    return " ".join(words[:max_words]).lower()


def select_diverse_exemplars_mmr(
    centroid: torch.Tensor,
    entity_embeddings: torch.Tensor,
    sentences: Sequence[str],
    k: int = 5,
    mmr_lambda: float = 0.7,
) -> List[str]:
    """Select k diverse and centroid-relevant exemplar sentences with MMR."""
    if entity_embeddings.ndim != 2:
        raise ValueError("entity_embeddings must have shape [N, D].")
    if centroid.ndim != 1:
        raise ValueError("centroid must have shape [D].")
    if entity_embeddings.shape[0] != len(sentences):
        raise ValueError("entity_embeddings and sentences must have same first dimension.")
    if not sentences:
        return []

    num_items = entity_embeddings.shape[0]
    k = min(k, num_items)

    emb_norm = F.normalize(entity_embeddings, p=2, dim=-1)
    centroid_norm = F.normalize(centroid.view(1, -1), p=2, dim=-1)
    relevance = (emb_norm @ centroid_norm.T).squeeze(1)  # [N]
    pairwise_sim = emb_norm @ emb_norm.T  # [N, N]

    selected: List[int] = []
    remaining = set(range(num_items))

    first_idx = int(torch.argmax(relevance).item())
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < k and remaining:
        rem_idx = torch.tensor(list(remaining), device=entity_embeddings.device)
        sel_idx = torch.tensor(selected, device=entity_embeddings.device)

        rel_score = relevance[rem_idx]  # [R]
        redundancy = pairwise_sim[rem_idx][:, sel_idx].max(dim=1).values  # [R]
        mmr_score = mmr_lambda * rel_score - (1.0 - mmr_lambda) * redundancy

        best_local = int(torch.argmax(mmr_score).item())
        best_global = int(rem_idx[best_local].item())
        selected.append(best_global)
        remaining.remove(best_global)

    return [sentences[i] for i in selected]


def _build_cluster_exemplar_prompts(
    cluster_emb: torch.Tensor,
    cluster_sentence_texts: Sequence[str],
    cluster_entity_texts: Sequence[str],
    num_exemplars: int,
    mmr_lambda: float,
    naming_templates: Sequence[str],
    mask_token: str = "[MASK]",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Select MMR exemplars (1:1 with embeddings), then expand with templates."""
    centroid = cluster_emb.mean(dim=0)
    # Keep 1:1 cardinality with embeddings while selecting diverse examples.
    base_example_texts = [
        f"{entity} || {sentence}"
        for sentence, entity in zip(cluster_sentence_texts, cluster_entity_texts)
    ]
    exemplar_base_texts = select_diverse_exemplars_mmr(
        centroid=centroid,
        entity_embeddings=cluster_emb,
        sentences=base_example_texts,
        k=num_exemplars,
        mmr_lambda=mmr_lambda,
    )
    exemplar_pairs: List[tuple[str, str]] = []
    for item in exemplar_base_texts:
        if " || " in item:
            entity, sentence = item.split(" || ", 1)
        else:
            entity, sentence = "", item
        exemplar_pairs.append((entity, sentence))

    exemplar_prompts = [
        template.format(sentence=sentence, entity=entity).replace("[MASK]", mask_token)
        for entity, sentence in exemplar_pairs
        for template in naming_templates
    ]
    return exemplar_prompts, exemplar_pairs


@torch.no_grad()
def generate_cluster_name_map_with_mlm(
    entity_embeddings: torch.Tensor,
    cluster_ids: Sequence[int],
    sentence_texts: Sequence[str],
    entity_texts: Sequence[str],
    plm_name: str,
    num_exemplars: int = 16,
    mmr_lambda: float = 0.7,
    naming_template: str = "{sentence} {entity} is a [MASK].",
    naming_templates: Sequence[str] | None = None,
    use_mmr: bool = True,
    seed: int | None = None,
    save_path: str | None = None,
) -> Dict[int, Dict[str, Any]]:
    """Generate one label token per cluster via MLM consensus with exemplar logging.

    Uses the OWNER paper prompt (Eq. 2): '{sentence} {entity} is a [MASK].'
    use_mmr=True selects diverse exemplars via MMR; False uses random sampling.
    Results are saved to save_path as JSON if provided.
    """
    if entity_embeddings.ndim != 2:
        raise ValueError("entity_embeddings must have shape [N, D].")
    if len(cluster_ids) != entity_embeddings.shape[0]:
        raise ValueError("cluster_ids length must match number of embeddings.")
    if len(sentence_texts) != entity_embeddings.shape[0]:
        raise ValueError("sentence_texts length must match number of embeddings.")
    if len(entity_texts) != entity_embeddings.shape[0]:
        raise ValueError("entity_texts length must match number of embeddings.")

    if entity_embeddings.shape[0] == 0:
        return {}

    device = entity_embeddings.device
    tokenizer = AutoTokenizer.from_pretrained(plm_name)
    model = AutoModelForMaskedLM.from_pretrained(plm_name).to(device).eval()
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id
    if mask_token is None or mask_token_id is None:
        raise ValueError(f"Tokenizer {plm_name} does not provide a mask token.")

    rng = random.Random(seed)
    cluster_tensor = torch.tensor(cluster_ids, device=device, dtype=torch.long)
    unique_clusters = torch.unique(cluster_tensor).tolist()
    name_map: Dict[int, Dict[str, Any]] = {}

    templates = list(naming_templates) if naming_templates else [naming_template]

    for cluster_id in unique_clusters:
        selector = cluster_tensor == int(cluster_id)
        cluster_emb = entity_embeddings[selector]
        cluster_sentence_texts = [
            s for s, keep in zip(sentence_texts, selector.tolist()) if keep
        ]
        cluster_entity_texts = [
            e for e, keep in zip(entity_texts, selector.tolist()) if keep
        ]
        if not cluster_sentence_texts:
            name_map[int(cluster_id)] = {
                "name": f"cluster_{int(cluster_id)}",
                "examples": [],
                "top_candidates": [],
            }
            continue

        if use_mmr:
            exemplars, exemplar_pairs = _build_cluster_exemplar_prompts(
                cluster_emb=cluster_emb,
                cluster_sentence_texts=cluster_sentence_texts,
                cluster_entity_texts=cluster_entity_texts,
                num_exemplars=num_exemplars,
                mmr_lambda=mmr_lambda,
                naming_templates=templates,
                mask_token=mask_token,
            )
        else:
            k = min(num_exemplars, len(cluster_sentence_texts))
            indices = rng.sample(range(len(cluster_sentence_texts)), k)
            exemplar_pairs = [
                (cluster_entity_texts[i], cluster_sentence_texts[i]) for i in indices
            ]
            exemplars = [
                template.format(sentence=sentence, entity=entity).replace("[MASK]", mask_token)
                for entity, sentence in exemplar_pairs
                for template in templates
            ]
        if not exemplars:
            name_map[int(cluster_id)] = {
                "name": f"cluster_{int(cluster_id)}",
                "examples": [],
                "top_candidates": [],
            }
            continue

        encoded = tokenizer(
            exemplars,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
        if mask_positions.shape[0] != input_ids.shape[0]:
            name_map[int(cluster_id)] = {
                "name": f"cluster_{int(cluster_id)}",
                "examples": [
                    template.format(sentence=sentence, entity=entity).replace("[MASK]", mask_token)
                    for entity, sentence in exemplar_pairs
                    for template in templates
                ],
                "top_candidates": [],
            }
            continue

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)

        batch_idx = mask_positions[:, 0]
        tok_idx = mask_positions[:, 1]
        mask_probs = probs[batch_idx, tok_idx, :]  # [B, V]

        # Each exemplar votes for its top valid predicted token (most frequent wins).
        vote_counts: Dict[str, int] = {}
        for i in range(mask_probs.shape[0]):
            for tok_id in torch.argsort(mask_probs[i], descending=True).tolist():
                tok = tokenizer.convert_ids_to_tokens(int(tok_id))
                if _is_good_label_token(tok):
                    vote_counts[tok] = vote_counts.get(tok, 0) + 1
                    break

        top_candidates: List[Dict[str, int | str]] = sorted(
            [{"token": tok, "votes": count} for tok, count in vote_counts.items()],
            key=lambda x: x["votes"],
            reverse=True,
        )[:5]

        if top_candidates:
            token = str(top_candidates[0]["token"])
        else:
            # Fallback: best raw token from first exemplar.
            for tok_id in torch.argsort(mask_probs[0], descending=True).tolist():
                tok = tokenizer.convert_ids_to_tokens(int(tok_id))
                if not tok.startswith("##"):
                    token = tok
                    break
        name_map[int(cluster_id)] = {
            "name": token,
            "examples": [
                template.format(sentence=sentence, entity=entity).replace("[MASK]", mask_token)
                for entity, sentence in exemplar_pairs
                for template in templates
            ],
            "top_candidates": top_candidates,
        }

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in name_map.items()}, f, indent=2, ensure_ascii=False)

    return name_map


def generate_cluster_name_map_with_ollama(
    entity_embeddings: torch.Tensor,
    cluster_ids: Sequence[int],
    sentence_texts: Sequence[str],
    entity_texts: Sequence[str],
    model_name: str = "llama3",
    n_samples: int = 16,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
    seed: int | None = None,
    max_words: int = 3,
    save_path: str | None = None,
) -> Dict[int, Dict[str, Any]]:
    """Generate one human-readable label per cluster via local Ollama LLM.

    Uses the OWNER paper prompt (Figure 3), temperature=0, n_samples=16:
      System: "A virtual assistant answers questions from a user based on the provided text."
      User: "Given the list of entities {e_1, ..., e_n}, which all belong to the same type,
             predict the entity type corresponding to all these entities.
             Respond only with the name of the entity type."
    use_mmr=True selects diverse entities via MMR instead of random sampling.
    Results are saved to save_path as JSON if provided.
    """
    if ollama is None:
        raise ImportError(
            "ollama package not installed. Install with `pip install ollama`."
        )
    if entity_embeddings.ndim != 2:
        raise ValueError("entity_embeddings must have shape [N, D].")
    if len(cluster_ids) != entity_embeddings.shape[0]:
        raise ValueError("cluster_ids length must match number of embeddings.")
    if len(sentence_texts) != entity_embeddings.shape[0]:
        raise ValueError("sentence_texts length must match number of embeddings.")
    if len(entity_texts) != entity_embeddings.shape[0]:
        raise ValueError("entity_texts length must match number of embeddings.")
    if entity_embeddings.shape[0] == 0:
        return {}

    rng = random.Random(seed)
    cluster_tensor = torch.tensor(cluster_ids, device=entity_embeddings.device, dtype=torch.long)
    unique_clusters = torch.unique(cluster_tensor).tolist()
    name_map: Dict[int, Dict[str, Any]] = {}

    for cluster_id in unique_clusters:
        selector = cluster_tensor == int(cluster_id)
        cluster_emb = entity_embeddings[selector]
        cluster_sentence_texts = [s for s, keep in zip(sentence_texts, selector.tolist()) if keep]
        cluster_entity_texts = [e for e, keep in zip(entity_texts, selector.tolist()) if keep]
        if not cluster_entity_texts:
            name_map[int(cluster_id)] = {
                "name": f"cluster_{int(cluster_id)}",
                "sampled_entities": [],
            }
            continue

        if use_mmr:
            base_texts = [
                f"{entity} || {sentence}"
                for entity, sentence in zip(cluster_entity_texts, cluster_sentence_texts)
            ]
            selected = select_diverse_exemplars_mmr(
                centroid=cluster_emb.mean(dim=0),
                entity_embeddings=cluster_emb,
                sentences=base_texts,
                k=min(n_samples, len(cluster_entity_texts)),
                mmr_lambda=mmr_lambda,
            )
            sampled = [item.split(" || ", 1)[0] for item in selected]
        else:
            sampled = rng.sample(cluster_entity_texts, min(n_samples, len(cluster_entity_texts)))
        entities_str = "{" + ", ".join(sampled) + "}"
        user_prompt = (
            f"Given the list of entities {entities_str}, which all belong to the same type, "
            "predict the entity type corresponding to all these entities. "
            "Respond only with the name of the entity type."
        )
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "A virtual assistant answers questions from a user based on the provided text.",
                },
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0},
        )
        raw_name = response.get("message", {}).get("content", "unknown")
        clean_name = _clean_ollama_label(raw_name, max_words=max_words)
        name_map[int(cluster_id)] = {
            "name": clean_name,
            "sampled_entities": sampled,
            "raw_response": raw_name,
        }

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in name_map.items()}, f, indent=2, ensure_ascii=False)

    return name_map

