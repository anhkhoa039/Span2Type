# Thesis Revision Report

## Summary

| Chapter / Section | change |
|---|---|
| Abstract (English) | Keywords: 12 → 5; body condensed to 1 page (3 paragraphs) |
| 摘要 (Chinese abstract) | Keywords: 12 → 5; body condensed to match English |
| Chapter 2 — Related Work | Multi-view span detection expanded; new template section added; cluster naming expanded |
| Chapter 3 — Methodology, Section 3.2 | Triplet loss worked example added; cross-reference to new Related Work section |
| Chapter 4 — Evaluation Metrics | AMI formula added; BERTScore P/R/F1 formulas added with examples |

---

## Change 1 — Keywords Reduced to 5

| Before (12 keywords) | After (5 keywords) |
|---|---|
| Open-world NER; Unsupervised NER; Domain Shift; Mention Detection; Multi-view Decoding; Entity Typing; Soft Prompting; Representation Learning; Clustering; Cluster Naming; Masked Language Modeling; Local LLM | Open-world Named Entity Recognition; Unsupervised NER; Multi-view Span Detection; Soft Prompt Entity Typing; Cluster Naming |

---

## Change 2 — Chapter 2 Related Work Expanded Per Stage

### 3a. Section 2.2 Multi-View Span Detection (existing section, expanded)

- A concrete BIO failure example: *"convolutional neural network"* requires three consecutive I-tags; one wrong tag breaks the span.
- Intuitive comparison of SE and TB signals against BIO — SE/TB are local (≤2 tokens) whereas BIO propagates state across the full span.
- New closing paragraph explaining why multi-view decoding benefits open-world settings specifically.

### 3b. Section 2.3 New section — "Fill-in-the-Blank Prompts for Entity Typing" (entirely new)

- Traces the template origin to the LAMA benchmark (Petroni et al., 2019).
- Cites Jiang et al. (2020) on template wording quality.
- Explains why `{sentence} {entity} is a [MASK].` matches BERT's pretraining distribution.
- Concrete worked example: *"BERT was introduced by Google researchers in 2018"* → top MLM predictions: *model, system, framework*.
- Compares against naive span pooling and context-free templates.

### 3c. Section 2.5 Automatic Cluster Naming (existing section, expanded)

- MLM majority-vote example: {*Einstein, Newton, Darwin*} → {*physicist, mathematician, biologist*} → majority selects *physicist*.
- LLM naming example: same cluster → *"historical scientist"* (multi-word, impossible with single `[MASK]`).
- Centroid-redundancy example motivating MMR.

---

## Change 3 — Chapter 3 More Examples for Background-Heavy Concepts

**Location:** Section 3.2 Entity Typing

Added worked example: *Obama (PER), Biden (PER), Google (ORG)* → valid triplet → loss term computed → gradient direction explained. Makes the abstract loss function concrete before the clustering stage.

---

## Change 4 — Chapter 4 Evaluation Metrics Formulas Added

### 5a. AMI Formula

Added after the existing AMI description:

- Mutual Information formula $\text{MI}(U,V)$ with joint/marginal probability notation.
- Full AMI formula showing chance-correction via $\mathbb{E}[\text{MI}]$ and entropy normalisation.
- Worked example: perfect clustering → AMI = 1.0; random split → AMI ≈ 0.

### 5b. BERTScore Formulas

Added after the existing BERTScore description:

- Three equations: $P_{\text{BERT}}$, $R_{\text{BERT}}$, $F_{\text{BERT}}$ (greedy max-cosine matching).
- Token notation $\hat{y}$, $y$, embedding $\mathbf{e}$ defined before equations.
- Single compact example covering: sentence-level token matching (*cat/mat* → F₁=0.953), BLEU vs BERTScore on paraphrases (BLEU: 0.034 vs BERTScore: 0.952), and NER naming context (*musician* vs *musicalartist* → F₁≈0.94).

---