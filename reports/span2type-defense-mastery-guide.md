# Span2Type — Defense Mastery Guide

---

## Section 1: Core Logic & Motivation

### The Fundamental Problem

Named Entity Recognition in well-resourced domains is a solved problem — BERT-based models achieve near-human F1 on CoNLL-2003. The bottleneck is **real-world deployment**: every new domain (clinical, legal, scientific, financial) has its own entity vocabulary, and manually defining types + annotating data is expensive, slow, and requires expert knowledge that may not exist at scale.

**OWNER** was an early solution: train a span detector on a labeled source (e.g., CoNLL-2003), transfer it zero-shot to the target, then use fill-in-the-blank BERT templates + k-means clustering to discover entity types without predefined labels. It was a principled, label-free pipeline.

### Where OWNER Falls Short

OWNER has two concrete weaknesses that Span2Type directly targets:

- **Brittle span detection.** OWNER's BIO-only mention detector is a sequential tagger trained on 4-type newswire. When applied to domains with multi-token technical terms (e.g., *support vector machine*, *convolutional neural network* in AI/Science), BIO tagging must decide each token's label independently. It frequently splits these spans or misses their boundaries entirely — dropping recall dramatically. Evidence: OWNER BIO scores only 60.5 F1 on the AI domain, compared to 81.35 for the DTrans multi-view detector (+20.85).
- **Uninterpretable cluster output.** After clustering, OWNER's output is *Cluster 3*, *Cluster 7*. The system's original naming strategy applies a single-template BERT query to all cluster entities and picks the majority-voted token. This produces redundant evidence (many near-identical entities near the centroid) and is constrained to a single vocabulary token. Domain experts receive no actionable label. This is what the thesis calls the **interpretability gap**.

### How Span2Type Bridges the Gap

Span2Type applies three surgical improvements to OWNER:

1. **Replace the BIO-only detector** with the DTrans multi-view module (BIO + Start-End + Tie-Break), which captures complementary span boundary signals simultaneously.
2. **Introduce soft prompt entity typing**, a parameter-efficient variant that adds only 3,840 learnable parameters while achieving better type-discriminative embeddings.
3. **Replace centroid-nearest naming** with MMR-based diverse exemplar selection + LLM/MLM consensus naming, closing the interpretability gap.

The pipeline requires **zero target-domain labels** at any stage. The result: average AMI jumps from 49.4 (OWNER) to 61.4 (Span2Type), a +12.0-point improvement across five CrossNER domains.

> **Defense Tip.** Frame this as a story of two bottlenecks: "OWNER proved the concept is valid, but left two engineering problems unsolved — finding spans reliably across domains, and giving those clusters names a human can act on. Span2Type solves both." This narrative is compelling and hard to attack.

---

## Section 2: The 3-View Detection Mechanism

### The Intuition: Three Angles on the Same Question

Every entity span can be viewed from multiple complementary perspectives. Imagine a sentence with the entity *"support vector machine"*:

- **BIO View:** Each token is labeled B, I, or O in sequence — *B-support, I-vector, I-machine*. This captures global sequential context but forces the model to make a sequence of locally dependent decisions; if it misclassifies the first token, the cascade fails.
- **Start-End (SE) View:** Two separate binary classifiers ask, *"Is this token the start of an entity?"* and *"Is this token the end of an entity?"* The end classifier is conditioned on the predicted start probability, improving boundary coherence. This gives explicit positional anchors that BIO's sequential logic may miss.
- **Tie-Break (TB) View:** For every adjacent token pair $(t, t{+}1)$, a binary classifier asks, *"Do these two tokens belong to the same span (Tie) or does a boundary fall between them (Break)?"* The hidden state used is $\mathbf{h}*t + \mathbf{h}*{t+1}$. This models local adjacency directly — something neither BIO nor SE represent.

### Why Three Views Together Are More Robust Than One

Each view has a distinct failure mode:


| View | Strength                     | Weakness                                                   |
| ---- | ---------------------------- | ---------------------------------------------------------- |
| BIO  | Global sequential structure  | Cascade errors; struggles with irregular multi-token terms |
| SE   | Precise boundary anchoring   | Cannot enforce span coherence between start and end        |
| TB   | Local adjacency within spans | Cannot detect the span as a whole unit                     |


At inference time, spans from all three decoders are **unioned** (after confidence thresholding). A span that BIO misses may be recovered by SE (which noticed a clear start token) or by TB (which noticed all adjacent pairs were Tied). This union strategy is a high-recall design: it is better to pass more candidates to the entity typing stage than to miss genuine entities. The downstream clustering step absorbs false positives far more gracefully than it absorbs false negatives.

**Empirically confirmed:** DTrans BIO-SE-TB achieves 82.35 average F1 vs. 75.68 for BIO-only (+6.67). The AI domain is the extreme case: +20.85 F1, because AI contains the most irregular multi-token technical terms.

### The Mean Teaching (EMA) Stabilizer

A teacher model maintains a running average of student parameters: $\tilde{\boldsymbol{\theta}}*t = \alpha\tilde{\boldsymbol{\theta}}*{t-1} + (1 - \alpha)\boldsymbol{\theta}_t$ with $\alpha = 0.995$. The teacher generates soft pseudo-labels on unlabeled data as an auxiliary consistency loss. This prevents the student from over-fitting to the source domain by regularizing it toward a temporally smoothed version of itself — the intuition is that a "wise average self" generalizes better than any single-epoch snapshot.

> **Defense Tip.** Use an analogy: "Think of it like three independent eyewitnesses describing the same crime scene from different vantage points. The jury trusts the testimony more when all three witnesses agree on a detail — and they catch more details combined than any one witness alone."

---

## Section 3: Clustering & BIC

### The Core Problem BIC Solves

K-means requires you to specify $k$ in advance. In open-world NER, the number of entity types in the target domain is unknown by definition. You need a principled criterion to automatically select the best $k$ from a candidate range $[k_{\min}, k_{\max}]$.

### The BIC Formula and its Mathematical Intuition

The BIC used in Span2Type is:

$$\text{BIC}(k) = n \cdot \log\left(\frac{\mathcal{I}(k)}{n}\right) + \log(n) \cdot k$$

- **First term** — *fit quality* (log-likelihood term): $\mathcal{I}(k)$ is the k-means inertia (total sum of squared distances from every point to its assigned centroid). As $k$ increases, inertia decreases monotonically because more clusters can fit more tightly. This term rewards models that explain the data well.
- **Second term** — *complexity penalty*: $\log(n) \cdot k$ grows linearly with $k$ and logarithmically with the dataset size. It penalizes adding clusters beyond what the data complexity warrants. The factor $\log(n)$ is not arbitrary — it comes from the asymptotic approximation of the marginal likelihood under the assumption of Gaussian cluster distributions (Schwarz, 1978). Larger datasets are harder to overfit, so the penalty grows with $n$.
- **Optimal $k$**: $k^* = \arg\min_k \text{BIC}(k)$. The minimum is where the marginal gain in fit quality no longer justifies the complexity cost. Beyond this point, each additional cluster is "explaining noise."

### Why BIC, Not Silhouette or Elbow?

This is a high-probability committee question. Here is a precise defense:

**Against Silhouette Score:**
The Silhouette score measures the mean ratio of inter-cluster to intra-cluster distances for each sample. It is a geometric measure that requires computing pairwise distances — $O(n^2)$ in the worst case. More critically, Silhouette makes no assumption about what the clusters represent and has no principled statistical derivation. Its scale (−1 to +1) is useful for interpretation but provides no theoretical guarantee that the maximizing $k$ matches the true number of generating distributions. BIC, derived from Bayesian model comparison, has a principled statistical basis: it approximates the posterior probability of the model given the data, so selecting $k^* = \arg\min \text{BIC}$ is equivalent to selecting the most probable number of Gaussian components.

**Against the Elbow Method:**
The elbow method plots inertia vs. $k$ and looks for a "kink." It has no formal definition of where the kink is — it is a heuristic requiring human judgment (or secondary criteria like second derivatives). It is non-reproducible and fails when the inertia curve is smooth with no sharp elbow, which is common in high-dimensional embedding spaces. BIC provides an objective, computable minimum.

**Summary for the jury:** "BIC is the theoretically grounded choice — it is derived from Bayesian model selection, automatically balances fit and complexity, and is fully reproducible. The alternatives either require human interpretation (Elbow) or lack statistical motivation (Silhouette)."

> **Defense Tip.** Commit this sentence to memory: *"BIC penalizes complexity in exact proportion to the logarithm of the sample size, which follows from Laplace's method of approximating integrals over model parameters — it is not an ad hoc penalty, it is Bayesian."* Even if you do not remember the full derivation, this phrasing signals mathematical literacy.

---

## Section 4: Ablation Results — What "Span2Type w/o Ensemble MD" Tells Us

### Reading the Ablation Table


| System                    | AI       | Lit.     | Music    | Politics | Science  | **Avg.**  |
| ------------------------- | -------- | -------- | -------- | -------- | -------- | --------- |
| OWNER (CoNLL)             | 44.3     | 46.6     | 50.1     | 53.7     | 52.3     | **49.4**  |
| Span2Type w/o Soft Prompt | 52.3     | 56.3     | 57.5     | 63.9     | 62.8     | **58.6**  |
| Span2Type w/o Ensemble MD | 49.31    | 53.01    | 58.84    | 64.09    | 63.75    | **57.80** |
| **Span2Type (Full)**      | **57.6** | **56.9** | **60.0** | **66.3** | **66.0** | **61.4**  |


### Interpreting "w/o Ensemble MD"

The "Span2Type w/o Ensemble MD" variant replaces the DTrans multi-view detector with a **single BIO head** (same backbone) while keeping the soft-prompt entity typing. Its average AMI is 57.80.

**What this isolates:** The contribution of the 3-view ensemble detector, holding the soft prompt constant. Comparing to the Full Model: $61.4 - 57.80 = 3.6$ AMI points are attributable specifically to using all three decoders vs. one, given that soft prompting is already applied.

But note also: "w/o Soft Prompt" (which has DTrans but no soft prompt) scores 58.6. So soft prompt alone contributes +2.8 points. Together, both components produce 61.4 — the improvements are **additive but not perfectly so** (small synergy).

**The deeper message:** The "w/o Ensemble MD" result shows that the multi-view detector's contribution flows through the entire pipeline. Better spans → more coherent entity embeddings at [MASK] → cleaner k-means solution → higher AMI. The ablation confirms a **cascade effect**: detection quality improvement propagates and amplifies through downstream typing and clustering (the +12.0 total AMI gain exceeds the +6.67 detection F1 gain, pointing to this compounding).

> **Defense Tip.** When presenting this, say: *"The 'without ensemble' row is the critical control. It tells us that even when we give the system the best entity typing we have — soft prompts — it still loses 3.6 AMI points without the multi-view detector. The two improvements are not redundant; they address different failure modes independently."*

---

## Section 5: The "Why" Behind the Math — Loss Functions & Embedding Refinement

### The Fill-in-the-Blank Template: What Are We Forcing the Model to Learn?

The template `[CLS] {sentence} {entity} is a [MASK] . [SEP]` is not arbitrary. BERT was pre-trained to predict masked tokens in context. By asking it to fill in *"{entity} is a [MASK]"*, we are exploiting the model's learned world knowledge to produce a **type-predictive representation** at the [MASK] position.

In plain English: instead of asking "what embedding does this token sequence have?", we are asking BERT "what kind of thing is this entity?" The [MASK]-position hidden state becomes type-sensitive by construction — it encodes the model's best guess about the category label. This is far more discriminative for clustering than a raw span embedding (e.g., average of token embeddings), which encodes surface form rather than semantic type.

### The Triplet Margin Loss: What Are We Forcing the Model to Learn?

$$\mathcal{L}*{\text{triplet}} = \frac{1}{|\mathcal{T}|} \sum*{(i,j,k) \in \mathcal{T}} \max\left(0, \mathbf{e}_i - \mathbf{e}_j_2 - \mathbf{e}_i - \mathbf{e}_k_2 + \gamma\right)$$

In plain English: for every triple of entities (anchor, positive=same type, negative=different type), we demand that the anchor is **at least $\gamma = 1.0$ units closer to the same-type entity than to the different-type entity**. If this margin is violated, the loss is positive and backpropagation adjusts the embeddings.

**What the model learns:** A metric space where same-type entities form tight clusters and different-type entities are pushed apart. After training, when you run k-means on these embeddings, the geometric structure is type-aligned — the algorithm is recovering boundaries the encoder was explicitly trained to create.

**Why this transfers across domains:** Entity type boundaries (e.g., *Person vs. Organization*) are more stable across domains than the specific vocabulary. A physicist and a politician are both *Person*-type entities even though their contexts are radically different. The triplet loss on source-domain labels teaches the model these structural distinctions, which generalize.

### The Soft Prompt: What Are We Forcing the Model to Learn?

Five trainable vectors $\mathbf{P} = [\mathbf{p}_1, \ldots, \mathbf{p}_5] \in \mathbb{R}^{5 \times 768}$ are prepended to every input. These vectors have no linguistic meaning initially — they are random noise. Through backpropagation, they learn to **shift the BERT encoder's attention distribution** toward type-relevant features.

In plain English: instead of modifying the encoder's 110M parameters (which risks overwriting valuable pre-trained knowledge), we are learning a 5-token "prefix" that steers the frozen encoder toward producing more type-discriminative [MASK] representations. The backbone stays intact; the prompt teaches it how to "look at" entity mentions differently.

The **two-stage schedule** is essential: Stage 1 warms up the prompts (so they learn meaningful steering directions), then Stage 2 fine-tunes the whole model at $10^{-5}$ (far below the usual $2 \times 10^{-5}$) to prevent catastrophic forgetting — the prompt shows the backbone where to look, and the backbone then refines globally without forgetting its pre-trained world knowledge.

### The MMR Objective: What Are We Forcing the Selection to Achieve?

$$\text{MMR}(s) = \lambda \cdot \cos(\mathbf{e}_s, \boldsymbol{\mu}*c) - (1-\lambda) \cdot \max*{r \in \mathcal{R}} \cos(\mathbf{e}_s, \mathbf{e}_r)$$

In plain English: at each selection step, choose the entity that is (a) close to the cluster centroid (relevant) but (b) far from entities already selected (diverse). The $\lambda = 0.7$ weighting slightly favors relevance over diversity.

**What we are forcing:** Coverage of the cluster's semantic range, not redundancy near its center. If the first selected exemplar is *"Einstein"*, we do not want exemplar 2, 3, 4 to all be other physicists. We want the LLM to see the full distribution: a physicist, a philosopher, a composer — so it can correctly infer the broader label *"person"* rather than the narrow label *"scientist"*.

> **Defense Tip.** For the triplet loss, say: *"We are sculpting the embedding space — the loss is a geometry teacher telling the model: if two entities have the same type, they must live near each other; if they have different types, they must live far apart. After training, k-means just has to find the natural clusters that the loss created."*

---

## Section 6: Critical Definitions — The 5 You Must Know Cold

### 1. Adjusted Mutual Information (AMI)

**Formal:** AMI measures the agreement between two partitions (predicted cluster assignments vs. gold type labels) after correcting for chance:

$\text{AMI}(U, V) = \frac{I(U, V) - \mathbb{E}[I(U, V)]}{\frac{1}{2}[H(U) + H(V)] - \mathbb{E}[I(U, V)]}$

where $I(U, V)$ is mutual information and $H$ is entropy.

**Why used:** AMI is **mapping-free** — it does not require a cluster-to-type assignment, so it is immune to label permutation. It is also corrected for the baseline chance agreement, which matters because naive random clustering could score non-trivially on mutual information by coincidence. Score of 0 = random; score of 1 = perfect agreement.

**One-line definition for the jury:** *"AMI tells us how much knowing the cluster assignment reduces our uncertainty about the true entity type, after subtracting out what we would have gotten by chance."*

---

### 2. Bayesian Information Criterion (BIC)

**Formal:** $\text{BIC}(k) = n \cdot \log(\mathcal{I}(k)/n) + \log(n) \cdot k$

**Intuition:** A penalized log-likelihood: the log-fit term rewards explanatory power, the $\log(n) \cdot k$ term penalizes complexity. Derived from the Laplace approximation of the Bayesian marginal likelihood. Selects the $k$ that represents the best trade-off between model fit and parsimony.

---

### 3. Maximal Marginal Relevance (MMR)

**Formal:** $\text{MMR}(s) = \lambda \cdot \cos(\mathbf{e}_s, \boldsymbol{\mu}*c) - (1-\lambda) \cdot \max*{r \in \mathcal{R}} \cos(\mathbf{e}_s, \mathbf{e}_r)$

**Intuition:** A greedy selection algorithm that balances query relevance against redundancy with already-selected items. Originally proposed by Carbonell & Goldstein (1998) for document summarization; applied here to select diverse cluster exemplars for naming.

---

### 4. Batch Triplet Margin Loss

**Formal:**

$$\mathcal{L}*{\text{triplet}} = \frac{1}{|\mathcal{T}|} \sum*{(i,j,k) \in \mathcal{T}} \max(0, \mathbf{e}_i - \mathbf{e}_j_2 - \mathbf{e}_i - \mathbf{e}_k_2 + \gamma)$$

where $(i, j, k)$ is a valid triplet with $y_i = y_j$, $y_i \neq y_k$, and $\gamma = 1.0$ is the margin.

**Intuition:** A metric learning objective that shapes the embedding space so same-type entities cluster together and different-type entities are pushed apart by at least $\gamma$ units. All valid triplets within a batch are extracted in-batch (no need to mine triplets externally).

---

### 5. BERTScore-F1

**Formal:** For a predicted string $\hat{y}$ and reference string $y$, BERTScore computes token-level cosine similarities between contextual embeddings (RoBERTa-large in this thesis) and reports the greedy-matched Precision, Recall, and F1.

**Why used:** Unlike exact-match accuracy, BERTScore awards partial credit for semantically close predictions (e.g., *researcher* vs. *author* scores ~0.95 rather than 0). This is essential for evaluating open-vocabulary cluster names where semantic proximity matters more than string equality. Note: even a completely wrong label scores ~0.86 due to shared embedding space — differences above that floor reflect genuine semantic content.

> **Defense Tip.** Anticipate the question *"Isn't 0.86 too high a floor — isn't everything scoring high?"* Your answer: *"The floor exists because all labels share a common linguistic space, but the ceiling is 1.0 and the range above 0.86 is discriminative. In our results, the gap between OWNER-MLM (0.892–0.906) and Span2Type-LLM-MMR (0.922–0.943) represents a meaningful step-up in semantic precision, not noise."*

---

## Quick-Reference Defense Card


| If asked about...    | Core answer in one sentence                                                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why open-world NER?  | Real domains have unknown, evolving entity types — predefined ontologies are always incomplete.                                                                         |
| Why DTrans over BIO? | BIO captures one sequential view; DTrans captures three complementary views (boundary positions, adjacency) — union of all three recovers spans any single view misses. |
| Why BIC over Elbow?  | BIC has a principled Bayesian derivation; the Elbow method requires subjective human judgment.                                                                          |
| Why MMR for naming?  | Centroid-nearest exemplars are redundant; MMR ensures the LLM sees the cluster's full semantic range.                                                                   |
| Why soft prompt?     | It achieves near-full-model expressiveness at 0.003% of the parameter count (3,840 vs. 110M).                                                                           |
| Key result?          | Span2Type (CoNLL) achieves 61.4 avg AMI vs. 49.4 for OWNER — a 12-point gain, with 9.2 from DTrans and 2.8 from soft prompts.                                           |


