# Speaker Notes — Span2Type Thesis Defense
**Target duration:** 35–40 minutes  
**Total slides:** 32 main + 4 appendix  
**Pace guide:** ~1.5 min/slide on average; methodology slides run 2 min, result slides 1 min

---

## Timing Overview

| Section | Slides | Target time |
|---|---|---|
| Opening | 1–2 | 3 min |
| Problem Statement | 3–4 | 4 min |
| Background & Related Work | 5–9 | 7 min |
| Proposed Solution | 10–11 | 4 min |
| Methodology | 12–19 | 13 min |
| Experiments & Results | 20–25 | 8 min |
| Discussion | 26 | 3 min |
| Conclusion | 27–28 | 3 min |
| Closing | 29 | 1 min |
| **Total** | | **~40 min** |

---

## Section 1 — Opening

### Slide 1 · Title Page
**⏱ ~1 min**

Good morning, committee members, and thank you for being here today.
My name is Nguyen Minh Khoa, and this is my thesis defense for the degree of Master of Science in Computer Science.
The title of my thesis is *Span2Type: An Improved Open-World Named Entity Recognition Pipeline*, and I will be presenting the research I have conducted over the past year under the supervision of Professor [Supervisor Name].

> **Tip:** Pause after saying the thesis title. Let the committee settle in before moving on.

---

### Slide 2 · Outline
**⏱ ~1 min**

My presentation is structured in seven sections.
I will begin with the problem that motivates this work, then briefly cover the related work that situates my contributions.
The bulk of the talk — roughly half the time — will be spent on the methodology and experimental results, where I want to explain not just *what* the system does but *why* each design decision was made.
Please hold questions for the Q&A after the presentation, though I will pause naturally between major sections if you need a moment to reflect.

> **Tip:** Point to the two columns on the slide as you name the sections. This sets expectations clearly.

---

## Section 2 — Problem Statement

### Slide 3 · Named Entity Recognition in the Wild
**⏱ ~2 min**

Named Entity Recognition is the task of identifying mentions of entities in text and classifying them into predefined semantic types such as Person, Organization, and Location.
Modern systems based on BERT and RoBERTa achieve near-human performance on standard benchmarks like CoNLL-2003, so one might ask: is the problem solved?

The answer is no — because every high-performing NER system operates under a hidden assumption that is almost never satisfied in practice.
The type inventory must be fixed and known in advance, and large amounts of labeled data must exist for every type.
The moment we move to a new domain — say, NER for an artificial intelligence research corpus, or a pharmaceutical knowledge base — this assumption collapses entirely.
The two panels on this slide illustrate the gap: on the left, a clean CoNLL sentence with neat labels; on the right, an AI domain sentence where the entity types are simply unknown.

> **Tip:** Point to the right panel (red box) when saying "the assumption collapses." This visual reinforces the problem.

---

### Slide 4 · Three Concrete Challenges
**⏱ ~2 min**

I want to frame the problem around three concrete challenges, because these challenges map directly onto the three contributions of this thesis.

The first challenge is **domain shift in span detection**: a BIO tagger trained on newswire text struggles with multi-token technical entities like *support vector machine* or *convolutional neural network*, which have irregular boundaries that a single-view sequential tagger often gets wrong.

The second challenge is the **unknown type inventory**: new domains have finer-grained types, entirely new categories, or evolving terminology that supervised models simply cannot discover.

The third challenge, which I think is the most underappreciated, is the **interpretability gap**: even if an unsupervised system clusters entity mentions correctly, the output is *Cluster 3*, *Cluster 7* — anonymous identifiers that a domain expert must still manually inspect and label, reintroducing the human effort we were trying to eliminate.

The table at the bottom of this slide shows that existing methods address at most one or two of these challenges. Span2Type is the first system to address all three simultaneously.

> **Tip:** Tap the table row for "Span2Type" and note the three checkmarks. This is a strong visual argument — let it land before transitioning.

---

## Section 3 — Background & Related Work

### Slide 5 · NER: From CRFs to Transformers
**⏱ ~1.5 min**

NER research has progressed impressively over two decades, from Conditional Random Fields in 2001, through BiLSTM-CRF in 2016, to transformer-based models that achieved near-human F1 on CoNLL-2003 around 2019.
The performance plateau you can see on this timeline is real — benchmark performance has saturated.

But this benchmark success created a subtle illusion of progress.
Every method on this timeline — even the most recent large language model approaches — assumes either a fixed type inventory or at least a few labeled examples per type.
Open-world NER removes both assumptions: no predefined types, no target-domain annotations, types discovered automatically via clustering.

> **Tip:** Use the timeline as a pointer path — draw the audience's eye left to right, then pause at "Open-world NER" on the right.

---

### Slide 5b · Related Methods — Span Detection Approaches
**⏱ ~1.5 min**

Before we look at OWNER specifically, I want to place it in the context of span detection methods in NER, because four of the methods on this slide reappear as baselines in my evaluation.

The traditional BIO tagging approach — used in OWNER — is efficient but fragile: a single token mislabelled cascades into a lost span.
Span enumeration methods like PURE explicitly score every possible span, removing the error propagation problem, but their cross-domain performance is surprisingly poor — only F1 35.8 on average on CrossNER — because the scoring function overfits to source-domain surface patterns.

SpanProto and WL-Coref take transfer-oriented approaches — prototype matching and coreference boundaries respectively — and do better at 61 and 65 F1, but they each rely on a single detection signal.

DTrans, proposed by Zheng et al. in 2022, resolves this by running three complementary decoders simultaneously — BIO, Start-End, and Tie-Break — and using Mean Teacher EMA to enforce consistency across views.
At 82.4 average F1, it achieves the strongest cross-domain detection, and this is the module I integrate into Span2Type.

> **Tip:** Point to each method row as you name the F1 numbers. The jump from 65 to 82 is striking — let it register.

---

### Slide 5c · Related Methods — Entity Typing & Cluster Naming
**⏱ ~1.5 min**

This slide covers the two remaining dimensions of related work: how prior systems handle the entity type list, and how they select exemplars for cluster naming.

On the type list question: supervised NER requires a human to define the type inventory upfront — PER, ORG, LOC.
Few-shot approaches reduce the labeling burden, but they still need the target type names at test time.
OWNER is the first system to generate the type list automatically via clustering — no one tells it what types exist.
However, the names it assigns to those clusters are often coarse or redundant, which is the limitation this thesis addresses.

For parameter-efficient tuning: full fine-tuning updates over 110 million parameters, making it expensive for domain transfer.
Soft prompts — introduced by Lester et al. in 2021 — freeze the entire PLM and prepend a small number of trainable token embeddings.
At fewer than 4,000 parameters, they are over 25,000 times more parameter-efficient than full fine-tuning, and this is the approach I adopt for entity type embedding.

For naming: OWNER's original MLM method feeds all entities in a cluster through a single fixed template and takes a majority vote — no exemplar selection at all.
When the OWNER paper is adapted to use an LLM backend, random sampling of 16 entities is used.
Both approaches have weaknesses: all-entities is slow and redundant on large clusters; random sampling is unstable across runs.
Centroid-nearest selection — which I introduce as an ablation baseline — is representative, but the selected members tend to be nearly identical surface forms.
MMR addresses this by jointly optimising relevance to the cluster centroid and diversity among selected exemplars.

> **Tip:** Use the exampleblock at the bottom right as your pivot point — "these three gaps are exactly what I address, which I'll describe on the next two slides."

---

### Slide 6 · OWNER — The State-of-the-Art Baseline
**⏱ ~1 min**

The direct baseline for my thesis is OWNER, proposed by Wang et al. in 2023, which is the first complete open-world NER pipeline.
OWNER works in three stages: a DeBERTa-based BIO mention detector trained on a labeled source domain, a BERT fill-in-the-blank template to encode each detected entity span, and BIC-guided k-means clustering to assign unsupervised type labels.
It achieves 49.4 average AMI on the CrossNER benchmark — a strong result for an entirely unsupervised system.

For cluster naming, OWNER feeds every entity in a cluster through a single fixed template and takes the most frequent MLM prediction as the type name — no exemplar selection step.
This is fast for small clusters, but becomes slow and noisy when clusters are large, and the naming output tends to be generic.

I identified two specific bottlenecks in OWNER that Span2Type directly targets: the single-view BIO tagger, which was never designed for cross-domain robustness, and the no-selection naming procedure.

> **Tip:** Point to the two red callout labels on the pipeline diagram as you name each weakness. You have already introduced these issues in detail on the previous two slides, so keep this slide brief — one sentence per weakness is enough.

---

### Slide 7 · Research Gap & Thesis Objectives
**⏱ ~2 min**

From the literature review, I identified three gaps that no existing work addresses.

**Gap 1:** No open-world NER system uses multi-view span detection for cross-domain robustness. All existing systems, including OWNER, use a single-view BIO tagger.

**Gap 2:** Soft prompt tuning — a parameter-efficient fine-tuning technique that has been shown to match full fine-tuning at scale — has never been applied to entity type embedding learning in the open-world setting.

**Gap 3:** Exemplar diversity for cluster naming is unexplored. OWNER's all-entity approach is noisy on large clusters, random subsampling is unstable, and centroid-nearest selection produces redundant exemplars — none guarantees a representative spread of cluster evidence.

These three gaps become my three thesis objectives, which I will describe in the methodology section.
I also want to emphasize the key constraint shown in the green box at the bottom right: **zero target-domain supervision at any pipeline stage** — no labels, no type names, nothing from the target domain during training or inference.

---

## Section 4 — Proposed Solution

### Slide 8 · Span2Type System Architecture
**⏱ ~1.5 min**

This slide shows the overall architecture of Span2Type.
The system takes an unannotated target-domain corpus as input and produces, for every sentence, a set of entity spans each assigned to a human-readable type name.

The pipeline has three stages.
Stage 1 uses the DTrans multi-view entity detector — I'll explain the architecture in detail shortly — which replaces OWNER's BIO-only tagger.
Stage 2 encodes each detected span using a BERT fill-in-the-blank template with an optional soft prompt variant, then clusters the embeddings with BIC-guided k-means.
Stage 3 selects diverse representative exemplars for each cluster using Maximal Marginal Relevance and queries either a masked language model or a locally hosted large language model to generate a human-readable type name.

The Span2Type.png figure here is taken directly from my thesis report, so the committee may recognize it.

> **Tip:** Point to each stage box in sequence (left to right) as you describe them. Take your time — this is the key overview slide.

---

### Slide 9 · Mapping Challenges to Contributions
**⏱ ~1.5 min**

This table is the conceptual core of the thesis.
Each row maps one challenge to the root cause I identified and the specific Span2Type solution that addresses it.

I want to be transparent about scope: Challenge 2 — the unknown number of entity types — was already solved elegantly by OWNER's BIC-guided clustering, and I retain it unchanged.
The three novel contributions in this thesis address Challenges 1 and 3, plus the soft prompt improvement which cross-cuts Challenge 2 by producing better embeddings for the clustering step.

The green box at the bottom captures what makes Span2Type more than the sum of its parts: the improvements compound through the pipeline.
Better span detection feeds cleaner spans into the typing stage, which produces more coherent clusters, which in turn makes naming easier.
The ablation study will confirm this cascade effect quantitatively.

> **Tip:** Read the numbers at the bottom right slowly: "OWNER 49.4, Span2Type 61.4, delta plus 12 points." These are the headline numbers — let them register.

---

## Section 5 — Methodology

### Slide 10 · Stage 1 — Multi-View Detection Motivation
**⏱ ~2 min**

Let me begin the methodology section with the motivation for multi-view entity detection.

The standard approach — used in OWNER — is to classify each token independently as B, I, or O in a sequential pass.
This works well when the source and target domain are stylistically similar, but it breaks down for multi-token technical entities because a single wrong token tag causes the entire span to be lost.

The key insight is that entity boundaries can be described from three complementary perspectives simultaneously.
The BIO view captures the global sequential context of the span.
The Start-End view explicitly marks the first and last token of each entity — directly encoding the boundary information that BIO must infer indirectly.
The Tie-Break view models whether two adjacent tokens co-belong to the same span — a local, pairwise signal that neither BIO nor SE captures alone.

The annotation example on the right illustrates this for the phrase *support vector machine*.
All three views agree on the same three-token span, and critically, even if BIO mislabels the middle token *vector* as O, the Start-End decoder still correctly marks *support* as Start and *machine* as End, recovering the full span.

> **Tip:** Walk through the three rows in the annotation table slowly. The committee should understand the complementary nature of the views before you proceed.

---

### Slide 11 · DTrans Architecture — Three Decoders
**⏱ ~2 min**

The DTrans entity detection module implements this multi-view strategy using BERT-base-cased as a shared encoder.
Given an input sentence, the encoder produces one contextual hidden state per token.
Three independent classification heads are then applied to these shared representations in parallel.

The BIO head is a standard three-class linear classifier — Beginning, Inside, Outside.

The Start-End head is more interesting: the Start classifier is a standard binary classifier, but the End classifier takes as input not just the token's hidden state but also the predicted Start probability from the Start classifier.
This conditioning enforces boundary coherence — the model is penalized for predicting an End token that is spatially inconsistent with the predicted Start.

The Tie-Break head operates on adjacent token pairs by summing their hidden states, then predicting whether those two tokens co-belong to a span.

The total training loss is simply the sum of the three cross-entropy losses, which allows the shared encoder to learn representations that are simultaneously useful for all three boundary perspectives.

At inference time, the three decoders independently predict span candidates and the final entity set is the **union** of all three outputs — this is the design choice that maximizes recall under domain shift.

At the bottom of the right column, I show the Mean Teaching strategy: a teacher model maintained as an exponential moving average of the student's parameters provides soft pseudo-labels during training, which stabilizes the cross-domain transfer.

> **Tip:** The TikZ diagram is the centerpiece here — point to the three branches fan-out and then to the union merge box.

---

### Slide 12 · Entity Typing — Fill-in-the-Blank Template
**⏱ ~2 min**

Stage 2 encodes each detected entity span into a type-sensitive vector representation using a fill-in-the-blank template.

The template is simple: the original sentence is followed by the entity mention repeated explicitly, and then the phrase "is a [MASK]".
For the entity *Albert Einstein* in the sentence about Germany, the template asks BERT: "Albert Einstein is a [MASK]" — and BERT, leveraging its pretraining on billions of text examples, fills in *physicist* or *scientist*.

This is elegant because it converts a representation learning problem into a cloze task that BERT was already trained on.
The entity embedding we extract is the hidden state at the [MASK] position — not a raw span average, but a representation that has been asked to predict the type of the entity.
This makes it inherently more type-discriminative than pooling the span tokens.

We then fine-tune the encoder using a Batch Triplet Margin Loss, which pushes embeddings of same-type entities together and pulls embeddings of different-type entities apart.
All valid triplets in each batch are used — the same strategy as FaceNet for face verification.
The goal is a metric space where k-means clustering will naturally recover semantically coherent entity type groups.

> **Tip:** The scatter plot on the right shows the before/after intuition. Reference it when saying "a metric space where k-means recovers coherent groups."

---

### Slide 13 · Soft Prompt — 3,840 vs. 110 Million Parameters
**⏱ ~2 min**

Full fine-tuning of the BERT encoder updates all 110 million parameters.
This is computationally expensive and, more importantly, risks catastrophic forgetting: when we fine-tune aggressively on source-domain data, the model can overwrite pretrained world knowledge that is crucial for zero-shot transfer to unseen domains.

My soft prompt variant addresses this by keeping the backbone frozen and instead prepending five trainable soft token embeddings — shown in gold on the token diagram — to the input sequence before passing it to the frozen BERT.
The attention mask and [MASK] position index are adjusted accordingly.

The parameter count comparison at the bottom of the slide is striking: 3,840 trainable parameters versus 110 million.
But the efficiency is not the primary motivation — the deeper reason is that a frozen backbone transfers more predictably to unseen domains, because its representations remain anchored to what BERT learned during pretraining.
The soft prompt acts as a domain-specific steering signal that tells BERT *how to interpret* the template for entity typing, without disturbing the backbone's generalization capacity.

> **Tip:** Reference the resized token diagram on the right — point to the gold soft tokens and the frozen BERT encoder box below them.

---

### Slide 14 · Two-Stage Prompt-Initialized Fine-Tuning
**⏱ ~2 min**

There is a fundamental tension in soft prompt tuning: soft prompts alone limit expressiveness, but full fine-tuning from the first epoch risks catastrophic forgetting.
I resolve this with a two-stage schedule controlled by a threshold epoch set to 10.

In Stage 1, the PLM is strictly frozen and only the five soft token embeddings are updated at a high learning rate of 3×10⁻³.
This allows the prompt to converge rapidly in a low-dimensional 3,840-parameter space, without perturbing the pretrained backbone at all.
Think of this as teaching the prompt how to steer BERT before you hand BERT the wheel.

At epoch 10, Stage 2 begins: the backbone is unfrozen and its parameters are added to the optimizer via `add_param_group()`.
The learning rate for the PLM is set to 1×10⁻⁵ — three hundred times smaller than the prompt learning rate.
This large disparity is essential: a high PLM learning rate at Stage 2 would overwrite the pretrained representations just as effectively as full fine-tuning from scratch.

Two implementation details are critical for correctness.
First, I use `add_param_group()` rather than reinitializing the optimizer — this preserves the accumulated first and second moment estimates for the soft prompt, which carry meaningful gradient history from Stage 1.
Second, the learning rate scheduler is reset at the Stage 2 boundary so the PLM learning rate decays from its full value rather than inheriting a mid-schedule multiplier.

> **Tip:** The learning rate schedule plot on the right is the clearest way to explain this. Point to the vertical dashed line at epoch 10 as the "unfreeze" moment.

---

### Slide 15 · BIC-Guided Cluster Count Selection
**⏱ ~1.5 min**

After training the entity encoder, all target-domain entity spans are embedded and we face a fundamental problem: we do not know how many entity types exist in the target domain.

I retain OWNER's BIC-guided AutoKmeans approach unchanged, because it is a principled and effective solution to this problem.
The system searches over a range of candidate k values from 2 to 30 in steps of 2, fitting k-means with k-means++ initialization and 10 random restarts for each candidate.
The Bayesian Information Criterion selects the k that best balances fit quality — measured by inertia — against model complexity — penalized linearly with k.

The BIC curve on the right shows the characteristic shape: BIC decreases initially as adding more clusters improves the fit, then increases as the complexity penalty dominates.
The optimal k is at the minimum.

I note in the discussion that BIC can overestimate k for domains with highly imbalanced type distributions — Politics has 24 predicted clusters versus 9 gold types — but this is a known limitation of parametric model selection and is outside the scope of this thesis.

---

### Slide 16 · Stage 3 — MMR-Based Cluster Naming
**⏱ ~2 min**

Once clusters are formed, we face the interpretability gap: how do we assign a human-readable name to each cluster without any ground-truth type information?

OWNER's approach is to pick the single entity closest to the cluster centroid and query a naming model with that one example.
The problem is that the centroid-nearest entity is often not representative of the cluster's full semantic range — it is just the most average member — and querying with a single redundant example produces unstable, over-specific type names.

My MMR-based approach selects 16 diverse, centroid-relevant exemplars for each cluster.
The MMR score for each candidate balances two terms: relevance to the cluster centroid, and dissimilarity from exemplars already selected.
The lambda parameter of 0.7 slightly favors centroid relevance over diversity.
The result is a set of 16 exemplars that spans the full semantic range of the cluster rather than clustering near the centroid.

The scatter plot on the right makes this concrete: the red squares (centroid-nearest) are all tightly clustered near the center; the green circles (MMR-selected) are spread across the cluster, including members near the periphery.
These peripheral members are exactly what helps the naming model understand the full scope of the cluster.

> **Tip:** The scatter plot is the key visual here. Trace the difference between red and green with your finger or pointer.

---

### Slide 17 · Two Naming Backends — MLM and LLM
**⏱ ~1.5 min**

The MMR-selected exemplars are passed to one of two interchangeable naming backends.

The MLM backend uses BERT with a fill-in-the-blank naming template.
It independently computes a vocabulary distribution for each of the 16 exemplars and averages them.
The strength is simplicity and speed; the fundamental limitation is that BERT can only predict a single subword token at the [MASK] position.
This makes it impossible to express compound CrossNER types like *musicalartist* or *literarygenre* — which are semantically multi-word but syntactically a single concatenated string.

The LLM backend sends all 16 entity mentions in a single two-turn prompt to Llama-3 running locally via Ollama.
Temperature is set to zero for deterministic output, and the response is post-processed to at most three words.
The LLM can generate multi-word labels like *political party* or *research institute*, and it is immune to the MLM hallucination artefact I will describe in the discussion.

The comparison table at the bottom previews the key result: the LLM backend achieves 0.956 average BERTScore-F1 versus 0.896 for the MLM backend.

---

## Section 6 — Experiments & Results

### Slide 18 · Experimental Setup
**⏱ ~1.5 min**

Let me describe the experimental setup before presenting results.

For source training data, I use two corpora.
CoNLL-2003 is the standard English newswire benchmark with four entity types and 14,000 training sentences.
Pile-NER is a large diverse corpus derived from the Pile web crawl, covering approximately 13,000 entity types across 50,000 sentences.

For target evaluation, I use the CrossNER benchmark across five domains: AI, Literature, Music, Politics, and Science — each with between 9 and 17 gold entity types.
The critical constraint is shown in the red box: gold labels are used **only for evaluation**.
They are never seen during training, clustering, or naming.

The hyperparameter table on the right summarizes the key settings for each component.
I follow the DTrans paper's hyperparameters for the entity detector and the OWNER paper's hyperparameters for the typing and clustering stages, making targeted changes only where I propose improvements.

---

### Slide 19 · Evaluation Metrics
**⏱ ~1.5 min**

I evaluate the pipeline at three levels, using one metric per stage.

For span detection, I report standard span-level F1 with exact match.

For entity typing, I report Adjusted Mutual Information, or AMI.
AMI is mapping-free — it does not require any correspondence between predicted cluster labels and gold type labels, making it appropriate for the open-world setting where cluster IDs are arbitrary.
It is robust to label permutations, over-segmentation, and over-merging. A score of 0 means random clustering; a score of 1 means perfect agreement with the gold partition.

For cluster naming, I report BERTScore-F1 using RoBERTa-large contextual embeddings.
I want to briefly explain the BERTScore range shown in the figure, because the absolute values require calibration.
Due to the shared contextual embedding space, even a completely wrong label scores approximately 0.86 with RoBERTa-large.
The meaningful range is therefore 0.86 to 1.0: a score of 0.95 corresponds roughly to synonym-level semantic equivalence, and 0.90 corresponds to a related concept.
Please keep this calibration in mind when I report the naming results.

> **Tip:** Point to the bertscore_bar.png figure as you explain the range. The visual makes the calibration intuitive.

---

### Slide 20 · Main Results — End-to-End AMI
**⏱ ~2 min**

The primary result is shown in this table and bar chart.
Span2Type trained on CoNLL-2003 achieves **61.4 average AMI** across the five CrossNER target domains — the best result among all systems compared.

The improvement over the direct OWNER baseline is **+12.0 points** — from 49.4 to 61.4.
Critically, this improvement is **consistent across all five domains**: AI gains 13.3 points, Literature 10.3, Music 9.9, Politics 12.6, and Science 13.7.
Consistency across domains is important because it suggests the improvements generalize across domain types rather than overfitting to a particular style of text.

Among the LLM-prompting baselines, UniNER with GPT-4o mini is the strongest at 47.4 — and Span2Type still outperforms it by 14 points, despite using only a local Llama-3 model and no GPT-4 access.
The ChatIE baselines lag far behind at 22 to 35 AMI, confirming that zero-shot LLM prompting alone is not sufficient for open-world NER.

I will discuss why LLM-prompting baselines structurally underperform in the discussion section.

> **Tip:** Read the five per-domain delta values (+13.3, +10.3, +9.9, +12.6, +13.7) clearly and slowly. Consistent gains across all five domains are a strong argument.

---

### Slide 21 · Entity Detection Results
**⏱ ~1.5 min**

Moving to the span detection stage, this table compares the detection F1 of different architectures, all trained on CoNLL-2003 and evaluated on the five CrossNER target domains.

The DTrans multi-view detector, BIO-SE-TB, achieves an average F1 of 82.35, compared to 75.68 for the OWNER BIO-only baseline — a gain of **6.67 F1 points** on average.

The most dramatic gain is in the AI domain: **+20.85 points**, from 60.5 to 81.35.
This makes sense because AI entity mentions are disproportionately long multi-token technical phrases — *support vector machine*, *recurrent neural network* — exactly the case where the Start-End and Tie-Break decoders add the most value over sequential BIO tagging.

I want to be transparent about the one regression: Literature, where OWNER BIO achieves 83.9 versus BIO-SE-TB's 82.68, a loss of 1.22 points.
Literary entity mentions — character names, author surnames, book titles — follow predictable proper-noun patterns that a sequential BIO tagger handles reliably.
In this regime, the extra span candidates introduced by the SE and TB decoders slightly reduce precision without a compensating recall gain.
This is an honest failure mode that I analyze in the discussion.

> **Tip:** Highlight the AI column (+20.85) and the Literature regression (-1.22). Acknowledging the regression builds credibility with the committee.

---

### Slide 22 · Cluster Naming Results
**⏱ ~1.5 min**

The naming results show a clear hierarchy among the variants.

Span2Type-LLM with MMR achieves the highest average BERTScore-F1 of **0.956** and is the best system in three of five domains.

The most important comparison is the **budget-controlled** one in the bottom half of the slide: both OWNER-LLM and Span2Type-LLM use exactly 16 exemplars with the same Llama-3 model, differing only in selection strategy — random versus MMR.
Span2Type-LLM outperforms OWNER-LLM by +0.012 on average, confirming that the MMR selection strategy itself adds value beyond what random sampling achieves under the same inference budget.

The naming_summary.png figure on the right panel is particularly instructive.
The right panel shows the correlation between cluster purity and BERTScore-F1 — and the Pearson r is essentially zero, at 0.01.
This means naming quality is **largely independent** of clustering quality: the LLM can assign semantically accurate type labels even to impure clusters by identifying the dominant type from the exemplar evidence.

---

### Slide 23 · Ablation Study
**⏱ ~2 min**

The ablation study isolates the contribution of each component by progressively removing or replacing one part of the full system.

Starting from the OWNER baseline at 49.4 average AMI, replacing the BIO-only detector with DTrans multi-view detection — while keeping all other components identical — improves the average to 58.6, a gain of **+9.2 points**.
This is the largest single contribution, confirming that span detection quality is the primary bottleneck in the original OWNER pipeline.

Adding the soft prompt entity typing on top of DTrans brings the full model to 61.4, a further **+2.8 points**.
While modest in absolute terms, this gain comes at essentially zero parameter cost — only 3,840 parameters versus 110 million for full fine-tuning.

Removing the multi-view detector while keeping the soft prompt — the "without Ensemble MD" row — yields 57.8, which is 3.6 points below the full model.
This confirms that both components contribute **independently**: the gains are not redundant and removing either one causes a measurable drop.

Finally, I note that the total end-to-end AMI gain of 12.0 points exceeds the sum of the individual ablation gains, which is consistent with the cascade effect I hypothesized: better spans produce cleaner embeddings, which produce more coherent clusters, compounding throughout the pipeline.

> **Tip:** Use the waterfall diagram on the right to walk through the additive contributions: 49.4 → +9.2 → 58.6 → +2.8 → 61.4.

---

## Section 7 — Discussion

### Slide 24 · Key Findings & Insights
**⏱ ~3 min**

Let me share four analytical findings that go beyond the headline numbers.

**Finding 1 — Source corpus style matters more than size.**
Span2Type trained on CoNLL-2003 outperforms the Pile-NER variant by 6 points on average, despite Pile-NER being much larger and more diverse.
CrossNER's Wikipedia-style text is syntactically closer to CoNLL-2003 newswire than to Pile-NER's heterogeneous web crawl.
This is a practical warning: a larger source corpus is not always better for cross-domain transfer — stylistic alignment between source and target matters at least as much as vocabulary breadth.

**Finding 2 — Why LLM-prompting baselines underperform.**
UniNER and ChatIE issue a single prompt per sentence asking the LLM to simultaneously detect and type entities.
They suffer three structural weaknesses: no dedicated span detection stage trained discriminatively, inconsistent type strings from open-vocabulary generation, and a fundamental mismatch with the AMI metric which measures agreement on a gold span partition.
Span2Type's separation of detection, embedding, and clustering into dedicated stages avoids all three weaknesses.

**Finding 3 — When MMR hurts.**
The Music domain is the exception where OWNER random selection (0.957) outperforms MMR (0.939).
Music entity clusters are semantically tight and homogeneous — *band*, *genre*, *album* are frequent and unambiguous single-token words.
In this regime, MMR's diversity pressure selects peripheral cluster members that introduce noise rather than broaden coverage.
The practical recommendation is: use MMR for heterogeneous clusters and centroid-nearest for tight, homogeneous clusters.

**Finding 4 — Naming decouples from clustering.**
The near-zero Pearson correlation between cluster purity and BERTScore-F1 means the naming module provides a layer of semantic resilience.
Even when BIC overestimates the number of clusters — which happens frequently in Politics — the LLM can still assign semantically meaningful labels.
This finding has practical value: it means the system remains useful even when the clustering is imperfect.

> **Tip:** This slide has a lot of content. Move briskly through findings 2, 3, and 4 — spend most time on finding 1, which is the most counterintuitive.

---

## Section 8 — Conclusion

### Slide 25 · Summary of Contributions
**⏱ ~1.5 min**

To summarize the three contributions of this thesis:

First, I replaced OWNER's single BIO mention detector with the DTrans multi-view Entity Detection module, jointly training BIO, Start-End, and Tie-Break decoders with Mean Teaching.
This improved span detection F1 from 75.68 to 82.35 — a gain of 6.67 points — with the largest improvement in the AI domain at +20.85 points.

Second, I introduced a two-stage prompt-initialized fine-tuning approach for entity typing, achieving parameter efficiency in early training with only 3,840 trainable parameters, and full-model expressiveness in later epochs.
This contributed +2.8 AMI points in the ablation study.

Third, I proposed MMR-based exemplar selection for automatic cluster naming, producing more diverse and representative evidence for the naming model.
The LLM backend with MMR achieves 0.956 average BERTScore-F1, outperforming random selection by +0.012 in the budget-controlled comparison.

Together, these three improvements bring Span2Type to **61.4 average AMI** on CrossNER — a **12-point improvement** over the OWNER baseline of 49.4.

---

### Slide 26 · Limitations & Future Work
**⏱ ~1.5 min**

I want to close with an honest assessment of what Span2Type does not solve.

The entity detector still requires a labeled source domain.
When the source and target domains are very distant — for example, newswire to clinical text — transfer quality degrades significantly.
Future work could explore domain-adaptive pretraining or self-training to bridge larger domain gaps without requiring target labels.

BIC-guided k-means can overestimate k for domains with highly imbalanced type distributions, as seen in Politics.
A hierarchical or density-based clustering approach could produce entity type hierarchies rather than flat partitions, which is much closer to how practitioners think about entity ontologies.

The MLM naming backend is structurally limited to single tokens, and the LLM backend requires a locally hosted model.
Improving the post-processing and filtering pipeline, or exploring more efficient LLM serving, remains an open problem.

Finally, Span2Type has only been evaluated on CrossNER.
The biomedical and legal domains, where the need for open-world entity type discovery is most acute, are compelling targets for future evaluation.

> **Tip:** Keep the tone positive but honest. The committee will appreciate that you can critically evaluate your own work.

---

### Slide 27 · Thank You
**⏱ ~30 sec**

Thank you very much for your attention.
I am happy to take questions on any aspect of the methodology, the experimental results, or the design decisions behind Span2Type.
My contact information is on the slide if you would like to follow up after the defense.

> **Tip:** Stop talking. Smile. Let the silence work. The committee needs a moment to formulate questions. Do not fill the silence with filler words.

---

## Appendix Slides — Use Only If Asked

### Appendix A · Naming Examples
Use this slide if the committee asks about **qualitative naming quality** or wants to see specific prediction examples.

> "The table shows examples from all five CrossNER domains. The LLM backend successfully generates multi-word labels like *political leader* and *biological protein*, while the MLM backend is constrained to single tokens. The *vegetarian* artefact in the MLM column is the hallucination I mentioned — it occurs when the entity context is out of distribution for BERT's pretraining data."

---

### Appendix B · Literature Per-Cluster Detail
Use this slide if the committee asks about **the best-performing domain** or wants to understand how BERTScore aligns with individual cluster results.

> "Literature achieves the highest domain mean BERTScore of 0.979. Of the 14 predicted clusters, 11 score above 0.99 — near-perfect semantic alignment. The two red-cell failures follow systematic patterns: C8 is a specificity mismatch where *film festival* is a hyponym of the broad gold label *event*, and C13 incurs a tokenization penalty because the gold label *literarygenre* is a compound string with no whitespace."

---

### Appendix C · Why LLM Baselines Underperform
Use this slide if the committee pushes back on the comparison to GPT-4o-based baselines.

> "The structural disadvantage of LLM-prompting baselines is threefold: they lack a discriminatively trained span detector, their open-vocabulary output produces inconsistent type strings that cannot be cleanly clustered, and their extracted spans differ structurally from gold annotation boundaries creating a systematic offset in AMI even when the semantics are correct. Span2Type avoids all three issues by separating detection, embedding, and clustering into dedicated stages."

---

### Appendix D · DTrans Full Hyperparameters
Use this slide if the committee asks about **reproducibility** or specific training settings.

> "All hyperparameters for the DTrans entity detector follow the original paper by Zhang et al. 2024. The learning rate of 1×10⁻⁵, batch size 32, and 50 training epochs are standard for BERT-based sequence taggers. The EMA decay of 0.995 and consistency weight of 100 are taken directly from the DTrans paper's recommended settings for cross-domain transfer."

---

## Anticipated Questions & Suggested Answers

**Q: Why not fine-tune DeBERTa instead of BERT for entity typing?**

> OWNER uses DeBERTa for mention detection and BERT for entity typing — this is the original design choice and I retain it to ensure a fair comparison. DeBERTa's disentangled attention is advantageous for sequence labeling tasks like BIO tagging, but the fill-in-the-blank template is specifically designed around BERT's masked language modeling pretraining objective. Replacing BERT with DeBERTa in the typing stage would be an interesting experiment for future work.

---

**Q: How sensitive is the system to the choice of k_min and k_max in BIC-guided clustering?**

> In my experiments, I use k ∈ [2, 30] with step 2. The CrossNER domains have between 9 and 17 gold types, so the range [2, 30] comfortably covers the true k in all cases. The BIC criterion is robust to the upper boundary as long as the true k is well within the search range. A practical recommendation is to set k_max to at least twice the expected number of entity types.

---

**Q: Is the soft prompt approach comparable to LoRA?**

> Soft prompt tuning and LoRA are both parameter-efficient fine-tuning methods, but they operate differently. Soft prompts prepend trainable tokens to the input and keep all model weights frozen; LoRA inserts low-rank adapter matrices into the attention layers and updates those adapters. LoRA typically achieves better performance at the cost of slightly more parameters — on the order of millions rather than thousands. Extending the entity typing stage with LoRA in place of soft prompts is an explicit future work direction I mention in the conclusion.

---

**Q: Why does Span2Type (Pile-NER) underperform Span2Type (CoNLL)?**

> This was a counterintuitive finding. The CrossNER target domains consist primarily of Wikipedia-style formal sentences about specific topics. CoNLL-2003 newswire text has similar syntax, capitalization conventions, and sentence structure to Wikipedia, so the span detector transfers well. Pile-NER is a much larger corpus but drawn from heterogeneous web crawl text that is noisier and more colloquial — this style mismatch slightly degrades span boundary precision on the formal CrossNER sentences. The takeaway is that source-target stylistic alignment may matter more than raw vocabulary coverage for cross-domain span detection.

---

**Q: What is the inference time of Span2Type?**

> I did not conduct a systematic inference time benchmark in this thesis, which is a limitation I acknowledge. The bottleneck is the LLM naming backend — querying Llama-3 locally for each cluster adds latency that scales with the number of clusters. The MLM backend is significantly faster since it is a single forward pass per exemplar. For production deployment, the MLM backend would be preferred for latency-sensitive applications, and the LLM backend reserved for offline analysis where naming quality is paramount.

---

*End of speaker notes.*
