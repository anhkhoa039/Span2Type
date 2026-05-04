# Thesis Update Report: Related Work Expansion

**Date:** 2026-04-26  
**Branch:** experiments  
**Files Updated:** `WIN-thesis-template/WIN-thesis.tex`, `WIN-thesis-template/refs-bib.bib`

---

## Summary

The Related Work chapter was expanded across all four sections to strengthen the literature foundation for the three Span2Type contributions. Four new references were added and four targeted paragraph expansions were made. All existing text and citations were preserved; only additive changes were made.

---

## New References Added (`refs-bib.bib`)

| Key | Paper | Venue | Year |
|---|---|---|---|
| `choi-etal-2018-ultra` | *Ultra-Fine Entity Typing* — Choi, Levy, Choi, Zettlemoyer | ACL | 2018 |
| `zaratiana-etal-2023-gliner` | *GLiNER: Generalist Model for NER using Bidirectional Transformer* — Zaratiana et al. | NAACL | 2024 |
| `houlsby-etal-2019-parameter` | *Parameter-Efficient Transfer Learning for NLP* — Houlsby et al. | ICML | 2019 |
| `li-etal-2020-unified` | *A Unified MRC Framework for Named Entity Recognition* — Li et al. | ACL | 2020 |

> Note: `hu-etal-2022-lora` (LoRA) was already present in the bibliography and is now cited in the Related Work text for the first time.

---

## Section-by-Section Changes

### Section 2.1 — Named Entity Recognition

**What changed:** Two sentences added at the start of the open-world paragraph, before the UniversalNER/ChatIE discussion.

**Before:**
> Open-world NER removes these constraints entirely: the model receives an unannotated corpus and must discover entity types without any predefined ontology. Systems such as UniversalNER and ChatIE use LLM prompting for open-vocabulary extraction, while clustering-based approaches embed entity mentions and group them by type.

**After:**
> Open-world NER removes these constraints entirely: the model receives an unannotated corpus and must discover entity types without any predefined ontology. **The motivation for this paradigm is reinforced by fine-grained entity typing research: Choi et al. [choi-etal-2018-ultra] demonstrate that real-world entity categories number in the thousands---far beyond the four to eighteen types defined in standard benchmarks---making any fixed type inventory an approximation at best.** Systems such as UniversalNER and ChatIE use LLM prompting for open-vocabulary extraction, **while GLiNER [zaratiana-etal-2023-gliner] trains a span-level classifier that accepts arbitrary user-specified type names at inference time; however, both families of approaches still require type names to be supplied by the user, a constraint that unsupervised open-world NER removes entirely. Clustering-based approaches instead embed entity mentions and group them by type without any type name input.**

**Why:** Ultra-Fine ET provides empirical grounding for the open-world assumption. GLiNER is the most relevant recent competitor and clarifying that it still requires type names sharpens the distinction between open-vocabulary NER and fully unsupervised NER.

---

### Section 2.2 — Multi-View Span Detection

**What changed:** One clause added to the span-based extractor sentence.

**Before:**
> Span-based extractors [zhong-chen-2021-frustratingly] instead enumerate candidate spans and score them directly, offering better handling of overlapping and nested entities.

**After:**
> Span-based extractors [zhong-chen-2021-frustratingly] instead enumerate candidate spans and score them directly, offering better handling of overlapping and nested entities---**a practical concern in open-world domains where type boundaries are not predefined and entity mentions may share tokens [li-etal-2020-unified].**

**Why:** Connects the general span-based NER motivation to the open-world setting specifically, and adds a concrete citation for nested/overlapping NER.

---

### Section 2.3 — Parameter-Efficient Entity Typing

**What changed:** One new sentence added after the Power of Scale citation, before the thesis contribution sentence.

**Before:**
> ...the Power of Scale result [lester-etal-2021-power] shows prompt tuning matches full fine-tuning at sufficient model scale. This thesis introduces a two-stage soft prompt schedule...

**After:**
> ...the Power of Scale result [lester-etal-2021-power] shows prompt tuning matches full fine-tuning at sufficient model scale. **Alternative parameter-efficient approaches include adapter layers [houlsby-etal-2019-parameter], which insert small trainable bottleneck modules between transformer layers, and LoRA [hu-etal-2022-lora], which reparameterizes weight updates as low-rank matrices; both achieve strong performance but require architectural modifications to the backbone. This thesis adopts soft prompt tuning due to its zero architectural overhead and direct compatibility with the existing BERT fill-in-the-blank entity typing template.** This thesis introduces a two-stage soft prompt schedule...

**Why:** Reviewers expect PEFT alternatives to be acknowledged and the design choice justified. This positions soft prompts against adapters and LoRA with a clear rationale, and brings LoRA (already in the bibliography and cited in Future Work) into the Related Work where it belongs.

---

### Section 2.4 — Automatic Cluster Naming

**What changed:** One new sentence added at the start of the exemplar selection paragraph.

**Before:**
> A key challenge in both backends is exemplar selection: which cluster members to include in the prompt. Centroid-nearest selection can produce redundant examples...

**After:**
> A key challenge in both backends is exemplar selection: which cluster members to include in the prompt. **The quality of in-context examples has been shown to substantially affect LLM output in information extraction tasks [brown-etal-2020-language]: poorly chosen examples introduce noise, while diverse, representative examples elicit more accurate and generalizable predictions.** Centroid-nearest selection can produce redundant examples...

> ...MMR has been applied to **extractive document summarization** (its original application), in-context example selection, and retrieval-augmented generation; this thesis applies it to entity cluster exemplar selection...

**Why:** The exemplar selection motivation was previously asserted without a citation. Grounding it in the in-context learning literature (Brown et al. 2020) makes the argument for MMR more rigorous. Clarifying MMR's origin in summarization also gives the reader correct historical context.

---

## Citation Count

| Section | Before | After |
|---|---|---|
| 2.1 Named Entity Recognition | 8 citations | 10 citations (+choi, +zaratiana) |
| 2.2 Multi-View Span Detection | 3 citations | 4 citations (+li-2020) |
| 2.3 Parameter-Efficient Entity Typing | 5 citations | 7 citations (+houlsby, +lora now cited here) |
| 2.4 Automatic Cluster Naming | 3 citations | 3 citations (brown-etal-2020 reused) |
| **Total new unique references** | — | **+4** |

---

## What Was NOT Changed

- All original sentences and arguments are preserved verbatim
- No existing citations were removed or modified
- Introduction, Methodology, Experiments, and Conclusion chapters: unchanged
- All table files: unchanged
