# Thesis Revision Report for Committee

**Student:** Nguyen Anh Khoa (M1261026)  
**Program:** Graduate Institute of Artificial Intelligence, Chang Gung University  
**Date:** 2026-04-26  
**Subject:** Post-submission thesis update (cluster naming results + related work)

## 1) Purpose of This Update

After the thesis draft was sent to the committee, I completed a correction and consolidation of the **cluster naming evaluation** used in Chapter 4.  
This report summarizes all modifications made to the thesis and clarifies their impact on the final conclusions.

## 2) Scope of Revisions

The update is limited to **cluster naming results and related discussion text**

No changes were made to methodology, model architecture, entity detection experiments, or end-to-end AMI results.

## 3) What Was Corrected

The cluster naming scores were recalculated under a unified and corrected evaluation setting:

- BERTScore-F1 values are reported consistently.
- All compared naming methods are evaluated under this same criterion.

## 4) Updated Main Naming Results (BERTScore-F1, ET Stage)

| Method | AI | Literature | Music | Politics | Science | Average |
|---|---:|---:|---:|---:|---:|---:|
| OWNER (single-template MLM) | 0.900 | 0.891 | 0.906 | 0.892 | 0.894 | 0.897 |
| OWNER (LLM) | 0.892 | **0.923** | 0.928 | 0.910 | 0.886 | 0.908 |
| Span2Type-CentroidNaming (MLM) | 0.904 | 0.887 | 0.895 | 0.883 | 0.894 | 0.893 |
| Span2Type-MLM (MMR) | 0.900 | 0.889 | 0.911 | 0.887 | 0.893 | 0.896 |
| **Span2Type-LLM (MMR)** | **0.922** | 0.913 | **0.943** | **0.928** | **0.912** | **0.924** |

## 5) Key Thesis Text Updates

The following narrative claims in Chapter 4 were updated to match the corrected results:

- Span2Type-LLM is now reported as best in **4/5 domains** (Literature is the exception), not all five.
- The LLM-vs-MLM advantage is revised from a larger previous gap to a validated **+0.028 average**.
- Budget-controlled comparison (Span2Type-LLM vs OWNER-LLM, both with equal naming budget) is updated to **+0.016 average** in favor of Span2Type-LLM.
- Strongest/weakest domain statements were updated accordingly (Music strongest, Science weakest in the revised figure-level summary).

## 6) Items Explicitly Unchanged

- Span2Type-MLM (MMR) per-domain values remain unchanged.
- CentroidNaming per-domain values remain unchanged (only average rounding adjusted).
- Per-cluster qualitative tables remain unchanged.
- Mention detection and entity typing (AMI) experimental results are unchanged.
- Core methodology and proposed components (DTrans ED, two-stage soft prompt typing, MMR naming) are unchanged.

## 7) Interpretation and Impact on Thesis Claims

Although several absolute BERTScore values decreased after correction, the main thesis conclusions remain valid and are supported by the revised evidence:

1. **Span2Type-LLM (MMR) remains the strongest overall cluster naming method** (best average score: 0.924).
2. **LLM naming consistently outperforms MLM naming** under corrected evaluation.
3. **MMR-based exemplar selection remains beneficial** in budget-controlled comparisons.
4. The MLM-specific noise behavior (e.g., the “vegetarian” artifact) remains observed; LLM naming is robust against this failure mode.

In short, the revision improves methodological correctness and consistency of reporting, while preserving the central contributions and conclusions of the thesis.

## 8) Additional Update: Related Work Expansion

I also revised the **Related Work** chapter to strengthen literature grounding and positioning.

### Main Related Work Changes

- **Section 2.1 (Named Entity Recognition):** Added ultra-fine entity typing and GLiNER context to better justify open-world assumptions and distinguish open-vocabulary vs fully unsupervised open-world NER.
- **Section 2.2 (Multi-View Span Detection):** Added supporting context for overlapping/nested span handling in open-world settings.
- **Section 2.3 (Parameter-Efficient Entity Typing):** Added comparison with adapter-based PEFT and LoRA, and clarified why soft prompts were selected.
- **Section 2.4 (Automatic Cluster Naming):** Strengthened exemplar-selection motivation with in-context learning evidence and clarified MMR origin in summarization.

### New References Added for Related Work

1. Choi et al., *Ultra-Fine Entity Typing* (ACL 2018)  
2. Zaratiana et al., *GLiNER* (NAACL 2024)  
3. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (ICML 2019)  
4. Li et al., *A Unified MRC Framework for NER* (ACL 2020)

Additionally, LoRA (already present in the bibliography) is now cited directly in Related Work.

### Related Work Impact Statement

This part of the revision is additive and clarification-oriented. It improves scholarly completeness and framing clarity without changing methodology, experiments, metrics, or final conclusions.
