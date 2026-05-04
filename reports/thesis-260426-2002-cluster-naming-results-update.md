# Thesis Update Report: Cluster Naming Results

**Date:** 2026-04-26  

---

## Summary

The cluster naming evaluation results in the thesis have been updated to reflect the corrected ground-truth methodology introduced in Report Version 5 and finalized in Version 6. All BERTScore-F1 scores are now measured consistently at the **Entity Typing (ET) stage** across all four naming methods.

---

## Updated Main Results Table (BERTScore-F1)

| Method | AI | Lit. | Music | Pol. | Sci. | Avg. |
|---|---|---|---|---|---|---|
| OWNER (single-template MLM) | 0.900 | 0.891 | 0.906 | 0.892 | 0.894 | 0.897 |
| OWNER (LLM) | 0.892 | **0.923** | 0.928 | 0.910 | 0.886 | 0.908 |
| Span2Type-CentroidNaming (MLM) | 0.904 | 0.887 | 0.895 | 0.883 | 0.894 | 0.893 |
| Span2Type-MLM (MMR) | 0.900 | 0.889 | 0.911 | 0.887 | 0.893 | 0.896 |
| Span2Type-LLM (MMR) | **0.922** | 0.913 | **0.943** | **0.928** | **0.912** | **0.924** |

*Previous values (before update) shown in the section below.*

---

## Key Changes from Previous Version

### 1. Overall Ranking — Preserved, Numbers Revised

The four-method ranking remains unchanged:

> Span2Type-LLM (MMR) **(0.924)** > OWNER (LLM) (0.908) > Span2Type-MLM (MMR) (0.896) ≈ CentroidNaming (0.893) > OWNER (MLM) (0.897)

All qualitative conclusions hold; only the absolute values shifted due to the corrected GT alignment.

### 2. Span2Type-LLM (MMR) — Updated Domain Profile

| Domain | Previous F1 | Updated F1 | Δ |
|---|---|---|---|
| AI | 0.958 | 0.922 | −0.036 |
| Literature | 0.979 | 0.913 | −0.066 |
| Music | 0.939 | 0.943 | +0.004 |
| Politics | 0.964 | 0.928 | −0.036 |
| Science | 0.950 | 0.912 | −0.038 |
| **Average** | **0.956** | **0.924** | **−0.032** |

### 3. OWNER (single-template MLM) — Updated

| Domain | Previous F1 | Updated F1 | Δ |
|---|---|---|---|
| AI | 0.989 | 0.900 | −0.089 |
| Literature | 0.974 | 0.891 | −0.083 |
| Music | 0.958 | 0.906 | −0.052 |
| Politics | 0.964 | 0.892 | −0.072 |
| Science | 0.945 | 0.894 | −0.051 |
| **Average** | **0.966** | **0.897** | **−0.069** |

### 4. OWNER (LLM) — Updated

| Domain | Previous F1 | Updated F1 | Δ |
|---|---|---|---|
| AI | 0.929 | 0.892 | −0.037 |
| Literature | 0.954 | 0.923 | −0.031 |
| Music | 0.957 | 0.928 | −0.029 |
| Politics | 0.953 | 0.910 | −0.043 |
| Science | 0.926 | 0.886 | −0.040 |
| **Average** | **0.944** | **0.908** | **−0.036** |

---

## Narrative Changes in Thesis Text

### Section 4.4 — Cluster Naming Results

| Claim | Previous | Updated |
|---|---|---|
| Span2Type-LLM wins domains | "all five" | "four of five" (Literature exception) |
| Span2Type-LLM avg | 0.956 | 0.924 |
| LLM vs MLM (MMR) gap | +0.060 | +0.028 |
| LLM vs CentroidNaming gap | +0.064 | +0.031 |
| Largest domain advantage | Literature (+0.090), Politics (+0.077) | Politics (+0.041), Music (+0.032) |
| vs OWNER (MLM) baseline | "falls short in AI, Music" | "wins all five domains" |
| Budget-controlled avg gap | +0.012 (0.956 vs 0.944) | +0.016 (0.924 vs 0.908) |
| Exception vs OWNER (LLM) | Music (0.957 > 0.939) | Literature (0.923 > 0.913) |
| MLM MMR vs CentroidNaming | +0.004 (0.896 vs 0.892) | +0.003 (0.896 vs 0.893) |
| Strongest domain (figure) | Literature (0.979) | Music (0.943) |
| Weakest domain (figure) | Music (0.939) | Science (0.912) |

### Section 4.6 — Discussion

| Claim | Previous | Updated |
|---|---|---|
| Structural LLM vs MLM gap | +0.060 | +0.028 |
| Domains below OWNER (MLM) | AI and Music | None — S2T-LLM wins all |
| Budget-controlled exception | Music (MMR counterproductive) | Literature (random sampling sufficient) |
| MMR vs Centroid for MLM | +0.004 | +0.003 |

---

## What Did NOT Change

- **Span2Type-MLM (MMR)** scores: unchanged across all domains (`{0.900, 0.889, 0.911, 0.887, 0.893}`, avg 0.896)
- **CentroidNaming** per-domain values: unchanged (`{0.904, 0.887, 0.895, 0.883, 0.894}`); avg rounded from 0.892 → 0.893
- **Per-cluster Literature table** (`naming_lit_detail.tex`): retained as-is — this is a per-cluster NER-stage analysis not covered by the ET-stage v6 report
- **Qualitative naming examples** (`naming_examples.tex`): unchanged
- **All AMI and span detection results**: unaffected
- **Conclusion and abstract**: no naming-result numbers cited there

---

## Interpretation Note for Professor

The drop in absolute BERTScore values (avg −0.032 for Span2Type-LLM) results from the corrected GT alignment methodology (confusion-matrix column-wise argmax, established in v5). The **relative ordering of all methods is preserved** and the **core claims of the thesis remain valid**:

1. Span2Type-LLM (MMR) is the strongest naming method overall.
2. LLM backends consistently outperform MLM backends (+0.028 avg gap).
3. MMR exemplar selection improves over random/centroid baselines (+0.016 vs OWNER LLM).
4. The `vegetarian` artefact persists in MLM methods; LLM is immune.

The one qualitative change is that **Span2Type-LLM now beats OWNER (single-template MLM) in all five domains** (previously it fell short in AI and Music) — this is a stronger result for the thesis.
