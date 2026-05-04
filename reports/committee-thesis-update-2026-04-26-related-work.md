# Thesis Revision Report for Committee (Related Work)

**Student:** Nguyen Anh Khoa (M1261026)  
**Program:** Graduate Institute of Artificial Intelligence, Chang Gung University  
**Date:** 2026-04-26  
**Subject:** Post-submission update on Related Work chapter

## 1) Purpose of This Update

After submitting the thesis PDF to the committee, I revised the **Related Work** chapter to strengthen the literature grounding of the Span2Type contributions.

This update is additive and clarification-oriented: it improves background coverage and positioning against relevant prior work.

## 2) Scope of Revisions

The update is limited to:

- `WIN-thesis-template/WIN-thesis.tex` (Related Work sections)
- `WIN-thesis-template/refs-bib.bib` (added references used by the new text)

No methodological, experimental, or results sections were changed.

## 3) Main Changes in Related Work

I expanded all four Related Work sections with targeted additions:

- **Section 2.1 (Named Entity Recognition):** Added ultra-fine entity typing and GLiNER context to better justify open-world assumptions and clarify the distinction between user-specified open-vocabulary NER and fully unsupervised open-world NER.
- **Section 2.2 (Multi-View Span Detection):** Added supporting context for overlapping/nested span handling in open-world settings.
- **Section 2.3 (Parameter-Efficient Entity Typing):** Added comparison with adapter-based PEFT and LoRA, and clarified why soft prompts were selected in this thesis.
- **Section 2.4 (Automatic Cluster Naming):** Strengthened exemplar-selection motivation with in-context learning evidence and clarified MMR historical origin (summarization).

## 4) New References Added

Four new works were added to support the above expansions:

1. Choi et al., *Ultra-Fine Entity Typing* (ACL 2018)  
2. Zaratiana et al., *GLiNER* (NAACL 2024)  
3. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* (ICML 2019)  
4. Li et al., *A Unified MRC Framework for NER* (ACL 2020)

Additionally, LoRA (already present in bibliography before) is now cited directly in Related Work for completeness.

## 5) What Was Not Changed

- Introduction, Methodology, Experiments, and Conclusion chapters
- Any tables, numeric results, or reported metrics
- Main thesis claims and conclusions

## 6) Impact Statement

This revision improves scholarly completeness and framing clarity of the thesis without altering technical contributions or empirical findings.  
In short, the update strengthens the literature foundation while preserving all original results and conclusions.

