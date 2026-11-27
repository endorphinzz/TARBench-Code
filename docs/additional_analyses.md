# Additional analyses and released predictions

This page indexes the per-sample predictions and the analysis scripts added after submission.
Every number below is produced by the listed script from files in this repository.

## 1. Per-sample predictions

`experiments/{icl,rag,token_compression}/outputs/` hold one prediction file per run for all
ICL, RAG and context-compression runs (27 runs, plus the three DeepSeek RAG shards), together
with the stored `*_metrics.json`. Each record keeps the sample ID, label, ground truth,
retrieved document references and the model's prediction and raw reply. Prompt contexts and
rule texts are omitted; they are rebuilt from the dataset, the prompt file and the ICL library
or knowledge base. Metrics regenerate from these files:

```
python experiments/icl/evaluate_icl.py <prediction file>
python experiments/rag/evaluate_rag.py <prediction file>
python experiments/token_compression/evaluate_token_compression.py <prediction file>
```

The LLM rows of the main table come from a run on the full 15,183-sample benchmark whose
per-sample outputs were not retained; `experiments/baselines/compare_llm_vs_symbolic.py`
carries them as constants (`PAPER_MAIN`).

## 2. Runs added on 2026-11-26

| Directory | Content |
|---|---|
| `experiments/icl/outputs/stylematch_ctrl/` | gpt-5.4 and gemini-3.1-flash-lite, zero-shot, on the 750-sample ICL set |
| `experiments/icl/outputs/stylematch/` | the same two models on the restyled set (Section 4) |
| `experiments/rag/outputs/same_session/` | no-retrieval baselines for gpt-5.4, claude-sonnet-4-5 and deepseek-v4-flash; deepseek-v4-flash with the full knowledge base and with the definitions-only base |
| `experiments/rag/outputs/nocase/` | gpt-5.4 and claude-sonnet-4-5 with the definitions-only knowledge base |

All use `run_icl_benchmark.py` / `run_rag_benchmark.py` with temperature 0. The definitions-only
knowledge base (`experiments/rag/knowledge_base_no_cases/`, 22 documents) is built by
`experiments/rag/build_kb_no_cases.py` and selected with `--kb-file`; it holds only the type
definitions and contrastive guidelines, so no stored case can be retrieved.

## 3. Analyses

| Question | Script | Output |
|---|---|---|
| Can a rule-level lexical model localize the inserted fragment? | `analysis/lexical_localizer_probe.py` | `analysis/outputs/lexical_localizer_probe.json` |
| Does the lexical cue survive when fragment templates are withheld from training? | `analysis/template_disjoint_probe.py` | `analysis/outputs/template_disjoint_probe.json` |
| How much of the lexical separability do the three salient cues carry? | `analysis/cue_ablation.py` | `analysis/outputs/cue_ablation.json` |
| Which structural quantities separate fragments from benign rules? | `analysis/structure_probe.py` | `analysis/outputs/structure_probe.json` |
| Do Loop fragments oscillate under the semantics of Section 3.1? | `analysis/execution_semantics_audit.py`, `analysis/loop_spec_check.py` | `analysis/outputs/execution_semantics_audit.json`, `loop_spec_check.json` |
| Do annotators agree with the released labels at the binary level? | `analysis/iaa_binary_agreement.py` | `analysis/outputs/iaa_binary_agreement.json` |
| Do results change when Condition Block is excluded? | `analysis/sensitivity_condition_block.py` | `analysis/outputs/sensitivity_condition_block.json` |
| How often does the RAG retriever return the evaluated fragment? | `analysis/rag_leakage_realized.py` | `analysis/outputs/rag_leakage_realized.json` |
| What changes without stored cases in the knowledge base? | `analysis/rag_decompose.py` | `analysis/outputs/rag_decomposition.json` |
| Restyled test set and paired comparison | `experiments/style_match/` (pipeline; results in `data/`) | `data/stylematched_500_250.json`, `data/paired_report.json` |

`experiments/baselines/score_tess_main.py` scores the explanations of any ICL/RAG prediction
file under the TESS rubric used in the paper.

## 4. Restyled test set

`experiments/style_match/` restyles the 250 anomalous fragments of the 750-sample ICL set to the
host wording and device vocabulary. One model rewrites each fragment under semantic-preserving
constraints (`run_paraphrase.py`); a second, independent model verifies semantic equivalence
(`run_verify.py`, with a repair round in `repair_round.py`). 229 fragments pass and are inserted
in place (`build_stylematched.py`); the other 21 keep their original rules. `data/` holds every
rewrite with its verdict and reason.

## 5. Results

- Lexical localizer, paper's ARL-IoU protocol: 0.973 under 5-fold CV; 0.967 on the
  template-disjoint split. Fragments and hosts share no device in 4,599 of 4,600 anomalous samples.
- Lexical AUC of the probe: 0.996 (5-fold CV), 0.995 (template-disjoint), 0.994 (device-sharing
  variants, paper), 0.997 (restyled fragments), 0.990 (three salient cues removed).
- 37.4% of the device-attribute pairs that trigger fragment rules never occur as a trigger in a
  benign rule; 64.1% of fragment rules and 13.9% of benign rules are triggered by a state another
  rule sets.
- Prompted models on the restyled set, same 750 samples and session: gpt-5.4 F1 0.678 to 0.676,
  ARL-IoU 0.575 to 0.580; gemini-3.1-flash-lite F1 0.627 to 0.630, ARL-IoU 0.491 to 0.490.
- RAG without stored cases (FPR, then recall): gpt-5.4 0.082/0.772 (no retrieval), 0.016/0.640
  (definitions and guidelines only), 0.004/0.780 (full base); claude-sonnet-4-5 0.092/0.852,
  0.058/0.860, 0.028/0.904; deepseek-v4-flash 0.056/0.820, 0.016/0.796, 0.006/0.924.
- The retriever returns the complete evaluated fragment for 9.9% of the 1,000 RAG evaluation
  anomalies and at least one of its rules for 56.5%.
- Excluding Condition Block changes F1 by at most 0.025 across the 15 per-sample runs in
  `sensitivity_condition_block.py`.
- Binary-level agreement in the annotation study: Condition Bypass 36 of 38 flagged by majority;
  Loop 38 of 38; Condition Block 0 of 11.
