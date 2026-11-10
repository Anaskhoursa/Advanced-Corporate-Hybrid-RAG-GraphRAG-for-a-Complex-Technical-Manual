# Hybrid RAG + GraphRAG Evaluation Report

- Generated: 2026-11-10T14:40:24
- Gold set: `gold_set.json`
- Cache: `eval_cache.json`
- Runs: 116 (58 natural + 58 baseline)

## 1. System reliability

- Questions run / total: **58 / 58**
- Failed questions: **0**
- Non-empty answer rate: **100.0%**
- Error rate: **0.00%**

## 2. Intent classification

- Accuracy: **0.810**
- Macro precision: **0.866**
- Macro recall: **0.864**
- Macro F1: **0.860**
- Out-of-scope precision: **100.0%**

| Intent | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| comparison | 2 | 1.000 | 1.000 | 1.000 |
| factual | 26 | 0.833 | 0.769 | 0.800 |
| out_of_scope | 2 | 1.000 | 1.000 | 1.000 |
| procedural | 13 | 1.000 | 0.846 | 0.917 |
| safety | 7 | 0.364 | 0.571 | 0.444 |
| troubleshoot | 8 | 1.000 | 1.000 | 1.000 |

*Per-class F1s with support below 10 should be treated as anecdotal.*

## 3. Retrieval quality (natural graph_weight per question)

Evaluated questions: **56**

| Cut-off | Hit rate | Precision | Recall | MRR | nDCG |
|---|---:|---:|---:|---:|---:|
| k=5 | 0.946 | 0.257 | 0.920 | 0.822 | 0.824 |
| k=8 | 0.946 | 0.167 | 0.935 | 0.822 | 0.831 |

## 4. Grounding and safety

- Verification pass rate: **100.0%** (58/58)
- Mean verifier issues per question: **0.00**
- Safety-critical questions: **7**
- Safety-unverified: **0**
- Safety-unverified rate: **0.0%**
- Retry rate: **6.9%**
- Query reformulation rate: **3.4%**

## 5. Workflow

- Retry rate: **6.9%**
- Query reformulation rate: **3.4%**
- Graph usage rate: **46.6%**
