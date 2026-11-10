"""
Comprehensive evaluation for the HL Series Pump Manual GraphRAG system.

Reads the cache produced by `evaluation.py --batch`. Computes:

  §1 System reliability
  §2 Intent classification (accuracy, macro P/R/F1, per-class, out-of-scope)
  §3 Retrieval quality (hit@k, precision@k, recall@k, MRR@k, nDCG@k)
  §4 Graph contribution (natural vs. graph_weight=0.0 delta, gold attribution)
  §5 Grounding and safety
  §6 Reference-answer quality (EM, token F1, ROUGE-L) + fact recall
  §7 Workflow

Every question has one "natural" run and one "baseline" (gw=0.0) run in
the cache. Metrics use the natural run; §4 compares natural vs baseline.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Label Mapping (fixes the schema mismatch)
# ---------------------------------------------------------------------------

GOLD_TO_PRED = {
    "followup":   "factual",
    "global":     "factual",
    "off_topic":  "out_of_scope",
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gold(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["id"]: item for item in data}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def fmt(v, spec=".3f"):
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def normalize_text(text):
    return " ".join(str(text or "").lower().split())


def tokenize(text):
    return re.findall(r"\b\w+\b", normalize_text(text))


def token_f1(prediction, reference):
    pred = Counter(tokenize(prediction))
    ref = Counter(tokenize(reference))
    if not pred or not ref:
        return float(not pred and not ref)
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return float(normalize_text(prediction) == normalize_text(reference))


def rouge_l(prediction, reference):
    a = tokenize(prediction)
    b = tokenize(reference)
    if not a or not b:
        return float(not a and not b)
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    lcs = previous[-1]
    p = lcs / len(a)
    r = lcs / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def fact_recall(answer: str, required_facts: list) -> float | None:
    if not required_facts:
        return None
    ans = normalize_text(answer)
    hit = 0
    for fact in required_facts:
        if isinstance(fact, (list, tuple)):
            alternatives = [normalize_text(f) for f in fact]
        else:
            alternatives = [normalize_text(fact)]
        if any(alt and alt in ans for alt in alternatives):
            hit += 1
    return hit / len(required_facts)


# ---------------------------------------------------------------------------
# §2 Intent classification
# ---------------------------------------------------------------------------

def classification_metrics(rows):
    labelled = [r for r in rows
                if r.get("expected_intent") and r.get("intent") not in ("?", None)]
    if not labelled:
        return {"count": 0, "accuracy": None, "macro_precision": None,
                "macro_recall": None, "macro_f1": None,
                "per_class": {}, "confusion_matrix": {}}

    labels = sorted({r["expected_intent"] for r in labelled}
                    | {r["intent"] for r in labelled})
    confusion = {a: {p: 0 for p in labels} for a in labels}
    for r in labelled:
        confusion[r["expected_intent"]][r["intent"]] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[a][label] for a in labels if a != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        per_class[label] = {"support": sum(confusion[label].values()),
                            "precision": p, "recall": rec, "f1": f1}

    count = len(labelled)
    accuracy = sum(r["intent"] == r["expected_intent"] for r in labelled) / count
    gold_labels = [l for l in labels if per_class[l]["support"] > 0]

    return {
        "count": count,
        "accuracy": accuracy,
        "macro_precision": safe_mean([per_class[l]["precision"] for l in gold_labels]),
        "macro_recall": safe_mean([per_class[l]["recall"] for l in gold_labels]),
        "macro_f1": safe_mean([per_class[l]["f1"] for l in gold_labels]),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# §3 Retrieval ranking
# ---------------------------------------------------------------------------

def retrieval_metrics(retrieved_ids, relevant_ids, k):
    relevant = set(str(x) for x in (relevant_ids or []))
    if not relevant:
        return None
    ranked = [str(x) for x in list(retrieved_ids or [])[:k]]
    hits = [1 if cid in relevant else 0 for cid in ranked]
    hit_count = sum(hits)

    rr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            rr = 1.0 / rank
            break

    dcg = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

    return {
        f"hit@{k}": float(hit_count > 0),
        f"precision@{k}": hit_count / k,
        f"recall@{k}": hit_count / len(relevant),
        f"mrr@{k}": rr,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"gold_retrieved@{k}": hit_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/eval_cache.json")
    parser.add_argument("--gold", default="data/gold_set.json")
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 8])
    parser.add_argument("--out-dir", default="evaluation_results")
    args = parser.parse_args()

    args.top_k = sorted(set(k for k in args.top_k if k > 0))
    if not args.top_k:
        parser.error("--top-k must contain at least one positive integer.")

    cache = load_cache(args.cache)
    gold_by_id = load_gold(args.gold)

    # Re-key cache entries: one row per (gold_id, condition).
    rows_all: list[dict] = []
    for entry in cache["entries"].values():
        if entry.get("error"):
            continue
        gid = entry.get("gold_id")
        gold = gold_by_id.get(gid, {})
        row = {**entry}
        row["relevant_chunk_ids"] = gold.get("expected_chunks") or []
        row["required_facts"] = gold.get("required_facts") or []
        
        # FIX: Apply the label mapping here
        raw_cat = gold.get("category")
        row["expected_intent"] = GOLD_TO_PRED.get(raw_cat, raw_cat)
        
        row["answerable"] = bool(gold.get("answerable", True))
        row["human_verified"] = bool(gold.get("verified", False))
        rows_all.append(row)

    # Natural-condition rows = the ones metrics are computed on.
    rows_natural = [r for r in rows_all if r.get("condition") == "natural"]
    rows_baseline = [r for r in rows_all if r.get("condition") == "baseline"]

    # If condition field is missing (old cache), fall back: pick the row
    # with the highest graph_weight per gold_id.
    if not rows_natural:
        by_qid: dict[str, list[dict]] = {}
        for r in rows_all:
            by_qid.setdefault(r.get("gold_id"), []).append(r)
        for runs in by_qid.values():
            runs.sort(key=lambda x: -float(x.get("graph_weight", 0.0)))
            rows_natural.append(runs[0])
            rows_baseline.extend(runs[1:])

    # ---- §1 Reliability (natural runs) ----
    n_nat = len(rows_natural)
    reliability = {
        "questions_run": n_nat,
        "questions_total": n_nat,
        "failed_questions": sum(1 for r in rows_natural if r.get("error")),
        "nonempty_answer_rate": (
            sum(1 for r in rows_natural if (r.get("answer") or "").strip()) / n_nat
            if n_nat else 0.0
        ),
    }
    reliability["error_rate"] = (
        reliability["failed_questions"] / n_nat if n_nat else 0.0
    )

    # ---- §2 Intent (natural runs) ----
    cls = classification_metrics(rows_natural)
    oos = [r for r in rows_natural if r.get("expected_intent") == "out_of_scope"]
    oos_predicted = [r for r in oos if r.get("intent") == "out_of_scope"]
    oos_precision = len(oos_predicted) / len(oos) if oos else None

    # ---- §3 Retrieval quality (natural runs) ----
    retrieval_natural: dict = {"evaluated_questions": 0}
    rows_with_gold = [r for r in rows_natural if r.get("relevant_chunk_ids")]
    retrieval_natural["evaluated_questions"] = len(rows_with_gold)
    for k in args.top_k:
        scored = [retrieval_metrics(r["chunk_ids"], r["relevant_chunk_ids"], k)
                  for r in rows_with_gold]
        scored = [s for s in scored if s is not None]
        retrieval_natural[f"k={k}"] = {
            f"hit@{k}":       safe_mean([s[f"hit@{k}"] for s in scored]),
            f"precision@{k}": safe_mean([s[f"precision@{k}"] for s in scored]),
            f"recall@{k}":    safe_mean([s[f"recall@{k}"] for s in scored]),
            f"mrr@{k}":       safe_mean([s[f"mrr@{k}"] for s in scored]),
            f"ndcg@{k}":      safe_mean([s[f"ndcg@{k}"] for s in scored]),
        }

    # ---- §4 Graph contribution ----
    # (a) Per-question delta between natural and baseline at the same k.
    base_by_id = {r["gold_id"]: r for r in rows_baseline if r.get("gold_id")}
    nat_by_id = {r["gold_id"]: r for r in rows_natural if r.get("gold_id")}
    paired = [(nat_by_id[g], base_by_id[g]) for g in nat_by_id if g in base_by_id
              and nat_by_id[g].get("relevant_chunk_ids")]

    graph_delta: dict = {"paired_questions": len(paired)}
    if paired:
        for k in args.top_k:
            nat_hits, base_hits = [], []
            nat_mrr, base_mrr = [], []
            nat_ndcg, base_ndcg = [], []
            for n_row, b_row in paired:
                m_nat = retrieval_metrics(n_row["chunk_ids"],
                                          n_row["relevant_chunk_ids"], k)
                m_base = retrieval_metrics(b_row["chunk_ids"],
                                           b_row["relevant_chunk_ids"], k)
                if not m_nat or not m_base:
                    continue
                nat_hits.append(m_nat[f"hit@{k}"])
                base_hits.append(m_base[f"hit@{k}"])
                nat_mrr.append(m_nat[f"mrr@{k}"])
                base_mrr.append(m_base[f"mrr@{k}"])
                nat_ndcg.append(m_nat[f"ndcg@{k}"])
                base_ndcg.append(m_base[f"ndcg@{k}"])
            if nat_hits:
                graph_delta[f"k={k}"] = {
                    "n": len(nat_hits),
                    "hit_natural":  safe_mean(nat_hits),
                    "hit_baseline": safe_mean(base_hits),
                    "delta_hit":    safe_mean(nat_hits) - safe_mean(base_hits),
                    "mrr_natural":  safe_mean(nat_mrr),
                    "mrr_baseline": safe_mean(base_mrr),
                    "delta_mrr":    safe_mean(nat_mrr) - safe_mean(base_mrr),
                    "ndcg_natural": safe_mean(nat_ndcg),
                    "ndcg_baseline": safe_mean(base_ndcg),
                    "delta_ndcg":   safe_mean(nat_ndcg) - safe_mean(base_ndcg),
                }

    # (b) Gold attribution: natural-run vector_chunk_ids / graph_chunk_ids
    attribution = {"vector_only": 0, "graph_only": 0, "both": 0,
                   "neither": 0, "n": 0, "missed_ids": []}
    for r in rows_natural:
        gold = set(str(x) for x in (r.get("relevant_chunk_ids") or []))
        if not gold:
            continue
        v = set(str(x) for x in (r.get("vector_chunk_ids") or []))
        g = set(str(x) for x in (r.get("graph_chunk_ids") or []))
        v_hit = bool(gold & v)
        g_hit = bool(gold & g)
        attribution["n"] += 1
        if v_hit and g_hit:
            attribution["both"] += 1
        elif v_hit:
            attribution["vector_only"] += 1
        elif g_hit:
            attribution["graph_only"] += 1
        else:
            attribution["neither"] += 1
            if r.get("gold_id"):
                attribution["missed_ids"].append(r["gold_id"])

    # ---- §5 Grounding and safety (natural runs) ----
    successful = [r for r in rows_natural if not r.get("error")]
    verified_count = sum(1 for r in successful if r.get("verified") is True)
    unverified_count = sum(1 for r in successful if r.get("verified") is not True)
    safety = [r for r in rows_natural if r.get("expected_intent") == "safety"]
    safety_unverified = [r for r in safety if r.get("verified") is not True]

    grounding = {
        "verified_count": verified_count,
        "unverified_count": unverified_count,
        "verification_pass_rate": verified_count / len(successful) if successful else 0.0,
        "mean_verification_issues": safe_mean([r.get("verify_issue_count", 0)
                                               for r in successful]),
        "safety_critical_count": len(safety),
        "safety_unverified_count": len(safety_unverified),
        "safety_unverified_rate": (
            len(safety_unverified) / len(safety) if safety else None
        ),
        "retry_rate": (
            sum(1 for r in successful if r.get("retry_count", 0) > 0)
            / len(successful) if successful else 0.0
        ),
        "query_reformulation_rate": (
            sum(1 for r in successful if r.get("reformulated")) / len(successful)
            if successful else 0.0
        ),
    }

    # ---- §6 Reference-answer quality + fact recall (natural runs) ----
    with_ref = [r for r in rows_natural if r.get("reference_answer")]
    answer_quality = {
        "reference_answer_count": len(with_ref),
        "exact_match": safe_mean([exact_match(r["answer"], r["reference_answer"])
                                  for r in with_ref]) if with_ref else None,
        "token_f1": safe_mean([token_f1(r["answer"], r["reference_answer"])
                               for r in with_ref]) if with_ref else None,
        "rouge_l_f1": safe_mean([rouge_l(r["answer"], r["reference_answer"])
                                 for r in with_ref]) if with_ref else None,
    }
    with_facts = [r for r in rows_natural
                  if r.get("required_facts") and r.get("answerable")]
    fact_scores = [fact_recall(r["answer"], r["required_facts"]) for r in with_facts]
    fact_scores = [s for s in fact_scores if s is not None]
    answer_quality["fact_recall"] = safe_mean(fact_scores) if fact_scores else None
    answer_quality["fact_recall_n"] = len(fact_scores)

    # ---- §7 Workflow (natural runs) ----
    workflow = {
        "retry_rate": grounding["retry_rate"],
        "query_reformulation_rate": grounding["query_reformulation_rate"],
        "graph_usage_rate": (
            sum(1 for r in successful if r.get("used_graph")) / len(successful)
            if successful else 0.0
        ),
    }

    # ---- Assemble ----
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gold_set": args.gold,
        "cache": args.cache,
        "top_k": args.top_k,
        "runs_total": len(rows_all),
        "runs_natural": n_nat,
        "runs_baseline": len(rows_baseline),
        "reliability": reliability,
        "intent_classification": cls,
        "out_of_scope_precision": oos_precision,
        "retrieval_natural": retrieval_natural,
        "graph_contribution": {
            "delta": graph_delta,
            "attribution": attribution,
        },
        "grounding": grounding,
        "answer_quality": answer_quality,
        "workflow": workflow,
    }

    # ---- Write outputs ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"metrics_{stamp}.json"
    csv_path = out_dir / f"details_{stamp}.csv"
    md_path = out_dir / f"report_{stamp}.md"

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")

    columns = sorted({k for row in rows_natural for k in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows_natural:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False, default=str)
                if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            })

    # ---- Markdown report ----
    r = metrics
    lines: list[str] = []
    W = lines.append

    W("# GraphRAG Evaluation Report")
    W("")
    W(f"- Generated: {r['generated_at']}")
    W(f"- Gold set: `{r['gold_set']}`")
    W(f"- Cache: `{r['cache']}`")
    W(f"- Runs: {r['runs_total']} "
      f"({r['runs_natural']} natural + {r['runs_baseline']} baseline)")
    W("")

    W("## 1. System reliability")
    W("")
    rel = r["reliability"]
    W(f"- Questions run / total: **{rel['questions_run']} / {rel['questions_total']}**")
    W(f"- Failed questions: **{rel['failed_questions']}**")
    W(f"- Non-empty answer rate: **{rel['nonempty_answer_rate']:.1%}**")
    W(f"- Error rate: **{rel['error_rate']:.2%}**")
    W("")

    W("## 2. Intent classification")
    W("")
    c = r["intent_classification"]
    W(f"- Accuracy: **{fmt(c['accuracy'])}**")
    W(f"- Macro precision: **{fmt(c['macro_precision'])}**")
    W(f"- Macro recall: **{fmt(c['macro_recall'])}**")
    W(f"- Macro F1: **{fmt(c['macro_f1'])}**")
    if r["out_of_scope_precision"] is not None:
        W(f"- Out-of-scope precision: **{r['out_of_scope_precision']:.1%}**")
    W("")
    if c["per_class"]:
        W("| Intent | Support | Precision | Recall | F1 |")
        W("|---|---:|---:|---:|---:|")
        for label, s in c["per_class"].items():
            W(f"| {label} | {s['support']} | {s['precision']:.3f} "
              f"| {s['recall']:.3f} | {s['f1']:.3f} |")
        W("")
    W("_Per-class F1s with support < 10 should be treated as anecdotal._")
    W("")

    W("## 3. Retrieval quality (natural graph_weight per question)")
    W("")
    rq = r["retrieval_natural"]
    if not rq["evaluated_questions"]:
        W("**No labelled retrieval questions.** Populate `expected_chunks`.")
    else:
        W(f"Evaluated questions: **{rq['evaluated_questions']}**")
        W("")
        for k in args.top_k:
            block = rq.get(f"k={k}")
            if not block:
                continue
            W(f"- k={k}: hit={fmt(block[f'hit@{k}'])} "
              f"precision={fmt(block[f'precision@{k}'])} "
              f"recall={fmt(block[f'recall@{k}'])} "
              f"MRR={fmt(block[f'mrr@{k}'])} "
              f"nDCG={fmt(block[f'ndcg@{k}'])}")
        W("")

    W("## 4. Graph contribution")
    W("")
    attr = r["graph_contribution"]["attribution"]
    W(f"- Gold chunk recovered by vector only: **{attr['vector_only']}**")
    W(f"- Gold chunk recovered by graph only:  **{attr['graph_only']}**")
    W(f"- Gold chunk recovered by both:        **{attr['both']}**")
    W(f"- Gold chunk recovered by neither:     **{attr['neither']}**")
    if attr["missed_ids"]:
        W(f"  - Failed entirely: {', '.join(attr['missed_ids'])}")
    W(f"- Total attributed questions: {attr['n']}")
    W("")
    delta = r["graph_contribution"]["delta"]
    if delta.get("paired_questions"):
        W(f"Paired questions (natural vs. baseline): {delta['paired_questions']}")
        W("")
        for k in args.top_k:
            block = delta.get(f"k={k}")
            if not block:
                continue
            W(f"### k={k} (n={block['n']})")
            sign_h = "+" if block["delta_hit"] >= 0 else ""
            sign_m = "+" if block["delta_mrr"] >= 0 else ""
            sign_n = "+" if block["delta_ndcg"] >= 0 else ""
            W(f"- hit: natural={block['hit_natural']:.3f} "
              f"baseline={block['hit_baseline']:.3f} "
              f"Δ={sign_h}{block['delta_hit']:.3f}")
            W(f"- MRR: natural={block['mrr_natural']:.3f} "
              f"baseline={block['mrr_baseline']:.3f} "
              f"Δ={sign_m}{block['delta_mrr']:.3f}")
            W(f"- nDCG: natural={block['ndcg_natural']:.3f} "
              f"baseline={block['ndcg_baseline']:.3f} "
              f"Δ={sign_n}{block['delta_ndcg']:.3f}")
            W("")
    else:
        W("_No paired runs available — run `evaluation.py --batch` first._")
        W("")

    W("## 5. Grounding and safety")
    W("")
    g = r["grounding"]
    W(f"- Verification pass rate: **{g['verification_pass_rate']:.1%}** "
      f"({g['verified_count']}/{g['verified_count'] + g['unverified_count']})")
    W(f"- Mean verifier issues per question: {g['mean_verification_issues']:.2f}")
    W(f"- Safety-critical questions: **{g['safety_critical_count']}**")
    W(f"- Safety-unverified: **{g['safety_unverified_count']}**")
    if g["safety_unverified_rate"] is not None:
        W(f"- Safety-unverified rate: {g['safety_unverified_rate']:.1%}")
    W(f"- Retry rate: {g['retry_rate']:.1%}")
    W(f"- Query reformulation rate: {g['query_reformulation_rate']:.1%}")
    W("")
    W("> The verification pass rate is a **lower bound on ungrounded answers**, "
      "not a measure of correctness. The verifier is a same-family LLM with "
      "no ground truth. Safety-critical counts below 20 are not statistically "
      "reliable.")
    W("")

    W("## 6. Reference-answer quality")
    W("")
    a = r["answer_quality"]
    if a["reference_answer_count"]:
        W(f"- Reference-labelled questions: {a['reference_answer_count']}")
        W(f"- Exact match: {fmt(a['exact_match'])}")
        W(f"- Token F1: {fmt(a['token_f1'])}")
        W(f"- ROUGE-L F1: {fmt(a['rouge_l_f1'])}")
    else:
        W("_No `reference_answer` fields set — add them to the gold set to enable "
          "exact-match / token-F1 / ROUGE-L._")
    if a.get("fact_recall") is not None:
        W(f"- Fact recall (required_facts): {fmt(a['fact_recall'])} "
          f"(n={a['fact_recall_n']})")
    W("")
    W("> Lexical metrics only. They measure surface overlap, not semantic "
      "correctness. A paraphrased but correct answer scores low.")
    W("")

    W("## 7. Workflow")
    W("")
    w = r["workflow"]
    W(f"- Retry rate: {w['retry_rate']:.1%}")
    W(f"- Query reformulation rate: {w['query_reformulation_rate']:.1%}")
    W(f"- Graph usage rate: {w['graph_usage_rate']:.1%}")
    W("")

    W("## Limitations")
    W("")
    W("- Verifier pass rate is not proof of correctness.")
    W("- Lexical metrics (EM, token-F1, ROUGE-L) measure surface overlap only.")
    W("- Per-class intent metrics with support < 10 are anecdotal.")
    W("- Without a hand-checked safety subset, safety claims are not validated.")
    W("- Gold-set `verified=false` items should be hand-checked before citing.")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 64)
    print("GRAPHRAG EVALUATION SUMMARY")
    print("=" * 64)
    print(f"Runs:                  {r['runs_total']} "
          f"({r['runs_natural']} natural + {r['runs_baseline']} baseline)")
    print(f"Reliability:           {rel['questions_run']}/{rel['questions_total']} "
          f"(errors={rel['failed_questions']})")
    print(f"Intent accuracy:       {fmt(c['accuracy'])}")
    print(f"Intent macro-F1:       {fmt(c['macro_f1'])}")
    if rq["evaluated_questions"]:
        for k in args.top_k:
            block = rq.get(f"k={k}")
            if block:
                print(f"hit@{k} (natural):       {fmt(block[f'hit@{k}'])}")
    print(f"Verification pass:     {g['verification_pass_rate']:.1%}")
    print(f"Graph usage:           {workflow['graph_usage_rate']:.1%}")
    print(f"Reports:               {md_path}")
    print(f"                       {json_path}")
    print(f"                       {csv_path}")


if __name__ == "__main__":
    main()