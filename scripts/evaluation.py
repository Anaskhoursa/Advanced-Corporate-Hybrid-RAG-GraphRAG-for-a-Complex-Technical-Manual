"""
Phase 7 + Evaluation harness (gold-set aware).

Each question runs ONCE at its natural graph_weight (derived from its
expected_intent via INTENT_GRAPH_WEIGHTS), plus one baseline run at
graph_weight=0.0 so evaluate_metrics.py can compute the graph-contribution
delta (§4 of the metrics spec).
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from groq import Groq, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

from retrieval import GraphRAGPipeline, GROQ_API_KEY


# ---------------------------------------------------------------------------
# Cache / config
# ---------------------------------------------------------------------------

CACHE_VERSION = 4  # bumped: per-question graph weight, natural + baseline
DEFAULT_CACHE_PATH = "data/eval_cache.json"
DEFAULT_GOLD_SET = "data/gold_set.json"

MAX_RETRIES = 1
MIN_ANSWER_CHARS = 40
MAX_HISTORY_TURNS = 10

TRANSIENT_MAX_RETRIES = 4
TRANSIENT_BASE_DELAY_S = 2.0
TRANSIENT_MAX_DELAY_S = 30.0
MAX_TOTAL_WAIT_S = 45 * 60
RATE_LIMIT_MIN_WAIT_S = 1.0
RATE_LIMIT_JITTER_S = 1.5

groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Retry counters
# ---------------------------------------------------------------------------

_RETRY_STATS = {
    "llm_calls": 0,
    "transient_retries": 0,
    "rate_limit_retries": 0,
    "total_sleep_s": 0.0,
}


def reset_retry_stats() -> None:
    _RETRY_STATS.update({
        "llm_calls": 0,
        "transient_retries": 0,
        "rate_limit_retries": 0,
        "total_sleep_s": 0.0,
    })


# ---------------------------------------------------------------------------
# Rate-limit parsing
# ---------------------------------------------------------------------------

_RETRY_IN_RE = re.compile(
    r"try again in\s+"
    r"(?:(?P<h>\d+)\s*h)?\s*"
    r"(?:(?P<m>\d+)\s*m)?\s*"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*s)?",
    re.IGNORECASE,
)


def _parse_retry_after_seconds(err: BaseException) -> float | None:
    m = _RETRY_IN_RE.search(str(err))
    if not m:
        return None
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m") or 0)
    seconds = float(m.group("s") or 0.0)
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def _is_rate_limit(err: BaseException) -> bool:
    if isinstance(err, RateLimitError):
        return True
    text = str(err).lower()
    return "rate_limit_exceeded" in text or ("429" in text and "rate limit" in text)


def _is_transient(err: BaseException) -> bool:
    if isinstance(err, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(err, APIStatusError):
        status = getattr(err, "status_code", None)
        if status is not None and 500 <= status < 600:
            return True
    text = str(err).lower()
    if "getaddrinfo failed" in text or "connection" in text:
        return True
    if any(f" {code}" in text for code in ("500", "502", "503", "504")):
        return True
    return False


def _sleep_with_log(seconds: float, reason: str) -> None:
    seconds = max(0.0, seconds)
    print(f"    [retry] sleeping {seconds:.2f}s — {reason}", flush=True)
    _RETRY_STATS["total_sleep_s"] += seconds
    time.sleep(seconds)


T = TypeVar("T")


def call_with_retry(fn: Callable[[], T], *, label: str = "groq_call") -> T:
    _RETRY_STATS["llm_calls"] += 1
    transient_attempt = 0
    total_wait = 0.0

    while True:
        try:
            return fn()
        except Exception as e:
            if _is_rate_limit(e):
                _RETRY_STATS["rate_limit_retries"] += 1
                hinted = _parse_retry_after_seconds(e)
                if hinted is None:
                    delay = min(60.0, 2.0 ** min(6, _RETRY_STATS["rate_limit_retries"]))
                else:
                    delay = max(RATE_LIMIT_MIN_WAIT_S, hinted)
                delay += random.uniform(0.0, RATE_LIMIT_JITTER_S)
                if total_wait + delay > MAX_TOTAL_WAIT_S:
                    print(f"    [{label}] rate-limit wait would exceed "
                          f"{MAX_TOTAL_WAIT_S}s budget; raising.", flush=True)
                    raise
                _sleep_with_log(delay, f"{label}: 429 rate limit")
                total_wait += delay
                continue

            if _is_transient(e):
                transient_attempt += 1
                if transient_attempt > TRANSIENT_MAX_RETRIES:
                    raise
                _RETRY_STATS["transient_retries"] += 1
                delay = min(TRANSIENT_MAX_DELAY_S,
                            TRANSIENT_BASE_DELAY_S * (2 ** (transient_attempt - 1)))
                delay += random.uniform(0.0, 1.0)
                if total_wait + delay > MAX_TOTAL_WAIT_S:
                    raise
                _sleep_with_log(delay, f"{label}: transient ({type(e).__name__})")
                total_wait += delay
                continue

            raise


# ---------------------------------------------------------------------------
# Per-intent graph weights
# ---------------------------------------------------------------------------

INTENT_GRAPH_WEIGHTS: dict[str, float] = {
    "factual":      0.3,
    "procedural":   0.9,
    "safety":       1.0,
    "comparison":   0.7,
    "troubleshoot": 0.9,
    "out_of_scope": 0.0,
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REFORMULATOR_SYSTEM_PROMPT = """You are a query rewriter for a pump manual Q&A system.

Your job: decide whether the user's new query can be answered standalone,
or whether it depends on the conversation history for context. If it
depends on history, rewrite it into a self-contained query. If it does
not, return it unchanged.

Rules:
- Return the query UNCHANGED if it is already self-contained.
- REWRITE the query if it depends on prior turns. Signals: pronouns with
  no antecedent, elliptical follow-ups, bare references to concepts
  introduced earlier, short queries that only make sense with context.
- When rewriting, produce ONE self-contained query. Combine the prior
  turn's subject with the new intent.
- Never invent details not present in either the new query or history.

Respond with JSON only:
{"rewritten": "<the standalone query>", "changed": true | false, "reason": "<brief>"}
"""


CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for a pump operation and maintenance manual Q&A system.

Classify the user's query into exactly one of these categories:

- "factual": asking for a specific value, spec, part name, or definition.
- "procedural": asking how to perform a task or procedure.
- "safety": asking about warnings, hazards, or safety precautions.
- "comparison": comparing items, or asking for highest/lowest/best.
- "troubleshoot": asking why something is wrong or how to fix a problem.
- "out_of_scope": clearly unrelated to this equipment (general knowledge,
  other software, personal tasks). The manual covers: the HL pump and its
  models, engine, air compressor, batteries, gearbox, mechanical seal, the
  trailer (towing, brakes, tires, prop stands), installation, piping,
  alignment, lubrication, maintenance intervals, troubleshooting,
  specifications, torque values and safety warnings. If the query touches
  ANY of these, it is NOT out_of_scope. When unsure, choose "factual".

The query you receive is already self-contained. Classify it on its own merits.

Respond with JSON only:
{"intent": "<category>"}
"""


VERIFIER_SYSTEM_PROMPT = """You are a grounding verifier for a pump manual Q&A system.

You will receive a QUESTION, an ANSWER, and the CONTEXT that was provided
to the answerer. Identify any claim in the answer that is not fully
supported by the context.

Respond with JSON only:
{
  "grounded": true | false,
  "issues": [
    {"claim": "...", "problem": "not_in_context | contradicts_context | overstated | citation_mismatch",
     "explanation": "brief"}
  ]
}

Rules:
- Numerical values must match exactly (including units).
- Warning levels (DANGER/WARNING/CAUTION/NOTICE) must match exactly.
- Paraphrases that preserve meaning are fine.
"""


ANSWER_REFORMULATOR_SYSTEM_PROMPT = """You rewrite pump-manual answers that failed a grounding check.

You will receive the original query, the previous answer, and the list
of grounding issues. Produce a corrected answer that:
- Removes or corrects every issue.
- Keeps any parts of the original answer that were correct.
- Still only uses information from the provided context.

Respond with JSON only:
{"answer": "<corrected answer>"}
"""


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_query: str
    standalone_query: str
    intent: str
    retrieval: dict
    context: str
    answer: str
    verified: bool
    verify_issues: list[str]
    retry_count: int
    reformulation_reason: str
    conversation_history: list[dict]
    graph_weight_override: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        q = (turn.get("query") or "").strip()
        a = (turn.get("answer") or "").strip()
        a_short = a[:300] + ("…" if len(a) > 300 else "")
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a_short}")
        lines.append("")
    return "\n".join(lines).strip()


def _groq_json(*, label, system_prompt, user_prompt, max_tokens,
               temperature=0.0, reasoning_effort="low") -> dict:
    def _do_call():
        return groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
            reasoning_format="hidden",
        )
    resp = call_with_retry(_do_call, label=label)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reformulate_query_node(state: AgentState) -> dict:
    user_query = state.get("user_query") or state.get("query") or ""
    history = state.get("conversation_history") or []

    if not history:
        return {"standalone_query": user_query, "reformulation_reason": "", "retry_count": 0}

    history_block = _format_history_for_prompt(history)
    try:
        payload = _groq_json(
            label="reformulate_query",
            system_prompt=REFORMULATOR_SYSTEM_PROMPT,
            user_prompt=(f"Conversation history:\n{history_block}\n\nNew user query: {user_query}"),
            max_tokens=256,
        )
        rewritten = (payload.get("rewritten") or user_query).strip()
        changed = bool(payload.get("changed", False))
        reason = (payload.get("reason") or "").strip()
        if not rewritten:
            rewritten, changed = user_query, False
        return {"standalone_query": rewritten,
                "reformulation_reason": reason if changed else "",
                "retry_count": 0}
    except Exception as e:
        print(f"[reformulate_query] failed, using raw query: {e}", flush=True)
        return {"standalone_query": user_query, "reformulation_reason": "", "retry_count": 0}


def classify_query_node(state: AgentState) -> dict:
    query = state.get("standalone_query") or state.get("user_query") or ""
    try:
        payload = _groq_json(
            label="classify",
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=query,
            max_tokens=64,
        )
        intent = payload.get("intent", "factual")
    except Exception as e:
        print(f"[classify] falling back to 'factual': {e}", flush=True)
        intent = "factual"

    if intent not in INTENT_GRAPH_WEIGHTS:
        intent = "factual"

    if intent == "out_of_scope":
        pipeline = state.get("_pipeline")
        if pipeline is not None and pipeline.linker.link(query, drop_properties=False):
            intent = "factual"

    return {"intent": intent}


def retrieve_node(state: AgentState) -> dict:
    pipeline: GraphRAGPipeline = state["_pipeline"]
    query = state.get("standalone_query") or state.get("user_query") or ""
    intent = state.get("intent", "factual")

    # Honour per-run override (used by the graph-weight ablation); otherwise
    # fall back to the intent-derived natural weight.
    override = state.get("graph_weight_override")
    if override is not None:
        graph_weight = float(override)
    else:
        graph_weight = INTENT_GRAPH_WEIGHTS.get(intent, 0.5)

    if intent == "out_of_scope" and graph_weight == 0.0:
        return {
            "retrieval": {
                "chunks": [], "chunk_ids": [], "warnings": [], "steps": [],
                "vector_chunk_ids": [], "graph_chunk_ids": [],
                "linked_entities": [], "used_graph": False,
            },
            "context": "",
        }

    result, context = pipeline.retrieve(query, graph_weight=graph_weight)

    return {
        "retrieval": {
            "chunk_ids": result.chunk_ids,
            "vector_chunk_ids": result.vector_chunk_ids,
            "graph_chunk_ids": result.graph_chunk_ids,
            "linked_entities": result.linked_entities,
            "used_graph": result.used_graph,
            "warnings_count": len(result.warnings),
            "steps_count": len(result.steps),
        },
        "context": context,
    }


def generate_node(state: AgentState) -> dict:
    pipeline: GraphRAGPipeline = state["_pipeline"]
    query = state.get("standalone_query") or state.get("user_query") or ""
    context = state.get("context", "")

    if state.get("intent") == "out_of_scope":
        return {"answer": (
            "This question is outside the scope of the HL Series pump "
            "manual. I can only answer questions about the pump, its "
            "operation, maintenance, and safety."
        )}

    if not context.strip():
        return {"answer": (
            "The manual does not contain information needed to answer "
            "this question. Please rephrase or consult a Xylem representative."
        )}

    answer = call_with_retry(lambda: pipeline.generate(query, context), label="generate")
    return {"answer": answer}


def verify_node(state: AgentState) -> dict:
    intent = state.get("intent")
    if intent == "out_of_scope":
        return {"verified": True, "verify_issues": []}

    query = state.get("standalone_query") or state.get("user_query") or ""
    answer = state.get("answer", "")
    context = state.get("context", "")

    if len(answer.strip()) < MIN_ANSWER_CHARS:
        return {"verified": False, "verify_issues": ["Answer is too short to verify."]}

    try:
        payload = _groq_json(
            label="verify",
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=f"QUESTION:\n{query}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}",
            max_tokens=1024,
        )
        return {"verified": bool(payload.get("grounded", False)),
                "verify_issues": payload.get("issues", []) or []}
    except Exception as e:
        print(f"[verify] verifier call failed: {e}", flush=True)
        return {"verified": True, "verify_issues": []}


def reformulate_answer_node(state: AgentState) -> dict:
    query = state.get("standalone_query") or state.get("user_query") or ""
    answer = state.get("answer", "")
    context = state.get("context", "")
    issues = state.get("verify_issues", [])

    issue_text = "\n".join(
        f"- {i.get('problem', '?')}: {i.get('claim', '?')} — {i.get('explanation', '')}"
        for i in issues if isinstance(i, dict)
    ) or "(no specific issues listed)"

    try:
        payload = _groq_json(
            label="reformulate_answer",
            system_prompt=ANSWER_REFORMULATOR_SYSTEM_PROMPT,
            user_prompt=(
                f"Query: {query}\n\nPrevious answer:\n{answer}\n\n"
                f"Grounding issues:\n{issue_text}\n\n"
                f"Context (the only source you may use):\n{context}"
            ),
            max_tokens=4096, temperature=0.2, reasoning_effort="medium",
        )
        new_answer = (payload.get("answer") or answer).strip()
    except Exception as e:
        print(f"[reformulate_answer] failed, keeping original: {e}", flush=True)
        new_answer = answer

    return {"answer": new_answer or answer,
            "retry_count": state.get("retry_count", 0) + 1,
            "verified": False, "verify_issues": []}


def memory_node(state: AgentState) -> dict:
    history = list(state.get("conversation_history") or [])
    history.append({
        "query": state.get("user_query") or "",
        "standalone_query": state.get("standalone_query") or "",
        "answer": state.get("answer", ""),
        "verified": state.get("verified", True),
        "intent": state.get("intent", "factual"),
    })
    return {"conversation_history": history[-MAX_HISTORY_TURNS:]}


def route_after_verify(state: AgentState) -> Literal["reformulate_answer", "memory"]:
    if state.get("verified", False):
        return "memory"
    if state.get("intent") == "out_of_scope":
        return "memory"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "memory"
    return "reformulate_answer"


def build_graph(pipeline: GraphRAGPipeline):
    builder = StateGraph(AgentState)

    def bound(fn):
        def _fn(state):
            state = dict(state)
            state["_pipeline"] = pipeline
            return fn(state)
        return _fn

    builder.add_node("reformulate_query",  reformulate_query_node)
    builder.add_node("classify",           bound(classify_query_node))
    builder.add_node("retrieve",           bound(retrieve_node))
    builder.add_node("generate",           bound(generate_node))
    builder.add_node("verify",             verify_node)
    builder.add_node("reformulate_answer", reformulate_answer_node)
    builder.add_node("memory",             memory_node)

    builder.set_entry_point("reformulate_query")
    builder.add_edge("reformulate_query", "classify")
    builder.add_edge("classify",           "retrieve")
    builder.add_edge("retrieve",           "generate")
    builder.add_edge("generate",           "verify")
    builder.add_conditional_edges(
        "verify", route_after_verify,
        {"reformulate_answer": "reformulate_answer", "memory": "memory"},
    )
    builder.add_edge("reformulate_answer", "verify")
    builder.add_edge("memory",             END)

    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Conversation session
# ---------------------------------------------------------------------------

class ConversationSession:
    def __init__(self, graph, thread_id: str, initial_history: list[dict] | None = None):
        self.graph = graph
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        self.history: list[dict] = list(initial_history or [])

    def ask(self, query: str, graph_weight_override: float | None = None) -> dict:
        payload = {
            "user_query": query,
            "conversation_history": self.history,
        }
        if graph_weight_override is not None:
            payload["graph_weight_override"] = float(graph_weight_override)
        final_state = self.graph.invoke(payload, config=self.config)
        self.history = final_state.get("conversation_history", [])
        return final_state

    def reset(self):
        self.history = []


# ---------------------------------------------------------------------------
# Gold-set loader
# ---------------------------------------------------------------------------

def load_gold_set(path: str = DEFAULT_GOLD_SET) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict] = []
    for item in data:
        out.append({
            "id": item.get("id"),
            "question": item["question"],
            "expected_intent": item.get("category"),
            "relevant_chunk_ids": item.get("expected_chunks") or [],
            "required_facts": item.get("required_facts") or [],
            "answerable": bool(item.get("answerable", True)),
            "human_verified": bool(item.get("verified", False)),
            "note": item.get("note", ""),
        })
    return out


def TEST_QUESTIONS() -> list[dict]:
    return load_gold_set()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _question_fingerprint(question: str, expected_intent: str | None,
                          note: str, graph_weight: float) -> str:
    h = hashlib.sha256()
    for part in (str(CACHE_VERSION), question, expected_intent or "",
                 note, f"{graph_weight:.4f}"):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _load_cache(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"cache_version": CACHE_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "entries": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        backup = f"{path}.corrupt.{int(time.time())}"
        try:
            p.replace(backup)
        except Exception:
            pass
        print(f"[cache] could not read {path} ({e}); moved to {backup}", flush=True)
        return {"cache_version": CACHE_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "entries": {}}
    if raw.get("cache_version") != CACHE_VERSION:
        print(f"[cache] version mismatch "
              f"(file={raw.get('cache_version')}, code={CACHE_VERSION}); ignoring.",
              flush=True)
        return {"cache_version": CACHE_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "entries": {}}
    raw.setdefault("entries", {})
    return raw


def _save_cache_atomic(cache: dict, path: str) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    cache["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    graph,
    out_path: str | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    gold_path: str = DEFAULT_GOLD_SET,
    resume: bool = True,
    clear_cache: bool = False,
) -> str:
    """
    Run the gold-set evaluation.

    Each question runs at its natural graph_weight (derived from its
    expected_intent), plus one baseline run at graph_weight=0.0 for the
    graph-contribution ablation. So total runs = 2 * len(gold).
    """
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"evaluation_report_{ts}.txt"

    if clear_cache and Path(cache_path).exists():
        print(f"[cache] --clear-cache: removing {cache_path}", flush=True)
        Path(cache_path).unlink()

    cache = _load_cache(cache_path) if resume else {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entries": {},
    }

    gold = load_gold_set(gold_path)

    # Build the plan: one natural-weight run + one baseline per question.
    plan: list[dict] = []
    for i, item in enumerate(gold, start=1):
        intent = item.get("expected_intent")
        natural = round(INTENT_GRAPH_WEIGHTS.get(intent, 0.5), 3)
        for label, gw in (("natural", natural), ("baseline", 0.0)):
            fp = _question_fingerprint(
                item["question"], intent, item["note"], gw,
            )
            plan.append({
                "index": i,
                "gold": item,
                "condition": label,
                "graph_weight": gw,
                "fingerprint": fp,
                "cached": cache["entries"].get(fp),
            })

    n_total = len(plan)
    n_cached = sum(1 for p in plan if p["cached"])
    n_to_run = n_total - n_cached
    all_weights = sorted({p["graph_weight"] for p in plan})

    print("=" * 70)
    print(f"Evaluation: {len(gold)} questions -> {n_total} runs "
          f"({len(gold)} natural + {len(gold)} baseline)")
    print(f"  cached: {n_cached}, to run: {n_to_run}")
    print(f"  graph weights in plan: {all_weights}")
    print(f"Report -> {out_path}")
    print(f"Cache  -> {cache_path}")
    print("=" * 70)

    reset_retry_stats()

    # One session per condition so follow-ups stay coherent within a condition
    # but never bleed across conditions.
    session = ConversationSession(graph, thread_id="eval_session")

    # Seed from cache in plan order (matches gold order + condition).
    seed_hist: list[dict] = []
    for p in plan:
        if p["cached"]:
            e = p["cached"]
            seed_hist.append({
                "query": e.get("question", p["gold"]["question"]),
                "standalone_query": e.get("standalone_query", "") or "",
                "answer": e.get("answer_preview", "") or "",
                "verified": bool(e.get("verified", False)),
                "intent": e.get("intent", "factual"),
            })
        else:
            break
    session.history = seed_hist[-MAX_HISTORY_TURNS:]

    per_turn: list[dict] = []
    overall_start = time.perf_counter()

    for p in plan:
        gw = p["graph_weight"]
        item = p["gold"]
        i = p["index"]
        q = item["question"]
        cond = p["condition"]

        if p["cached"]:
            print(f"[{i:02d}/{len(gold)} {cond:<8} gw={gw:.2f}] "
                  f"(cached) {q[:55]}", flush=True)
            per_turn.append({**p["cached"], "from_cache": True,
                             "graph_weight": gw,
                             "condition": cond,
                             "gold_id": item["id"]})
            continue

        print(f"[{i:02d}/{len(gold)} {cond:<8} gw={gw:.2f}] {q[:55]}", flush=True)

        retries_before = dict(_RETRY_STATS)
        t0 = time.perf_counter()
        try:
            state = session.ask(q, graph_weight_override=gw)
            err = None
        except Exception as e:
            state = {}
            err = repr(e)
            session.reset()
        elapsed = time.perf_counter() - t0

        retries_after = dict(_RETRY_STATS)
        answer = state.get("answer", "") or ""
        issues = state.get("verify_issues") or []
        retrieval = state.get("retrieval", {}) or {}

        entry = {
            "index": i,
            "gold_id": item["id"],
            "condition": cond,
            "graph_weight": gw,
            "question": q,
            "expected_intent": item["expected_intent"],
            "note": item["note"],
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            "required_facts": item["required_facts"],
            "answerable": item["answerable"],
            "human_verified": item["human_verified"],
            "standalone_query": state.get("standalone_query", "") or "",
            "reformulated": bool(state.get("standalone_query")
                                 and state.get("standalone_query") != q),
            "intent": state.get("intent", "?"),
            "verified": bool(state.get("verified", False)),
            "retry_count": int(state.get("retry_count", 0) or 0),
            "verify_issue_count": len(issues) if isinstance(issues, list) else 0,
            "answer_chars": len(answer),
            "answer_preview": answer[:280].replace("\n", " ").strip(),
            "answer": answer,
            "used_graph": bool(retrieval.get("used_graph", False)),
            "chunk_ids": list(retrieval.get("chunk_ids", []) or []),
            "vector_chunk_ids": list(retrieval.get("vector_chunk_ids", []) or []),
            "graph_chunk_ids": list(retrieval.get("graph_chunk_ids", []) or []),
            "linked_entities": retrieval.get("linked_entities", []) or [],
            "latency_s": elapsed,
            "rate_limit_retries":
                retries_after["rate_limit_retries"] - retries_before["rate_limit_retries"],
            "transient_retries":
                retries_after["transient_retries"] - retries_before["transient_retries"],
            "sleep_s": retries_after["total_sleep_s"] - retries_before["total_sleep_s"],
            "error": err,
            "from_cache": False,
        }

        per_turn.append(entry)

        cache["entries"][p["fingerprint"]] = entry
        try:
            _save_cache_atomic(cache, cache_path)
        except Exception as e:
            print(f"    [cache] WARNING: could not write cache: {e}", flush=True)

    overall_elapsed = time.perf_counter() - overall_start

    # ---- report ----
    n = len(per_turn)
    latencies = [t["latency_s"] for t in per_turn if not t.get("from_cache")]
    errored = [t for t in per_turn if t["error"]]
    verified_count = sum(1 for t in per_turn if t["verified"])
    # Metrics are computed on the natural-condition runs only.
    natural_runs = [t for t in per_turn if t["condition"] == "natural"]
    n_nat = len(natural_runs)

    lines: list[str] = []
    W = lines.append

    W("=" * 78)
    W("HL SERIES PUMP MANUAL — GRAPHRAG EVALUATION")
    W("=" * 78)
    W(f"Generated:     {datetime.now().isoformat(timespec='seconds')}")
    W(f"Gold set:      {gold_path}  ({len(gold)} questions)")
    W(f"Runs:          {n}  ({len(gold)} natural + {len(gold)} baseline)")
    W(f"Cache hits:    {n_cached}/{n_total}")
    W(f"Wall time:     {overall_elapsed:.2f}s")
    W("")

    W("-" * 78)
    W("RELIABILITY")
    W("-" * 78)
    W(f"Successful runs:  {n - len(errored)}/{n}")
    W(f"Failed runs:      {len(errored)}/{n}")
    W(f"Non-empty answers:{sum(1 for t in per_turn if t['answer'].strip())}/{n}")
    W(f"Error rate:       {len(errored) / n:.2%}" if n else "n/a")
    if latencies:
        W(f"Mean latency (fresh): {statistics.mean(latencies):.2f}s")
    W("")

    W("-" * 78)
    W("GROUNDING (natural-weight runs only)")
    W("-" * 78)
    nat_verified = sum(1 for t in natural_runs if t["verified"])
    W(f"Verified:           {nat_verified}/{n_nat}")
    W(f"Verification pass:  {nat_verified / n_nat:.1%}" if n_nat else "n/a")
    safety = [t for t in natural_runs if t["expected_intent"] == "safety"]
    safety_unverified = [t for t in safety if not t["verified"]]
    W(f"Safety-critical questions: {len(safety)}")
    W(f"Safety-unverified:         {len(safety_unverified)}")
    W(f"Retry rate:        {sum(1 for t in natural_runs if t['retry_count'] > 0) / n_nat:.1%}"
      if n_nat else "n/a")
    W(f"Reformulation rate:{sum(1 for t in natural_runs if t['reformulated']) / n_nat:.1%}"
      if n_nat else "n/a")
    W(f"Graph usage rate:  {sum(1 for t in natural_runs if t['used_graph']) / n_nat:.1%}"
      if n_nat else "n/a")
    W("")

    W("=" * 78)
    W("END OF REPORT")
    W("=" * 78)

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Runs:              {n}")
    print(f"Failed:            {len(errored)}")
    print(f"Verified (natural):{nat_verified}/{n_nat} "
          f"({nat_verified / n_nat:.1%})" if n_nat else "Verified (natural): n/a")
    print(f"Graph usage:       "
          f"{sum(1 for t in natural_runs if t['used_graph']) / n_nat:.1%}"
          if n_nat else "Graph usage: n/a")
    print(f"Report written to: {out_path}")
    print("=" * 70)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HL Series Pump Manual — GraphRAG evaluator")
    parser.add_argument("--batch", action="store_true",
                        help="Run the evaluation suite (this is the only mode)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--gold", type=str, default=DEFAULT_GOLD_SET)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline(entities_path="data/entities.json")
    graph = build_graph(pipeline)

    if args.batch:
        run_evaluation(
            graph,
            out_path=args.out,
            cache_path=args.cache,
            gold_path=args.gold,
            resume=not args.no_resume,
            clear_cache=args.clear_cache,
        )
    else:
        print("Use --batch to run the evaluation.")


if __name__ == "__main__":
    main()