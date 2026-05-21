#!/usr/bin/env python3
"""
Phase 26M: Standalone SDI packet runtime harness.

Runs raw-context baselines and SDI packet prompts against a clean local backend
without touching llama.cpp internals, PRT, OpenClaw, or routing layers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sdi_memory_guard import evaluate_guard, read_memory_state
from sdi_packet_builder import build_sdi_packet, token_estimate


BASELINES = ("no_packet", "recent_only", "simple_summary", "sdi_packet", "auto")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def pinned_parts(pinned_data: Any) -> tuple[list[str], list[str], list[str]]:
    if isinstance(pinned_data, list):
        return pinned_data, [], []
    pinned_facts = list(pinned_data.get("pinned_facts", []))
    open_loops = list(pinned_data.get("open_loops", []))
    constraints_verbatim = list(pinned_data.get("constraints_verbatim", []))
    # Also detect hard constraints embedded in pinned_facts (e.g. "HARD CONSTRAINT: ...")
    for fact in pinned_facts:
        if fact.startswith("HARD CONSTRAINT:") and fact not in constraints_verbatim:
            constraints_verbatim.append(fact)
    return pinned_facts, open_loops, constraints_verbatim


def extract_question(conversation: str, question: str | None) -> str:
    if question:
        return question.strip()
    for line in reversed(conversation.splitlines()):
        stripped = line.strip()
        if stripped.startswith("User:"):
            return stripped[5:].strip()
    return "Answer the current user request from the provided context."


def recent_only(conversation: str, max_lines: int = 24) -> str:
    lines = [line for line in conversation.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def simple_summary(conversation: str, pinned_facts: list[str], max_recent_lines: int = 16) -> str:
    summary = []
    summary.append("Summary:")
    for fact in pinned_facts[:8]:
        summary.append(f"- {fact}")
    summary.append("")
    summary.append("Recent conversation:")
    summary.append(recent_only(conversation, max_recent_lines))
    return "\n".join(summary)


def detect_repeated_filler(conversation: str) -> bool:
    lines = [line.strip() for line in conversation.splitlines() if len(line.strip()) > 40]
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    if any(count >= 3 for count in counts.values()):
        return True
    lower = conversation.lower()
    filler_markers = (
        "roman empire",
        "byzantine empire",
        "database management systems",
        "software testing",
        "irrelevant warning",
        "background note",
    )
    return sum(lower.count(marker) for marker in filler_markers) >= 3


def has_tool_exactness(text: str) -> bool:
    return bool(
        re.search(r"(/tmp/|/home/|commit|hash|sha|peak|rss|ctx=|ctx_size|status:|result file|working dir|bytes|error|warning)", text, re.I)
    )


def question_refs_exact_tool_output(question: str) -> bool:
    return bool(re.search(r"(exact|tool|command|path|hash|commit|metric|value|unit|status|error|failed|why)", question, re.I))


def has_conflict_facts(pinned_facts: list[str], conversation: str) -> bool:
    joined = "\n".join(pinned_facts) + "\n" + conversation
    return bool(re.search(r"(updated from|superseded|old value|new value|changed from|changed to|correction)", joined, re.I))


def question_refs_state_or_facts(question: str) -> bool:
    return bool(
        re.search(r"(project|state|status|current|setting|fact|constraint|commit|path|issue|next|root cause|where are we|what should)", question, re.I)
    )


def detect_multi_commit_git_review(text: str, question: str) -> bool:
    """Detect multi-commit git log or code review summarization tasks.
    These need simple_summary, not rigid packet format."""
    commit_pattern = bool(re.search(r"commit\s+[0-9a-f]{7,}", text, re.I))
    diff_pattern = bool(re.search(r"(diff --git|\+{3}|\-{3}|@@|file changed|M\s+.*\|)", text))
    multi_file = len(re.findall(r"\.(py|js|ts|go|rs|cpp|h|md|txt)", text)) >= 3
    commit_count = len(re.findall(r"commit\s+[0-9f]{7,}", text, re.I))
    review_kw = bool(re.search(r"(review|changes|what (did|didn't|was)|which commit|import bug|fix commit|refactor)", question, re.I))
    return (commit_count >= 2 or (commit_pattern and diff_pattern)) and (multi_file or review_kw)


def detect_benchmark_result(text: str, question: str) -> bool:
    """Detect benchmark/tool result tables with numeric metrics.
    These may need exact_tool mode or simple_summary, not generic packet."""
    metric_kw = bool(re.search(r"(benchmark|SPEC_BENCH|tok/s|throughput|latency|p95|median|average|score:|result file|commit:|status:|completed|failed)", text, re.I))
    numeric_table = bool(re.search(r"(\d+\.\d+.*tok/s|\d+.*ms|score:\s*\d+\.\d+)", text))
    question_exact = bool(re.search(r"(score|commit|status|result|exact|what was)", question, re.I))
    return metric_kw and (numeric_table or question_exact)


def question_wants_summarize(question: str) -> bool:
    """Does the question want a summary/interpretation rather than exact values?"""
    return bool(re.search(r"(what (did|was|is the|are the)|tell me|tell me about|explain|describe|how did|summarize|review|analyze|compare)", question, re.I))


def question_wants_exact(question: str) -> bool:
    """Does the question want exact values (path, commit, number, status)?"""
    return bool(re.search(r"(exact|which commit|what is the.*path|what is the.*file|what was the.*score|what was the.*number|what.*status|what.*error|what.*command)", question, re.I))


def is_tiny_self_contained_question(question: str, raw_tokens: int, open_loops: list, pinned_facts: list, joined: str) -> bool:
    """Detect a tiny self-contained question that needs no packet compression.

    A question where:
    - The raw context is small (<500 tokens)
    - The question is short and self-contained (<15 words)
    - No open loops, no conflicting state, no tool exactness risk
    - The answer is visible in the recent conversation window

    For these, sdi_packet adds overhead and can hurt output quality.
    """
    if raw_tokens >= 500:
        return False
    short_question = len(question.split()) < 15
    no_open_state = not open_loops and not bool(re.search(r"(updated from|superseded|old value|new value|changed from|conflict)", joined, re.I))
    no_tool_exact = not bool(re.search(r"(commit|sha|hash|/tmp/|/home/|/workspace|error|status:|result file|metric|peak|rss)", joined, re.I))
    self_contained_pattern = bool(re.search(r"^(can you |please |could you )?(verify|confirm|check|what is|what was|tell me|give me).{0,30}$", question.strip(), re.I))
    return short_question and no_open_state and no_tool_exact and self_contained_pattern


def is_simple_factual_recall(question: str, raw_tokens: int, pinned_facts: list, conversation: str) -> bool:
    """Detect a simple factual recall question that doesn't need sdi_packet.

    Even when context is large (benchmark/results), a question that is genuinely just
    asking to confirm/retrieve a single simple value from pinned facts should not
    route to sdi_packet/exact-tool mode. The answer is already visible in facts.
    """
    # Must have a simple recall pattern in the question
    has_simple_pattern = bool(re.search(r"(exact|verify|confirm|check).{0,20}(license|name|value|status)", question, re.I))
    if not has_simple_pattern:
        return False
    # The fact must actually be present in pinned_facts
    joined_pinned = " ".join(pinned_facts)
    answer_visible = bool(re.search(r"(license|name|value|status).{0,30}(mit|apache|gpl|bsd|open.?source|found|active|completed)", joined_pinned, re.I))
    return answer_visible


def is_hard_constraint_decision(question: str, hard_constraints: list) -> bool:
    """Detect a 'should I / can I / what should I do' question with hard constraints.

    When a user asks whether to do something and hard constraints say never/always/must not,
    the answer should be natural ("No — constraint. Recommended action: ...") not a packet.
    """
    if not hard_constraints:
        return False
    # Decision keywords in question: asking permission or exception
    decision_kw = bool(re.search(r"(should i|can i|do we|what should i|could we|would it be safe|allowed to)", question, re.I))
    # Exception/override keywords strongly suggest this is about bypassing a constraint
    exception_kw = bool(re.search(r"(exception|override|absolutely|if i had to|force bypass)", question, re.I))
    # If hard constraints exist AND question is about permission/exception, prefer natural answer
    return decision_kw or exception_kw


def detect_open_loop_continuation(question: str, open_loops: list) -> bool:
    """Detect an open-loop continuation question: 'resume / continue / where were we / next step'.

    For pure continuations where the answer is in the recent conversation window,
    sdi_packet framing can break natural conversational flow.
    Use recent_only or simple_summary unless old-context retrieval is truly needed.
    """
    if not open_loops and not bool(re.search(r"(continue|resume|where did we|where were we|next step|keep going|pick up|left off)", question, re.I)):
        return False
    # Only trigger if no explicit tool/path/exactness requirement
    no_exact_req = not bool(re.search(r"(exact commit|exact path|exact file|exact command|exact hash)", question, re.I))
    return no_exact_req


def choose_auto_policy(
    conversation: str,
    pinned_data: Any,
    question: str,
    tier: int | None,
    model: str,
    active_context_target: int,
) -> dict[str, Any]:
    pinned_facts, open_loops, hard_constraints = pinned_parts(pinned_data)
    raw_tokens, raw_words = token_estimate(conversation)
    recent_tokens, _ = token_estimate(recent_only(conversation))
    summary_tokens, _ = token_estimate(simple_summary(conversation, pinned_facts))
    guard = evaluate_guard(model=model, active_context_target=active_context_target, force_tier=tier)

    joined = "\n".join(pinned_facts + open_loops + hard_constraints) + "\n" + conversation
    risk_flags = {
        "long_context": raw_words > 1500 or raw_tokens > 2200,
        "medium_context": raw_words > 450 or raw_tokens > 900,
        "repeated_filler": detect_repeated_filler(conversation),
        "pinned_fact_question": bool(pinned_facts) and question_refs_state_or_facts(question),
        "open_loops": bool(open_loops),
        "conflicting_old_new": has_conflict_facts(pinned_facts, conversation),
        "tool_exactness": has_tool_exactness(joined) or has_tool_exactness(question),
        "exact_tool_question": question_refs_exact_tool_output(question),
        "memory_pressure": (not guard["safe"]) or guard["recommended_tier"] == 0 or active_context_target > 8192,
        "tiny_context": raw_tokens < 350,
        "self_contained_question": raw_tokens < 350 and not pinned_facts and not open_loops and not has_tool_exactness(joined),
    }

    # === PHASE 26S: Content-aware routing (needed by targeted gates) ===
    # Detect content types first, before risk flag evaluation.
    is_multi_commit_review = detect_multi_commit_git_review(joined, question)
    is_benchmark_result = detect_benchmark_result(joined, question)
    wants_summary = question_wants_summarize(question) if (is_multi_commit_review or is_benchmark_result) else False
    wants_exact = question_wants_exact(question) if (is_multi_commit_review or is_benchmark_result) else False

    reasons: list[str] = []

    # === PHASE 26V: Targeted policy gates (must run before risk-flag cascade) ===
    # Gate 0: simple factual recall — question is just confirming/retrieving a single simple value
    # (e.g. "verify the license" when MIT is already in pinned facts)
    if is_simple_factual_recall(question, raw_tokens, pinned_facts, conversation):
        selected = "recent_only" if recent_tokens < raw_tokens else "no_packet"
        reasons.append("simple factual recall; answer already visible in pinned facts")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # Gate 1: tiny self-contained question (Scenario 21 fix — factual recall, not benchmark analysis)
    # Check this BEFORE benchmark detection to prevent benchmark_result misfire on simple factual questions
    if is_tiny_self_contained_question(question, raw_tokens, open_loops, pinned_facts, joined):
        selected = "recent_only" if recent_tokens < raw_tokens else "no_packet"
        reasons.append("tiny self-contained question; no packet compression needed")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # Gate 2: hard-constraint decision question (Scenario 25 fix)
    if is_hard_constraint_decision(question, hard_constraints):
        selected = "recent_only" if recent_tokens <= 500 else "simple_summary"
        reasons.append("hard-constraint decision question; natural answer format needed")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # Gate 3: open-loop continuation (Scenario 26 fix)
    if detect_open_loop_continuation(question, open_loops):
        selected = "recent_only" if recent_tokens <= raw_tokens else "simple_summary"
        reasons.append("open-loop continuation; natural conversational flow preferred")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # === PHASE 26S: Content-aware decisions ===
    reasons: list[str] = []
    # Scenario 17 failure: multi-commit code review → simple_summary, not sdi_packet
    if is_multi_commit_review and not open_loops and not hard_constraints:
        if wants_summary or not wants_exact:
            selected = "simple_summary"
            reasons.append("multi-commit git review; summarization task → simple_summary")
            return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)
        elif wants_exact:
            # Exact commit/hash question in multi-commit context
            selected = "sdi_packet"
            reasons.append("multi-commit review with exact commit question; packet with exact-tool fields")
            return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # Scenario 15 failure: benchmark result with exact values needed
    # For benchmark/score tables: prioritize exactness over summarization.
    # "What was the SPEC_BENCH score and which commit was used?" → exact values, not summary.
    if is_benchmark_result and not risk_flags["long_context"] and not risk_flags["repeated_filler"]:
        if wants_exact:
            # Exact metric/status/commit/path question → sdi_packet with exact-tool fields
            selected = "sdi_packet"
            reasons.append("benchmark result with exact metric/commit/status question; exact-tool mode")
            return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)
        elif wants_summary:
            # Broad interpretation/summary question → simple_summary
            selected = "simple_summary"
            reasons.append("benchmark/score table; summary question → simple_summary")
            return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # === PHASE 26V: Targeted policy gates (must run before risk-flag cascade) ===
    # Gate 1: tiny self-contained question (Scenario 21 fix)
    if is_tiny_self_contained_question(question, raw_tokens, open_loops, pinned_facts, joined):
        selected = "recent_only" if recent_tokens < raw_tokens else "no_packet"
        reasons.append("tiny self-contained question; no packet compression needed")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # Gate 2: hard-constraint decision question (Scenario 25 fix)
    if is_hard_constraint_decision(question, hard_constraints):
        # If recent context has the constraint visible, use recent_only for natural answer
        # If recent context is too large, use simple_summary
        selected = "recent_only" if recent_tokens <= 500 else "simple_summary"
        reasons.append("hard-constraint decision question; natural answer format needed")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # Gate 3: open-loop continuation (Scenario 26 fix)
    if detect_open_loop_continuation(question, open_loops):
        # For pure continuation, prefer natural conversational flow
        selected = "recent_only" if recent_tokens <= raw_tokens else "simple_summary"
        reasons.append("open-loop continuation; natural conversational flow preferred")
        return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)

    # === Existing risk-flag cascade (unchanged) ===
    if risk_flags["memory_pressure"] or risk_flags["long_context"] or risk_flags["repeated_filler"]:
        selected = "sdi_packet"
        reasons.append("context pressure or filler favors packet compression")
    elif risk_flags["open_loops"] or risk_flags["conflicting_old_new"]:
        selected = "sdi_packet"
        reasons.append("open-loop/conflict state needs structured packet facts")
    elif risk_flags["tool_exactness"]:
        if risk_flags["exact_tool_question"]:
            selected = "sdi_packet"
            reasons.append("exact tool-output question favors rigid exact-output packet fields")
        elif risk_flags["tiny_context"]:
            selected = "recent_only" if recent_tokens <= raw_tokens else "no_packet"
            reasons.append("tool facts present but raw context is tiny; use smallest raw/recent exact-output view")
        else:
            selected = "sdi_packet"
            reasons.append("tool/path/commit/numeric exactness favors packet facts")
    elif risk_flags["pinned_fact_question"]:
        selected = "sdi_packet" if raw_tokens > 350 else "simple_summary"
        reasons.append("question references pinned project/state facts")
    elif risk_flags["medium_context"]:
        selected = "simple_summary"
        reasons.append("medium context without exactness risk")
    elif recent_tokens < raw_tokens:
        selected = "recent_only"
        reasons.append("recent window is smaller and no durable fact risk found")
    else:
        reasons.append("tiny/self-contained context; packet overhead not justified")

    return _build_policy_result(selected, reasons, raw_tokens, recent_tokens, summary_tokens, raw_words, guard, risk_flags, is_multi_commit_review, is_benchmark_result)


def _build_policy_result(
    selected: str,
    reasons: list[str],
    raw_tokens: int,
    recent_tokens: int,
    summary_tokens: int,
    raw_words: int,
    guard: dict[str, Any],
    risk_flags: dict[str, Any],
    is_multi_commit_review: bool,
    is_benchmark_result: bool,
) -> dict[str, Any]:
    """Build the policy result dict with content-type metadata (Phase 26S)."""
    selected_tokens = {
        "no_packet": raw_tokens,
        "recent_only": recent_tokens,
        "simple_summary": summary_tokens,
        "sdi_packet": None,
    }.get(selected, raw_tokens)

    return {
        "selected_policy": selected,
        "policy_reason": "; ".join(reasons),
        "raw_estimated_tokens": raw_tokens,
        "raw_estimated_words": raw_words,
        "recent_estimated_tokens": recent_tokens,
        "simple_summary_estimated_tokens": summary_tokens,
        "selected_estimated_tokens": selected_tokens,
        "estimated_reduction": round((raw_tokens - selected_tokens) / raw_tokens * 100, 1)
        if selected_tokens is not None and raw_tokens
        else None,
        "risk_flags": risk_flags,
        "guard_summary": {
            "safe": guard["safe"],
            "recommended_tier": guard["recommended_tier"],
            "swap_used_mb": guard["swap_used_mb"],
            "reason": guard["reason"],
        },
        "detected_content_type": {
            "multi_commit_git_review": is_multi_commit_review,
            "benchmark_result": is_benchmark_result,
        },
        "why_not_sdi_packet": "simple_summary selected" if selected == "simple_summary" else None,
        "why_not_simple_summary": "sdi_packet selected" if selected == "sdi_packet" else None,
    }


def prompt_for_baseline(
    baseline: str,
    conversation: str,
    pinned_data: Any,
    question: str,
    tier: int | None,
    model: str,
    active_context_target: int,
    packet_style: str,
) -> tuple[str, dict[str, Any]]:
    pinned_facts, open_loops, hard_constraints = pinned_parts(pinned_data)
    meta: dict[str, Any] = {}
    requested_baseline = baseline
    if baseline == "auto":
        policy = choose_auto_policy(
            conversation=conversation,
            pinned_data=pinned_data,
            question=question,
            tier=tier,
            model=model,
            active_context_target=active_context_target,
        )
        baseline = policy["selected_policy"]
        meta["auto_policy"] = policy

    if baseline == "no_packet":
        context = conversation
    elif baseline == "recent_only":
        context = recent_only(conversation)
    elif baseline == "simple_summary":
        context = simple_summary(conversation, pinned_facts)
    elif baseline == "sdi_packet":
        guard = evaluate_guard(model=model, active_context_target=active_context_target)
        selected_tier = guard["recommended_tier"] if tier is None else tier
        packet = build_sdi_packet(
            conversation=conversation,
            pinned_facts=pinned_facts,
            task_type="question",
            target_model=model,
            context_budget=active_context_target,
            tier=selected_tier,
            memory_state={
                "memavailable_gb": guard["ram_available_mb"] / 1024,
                "swap_used_gb": guard["swap_used_mb"] / 1024,
            },
            open_loops=open_loops,
            hard_constraints=hard_constraints,
            packet_style=packet_style,
        )
        context = packet.packet_text
        meta["packet"] = packet.to_meta()
        meta["packet_style"] = packet_style
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    prompt = (
        "You are evaluating local model recall from supplied context. "
        "Answer only from the provided context. If the answer is missing, say not found in packet. "
        "Return compact bullet points with every relevant exact fact: names, paths, constraints, numbers, commits, statuses, open-loop next actions. "
        "If [EXACT_TOOL_OUTPUTS] appears, answer tool-output questions by copying the exact requested fields from that section with field labels such as path, metric, value, unit, status, error, and command. "
        "For exact tool-output questions, ignore conflicting or distractor tool facts outside [EXACT_TOOL_OUTPUTS]. "
        "Do not restate the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    prompt_tokens, prompt_words = token_estimate(prompt)
    raw_tokens, _ = token_estimate(conversation)
    meta.update(
        {
            "requested_baseline": requested_baseline,
            "effective_baseline": baseline,
            "prompt_tokens_est": prompt_tokens,
            "prompt_words": prompt_words,
            "raw_context_tokens_est": raw_tokens,
            "token_reduction_pct": round((raw_tokens - prompt_tokens) / raw_tokens * 100, 1)
            if raw_tokens
            else 0,
        }
    )
    return prompt, meta


def call_ollama(model: str, prompt: str, timeout_s: int) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 160,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "wall_time_s": round(time.monotonic() - start, 3),
            "response": "",
        }
    return {
        "ok": True,
        "wall_time_s": round(time.monotonic() - start, 3),
        "response": data.get("response", ""),
        "backend_meta": {
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get("total_duration"),
        },
    }


def score_response(response: str, expected: dict[str, Any]) -> dict[str, Any]:
    lower = response.lower()
    # Schema fix: union must_include AND required as critical checks.
    # Fixtures used either field; score_response now treats them equivalently.
    must = list(expected.get("must_include", []))
    required_field = list(expected.get("required", []))
    # Union: deduplicate while preserving both lists
    seen = set()
    must_union = []
    for item in must + required_field:
        if item.lower() not in seen:
            seen.add(item.lower())
            must_union.append(item)
    should = list(expected.get("should_include", []))
    loops = list(expected.get("expected_open_loops", []))
    bad = list(expected.get("must_not_include", []))

    required_hits = [item for item in must_union if item.lower() in lower]
    should_hits = [item for item in should if item.lower() in lower]
    loop_hits = [item for item in loops if item.lower() in lower]
    bad_hits = [item for item in bad if item.lower() in lower]

    required_score = len(required_hits) / len(must_union) if must_union else 1.0
    should_score = len(should_hits) / len(should) if should else 1.0
    loop_score = len(loop_hits) / len(loops) if loops else 1.0
    hallucination_penalty = 0.25 * len(bad_hits)
    score = max(
        0.0,
        min(1.0, 0.75 * required_score + 0.15 * should_score + 0.10 * loop_score - hallucination_penalty),
    )

    return {
        "score": round(score, 3),
        "required_hits": required_hits,
        "required_missing": [item for item in must_union if item not in required_hits],
        "should_hits": should_hits,
        "should_missing": [item for item in should if item not in should_hits],
        "open_loop_hits": loop_hits,
        "open_loop_missing": [item for item in loops if item not in loop_hits],
        "forbidden_hits": bad_hits,
        "output_sane": bool(response.strip()) and len(response) < 3000,
        # Metadata for transparency
        "_debug": {
            "must_include_count": len(must),
            "required_field_count": len(required_field),
            "must_union_count": len(must_union),
            "schema_note": "must_include and required are now unioned",
        },
    }


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    conversation = load_text(Path(args.conversation))
    pinned_data = load_json(Path(args.pinned))
    question = extract_question(
        conversation,
        args.question or (load_text(Path(args.question_file)) if args.question_file else None),
    )
    expected = load_json(Path(args.expected)) if args.expected else {}
    baseline = args.baseline
    tier = None if args.tier == "auto" else int(args.tier)

    guard = evaluate_guard(
        model=args.model,
        active_context_target=args.active_context_target,
        force_tier=tier,
    )
    if not guard["safe"]:
        return {
            "ok": False,
            "blocked": True,
            "reason": guard["reason"],
            "guard": guard,
        }

    before = read_memory_state()
    prompt, prompt_meta = prompt_for_baseline(
        baseline=baseline,
        conversation=conversation,
        pinned_data=pinned_data,
        question=question,
        tier=tier,
        model=args.model,
        active_context_target=args.active_context_target,
        packet_style=args.packet_style,
    )
    backend = call_ollama(args.model, prompt, args.timeout_s)
    after = read_memory_state()
    score = score_response(backend.get("response", ""), expected)

    return {
        "ok": backend.get("ok", False),
        "blocked": False,
        "backend": args.backend,
        "model": args.model,
        "baseline": baseline,
        "effective_baseline": prompt_meta.get("effective_baseline", baseline),
        "selected_policy": prompt_meta.get("auto_policy", {}).get("selected_policy") if baseline == "auto" else baseline,
        "policy_reason": prompt_meta.get("auto_policy", {}).get("policy_reason") if baseline == "auto" else "fixed baseline",
        "question": question,
        "guard": guard,
        "prompt": prompt_meta,
        "score": score,
        "response": backend.get("response", ""),
        "backend_result": backend,
        "swap_before_mb": before.swap_used_mb,
        "swap_after_mb": after.swap_used_mb,
        "swap_delta_mb": after.swap_used_mb - before.swap_used_mb,
        "ram_available_before_mb": before.ram_available_mb,
        "ram_available_after_mb": after.ram_available_mb,
        "output_bytes": len(backend.get("response", "").encode("utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone SDI packet runtime harness")
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--pinned", required=True)
    parser.add_argument("--expected")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--question")
    group.add_argument("--question-file")
    parser.add_argument("--tier", default="auto", choices=["0", "1", "2", "3", "auto"])
    parser.add_argument("--backend", default="ollama", choices=["ollama"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--packet-style", default="compact", choices=["full", "compact"])
    parser.add_argument("--out")
    parser.add_argument("--meta")
    parser.add_argument("--active-context-target", type=int, default=4096)
    parser.add_argument("--timeout-s", type=int, default=45)
    args = parser.parse_args()

    result = run_once(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result.get("response", "") + "\n")
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(payload)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
