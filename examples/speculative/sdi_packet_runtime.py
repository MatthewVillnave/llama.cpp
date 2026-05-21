#!/usr/bin/env python3
"""
Phase 26M: Standalone SDI packet runtime harness.

Runs raw-context baselines and SDI packet prompts against a clean local backend
without touching llama.cpp internals, PRT, OpenClaw, or routing layers.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sdi_memory_guard import evaluate_guard, read_memory_state
from sdi_packet_builder import build_sdi_packet, token_estimate


BASELINES = ("no_packet", "recent_only", "simple_summary", "sdi_packet")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def pinned_parts(pinned_data: Any) -> tuple[list[str], list[str], list[str]]:
    if isinstance(pinned_data, list):
        return pinned_data, [], []
    return (
        list(pinned_data.get("pinned_facts", [])),
        list(pinned_data.get("open_loops", [])),
        list(pinned_data.get("constraints_verbatim", [])),
    )


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


def prompt_for_baseline(
    baseline: str,
    conversation: str,
    pinned_data: Any,
    question: str,
    tier: int | None,
    model: str,
    active_context_target: int,
) -> tuple[str, dict[str, Any]]:
    pinned_facts, open_loops, hard_constraints = pinned_parts(pinned_data)
    meta: dict[str, Any] = {}

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
        )
        context = packet.packet_text
        meta["packet"] = packet.to_meta()
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    prompt = (
        "You are evaluating local model recall from supplied context. "
        "Answer only from the provided context. If the answer is missing, say UNKNOWN. "
        "Prefer exact names, paths, constraints, numbers, and status values.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    prompt_tokens, prompt_words = token_estimate(prompt)
    raw_tokens, _ = token_estimate(conversation)
    meta.update(
        {
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
                "num_predict": 96,
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
    must = expected.get("must_include", [])
    should = expected.get("should_include", [])
    loops = expected.get("expected_open_loops", [])
    bad = expected.get("must_not_include", [])

    required_hits = [item for item in must if item.lower() in lower]
    should_hits = [item for item in should if item.lower() in lower]
    loop_hits = [item for item in loops if item.lower() in lower]
    bad_hits = [item for item in bad if item.lower() in lower]

    required_score = len(required_hits) / len(must) if must else 1.0
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
        "required_missing": [item for item in must if item not in required_hits],
        "should_hits": should_hits,
        "should_missing": [item for item in should if item not in should_hits],
        "open_loop_hits": loop_hits,
        "open_loop_missing": [item for item in loops if item not in loop_hits],
        "forbidden_hits": bad_hits,
        "output_sane": bool(response.strip()) and len(response) < 3000,
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
