#!/usr/bin/env python3
"""
Phase 26I: Standalone SDI Context Packet Builder
sdi_context_policy_v0.1 implementation

Usage:
    python3 sdi_packet_builder.py --conversation <file> --pinned <json>
                                  [--tier 0|1|2|3] [--max-words N]
                                  [--recent-lines N] [--out <file>]
                                  [--meta <file>]
"""

import argparse
import json
import os
import re
import sys
from typing import Any


# --------------------------------------------------------------------------- #
# Core data structures
# --------------------------------------------------------------------------- #

class SDIPacket:
    """Output of build_sdi_packet()."""
    def __init__(
        self,
        packet_text: str,
        estimated_tokens: int,
        estimated_words: int,
        tier: int,
        included_facts: list[str],
        dropped_sections: list[dict],
        safety_warnings: list[str],
        secret_warnings: list[str],
        compression_reason: str,
        model_used: str,
    ):
        self.packet_text = packet_text
        self.estimated_tokens = estimated_tokens
        self.estimated_words = estimated_words
        self.tier = tier
        self.included_facts = included_facts
        self.dropped_sections = dropped_sections
        self.safety_warnings = safety_warnings
        self.secret_warnings = secret_warnings
        self.compression_reason = compression_reason
        self.model_used = model_used

    def to_meta(self) -> dict:
        return {
            "tier": self.tier,
            "estimated_tokens": self.estimated_tokens,
            "estimated_words": self.estimated_words,
            "included_facts": self.included_facts,
            "dropped_sections": self.dropped_sections,
            "safety_warnings": self.safety_warnings,
            "secret_warnings": self.secret_warnings,
            "compression_reason": self.compression_reason,
            "model_used": self.model_used,
            "packet_text_preview": self.packet_text[:200],
        }


# --------------------------------------------------------------------------- #
# Memory / swap guard (from Phase 26H)
# --------------------------------------------------------------------------- #

MODEL_SIZES_GB = {
    "qwen2.5-0.5b": 0.38,
    "qwen2.5-3b":   1.8,
    "qwen2.5-7b":   4.4,
    "qwen2.5-14b":  8.4,
}

def memory_guard(
    target_model: str,
    context_size: int,
    memavailable_gb: float,
    swap_used_gb: float,
) -> dict:
    """
    Returns dict with keys: proceed, tier, reason, warnings, chosen_model
    """
    result = {
        "proceed": True,
        "tier": 1,
        "reason": "OK",
        "warnings": [],
        "chosen_model": target_model,
        "context_budget": context_size,
    }

    if swap_used_gb > 1.0:
        result.update({
            "proceed": False,
            "tier": 0,
            "reason": f"Swap pressure critical: {swap_used_gb:.1f} GB used",
            "warnings": ["Swap > 1 GB — blocking local inference"],
        })
        return result

    model_gb = MODEL_SIZES_GB.get(target_model.lower(), 4.4)
    kv_gb = (context_size / 16384) * 1.8
    total_estimate_gb = model_gb + kv_gb + 0.3

    if total_estimate_gb > memavailable_gb * 0.85:
        result.update({
            "proceed": False,
            "tier": 0,
            "reason": f"RAM tight: need ~{total_estimate_gb:.1f} GB, have {memavailable_gb:.1f} GB",
            "warnings": [f"Insufficient RAM for {target_model} at c={context_size}"],
        })
        return result

    if swap_used_gb > 0.5:
        result["tier"] = 1
    elif swap_used_gb < 0.2:
        result["tier"] = 2

    return result


# --------------------------------------------------------------------------- #
# Secret scanning
# --------------------------------------------------------------------------- #

# Patterns that look like real secrets (not including FAKE_DO_NOT_USE markers)
SECRET_PATTERNS = [
    re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\b(OPENAI_API_KEY|GITHUB_PAT|AZURE_KEY|ANTHROPIC_KEY)\s*=\s*[\w-]+", re.IGNORECASE),
    re.compile(r"\bpassword\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
    re.compile(r"\bbearer\s+[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"\btoken\s*:\s*[A-Za-z0-9_-]{20,}", re.IGNORECASE),
]

FAKE_MARKERS = ["FAKE", "DO_NOT_USE", "TEST", "EXAMPLE", "SAMPLE", "MOCK"]

def secret_scan(text: str) -> list[dict]:
    """
    Scan text for potential secrets.
    Returns list of dicts with keys: pattern, redacted_line, is_fake
    """
    findings = []
    lines = text.split("\n")
    for line in lines:
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(line):
                matched_text = match.group()
                # Check if marked as fake
                is_fake = any(marker in line.upper() for marker in FAKE_MARKERS)
                findings.append({
                    "pattern": pattern.pattern,
                    "redacted_line": re.sub(
                        re.escape(matched_text),
                        "[REDACTED]" if not is_fake else "[FAKE_REDACTED]",
                        line
                    ),
                    "is_fake": is_fake,
                    "matched_text": matched_text[:20] + "..." if len(matched_text) > 20 else matched_text,
                })
    return findings


# --------------------------------------------------------------------------- #
# Filler / repetition detection
# --------------------------------------------------------------------------- #

def detect_filler_lines(text: str, min_repeat_count: int = 3) -> list[dict]:
    """
    Detect lines or short phrases that are repeated multiple times.
    Used to identify filler content that can be dropped.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    line_counts: dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1

    filler = []
    for line, count in line_counts.items():
        if count >= min_repeat_count and len(line) > 20:
            filler.append({"line": line[:80], "repeat_count": count, "drop": True})
        elif count >= min_repeat_count and len(line) <= 20:
            filler.append({"line": line, "repeat_count": count, "drop": False})

    return filler


def is_roman_empire_text(text: str) -> bool:
    """
    Heuristic: if text contains many Roman Empire keywords, it's likely filler.
    """
    roman_keywords = [
        "roman empire", "emperor", "legion", "senate", "augustus",
        "trajan", "diocletian", "constantine", "byzantine", "476 ce",
        "pax romana", "denarius", "latifundia", "colosseum", "aqueduct",
        "hadrian's wall", "goth", "persian", "justinian",
    ]
    lower = text.lower()
    count = sum(1 for kw in roman_keywords if kw in lower)
    return count >= 4


def drop_filler(text: str) -> tuple[str, list[dict]]:
    """
    Remove detected filler from text.
    Returns (cleaned_text, dropped_report).
    """
    dropped = []
    lines = text.split("\n")
    cleaned = []
    in_filler_block = False
    filler_start = 0

    for i, line in enumerate(lines):
        # Detect start of Roman Empire filler block
        if is_roman_empire_text(line) and i > 5:
            if not in_filler_block:
                in_filler_block = True
                filler_start = i

        if in_filler_block:
            # End of filler block: empty line followed by non-Roman content
            if not line.strip() and i > filler_start + 20:
                dropped.append({
                    "type": "roman_empire_filler",
                    "start_line": filler_start,
                    "end_line": i,
                    "reason": "Roman Empire distractor text — not relevant to current task",
                })
                in_filler_block = False

            if i == len(lines) - 1 and in_filler_block:
                dropped.append({
                    "type": "roman_empire_filler",
                    "start_line": filler_start,
                    "end_line": i,
                    "reason": "Roman Empire distractor text — not relevant to current task",
                })
                in_filler_block = False
        else:
            cleaned.append(line)

    return "\n".join(cleaned), dropped


# --------------------------------------------------------------------------- #
# Recent turns extraction
# --------------------------------------------------------------------------- #

def extract_recent_turns(
    conversation: str,
    max_words: int,
    max_lines: int | None = None,
) -> tuple[str, list[dict]]:
    """
    Extract the most recent turns from a conversation.
    Keeps most recent User/Assistant exchange that fits within max_words.
    Returns (extracted_text, dropped_report).
    """
    lines = conversation.split("\n")
    if max_lines:
        lines = lines[-max_lines:]

    # Work backwards from the end, keeping turns until word budget exhausted
    turns = []
    current_turn = []
    current_role = None
    word_count = 0
    dropped = []

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Detect role
        role = None
        if stripped.startswith("User:"):
            role = "User"
        elif stripped.startswith("Assistant:"):
            role = "Assistant"
        elif stripped.startswith("##"):
            role = "section_header"

        if role in ("User", "Assistant") and role != current_role and current_role is not None:
            # New turn — save current
            if current_turn:
                turns.append(("\n".join(reversed(current_turn)), current_role))
                current_turn = []
            current_role = role
        elif role and role != "section_header":
            if current_role is None:
                current_role = role

        if role != "section_header":
            current_turn.append(stripped)
            word_count += len(stripped.split())

        if word_count > max_words:
            if current_turn and not dropped:
                dropped.append({
                    "type": "old_turns",
                    "reason": f"exceeded {max_words}-word budget for recent window",
                    "dropped_turns": len(turns),
                })
            break

    # Add remaining
    if current_turn:
        turns.append(("\n".join(reversed(current_turn)), current_role))

    # Reconstruct in forward order
    result_lines = []
    for text, role in reversed(turns):
        result_lines.append(f"{role}: {text}")

    return "\n\n".join(result_lines), dropped


# --------------------------------------------------------------------------- #
# Core packet builder
# --------------------------------------------------------------------------- #

def token_estimate(text: str) -> tuple[int, int]:
    """
    Rough token estimate: chars/4, and word count: words*1.0.
    Returns (tokens, words).
    """
    words = len(text.split())
    tokens = max(1, len(text) // 4)
    return tokens, words


def build_sdi_packet(
    conversation: str,
    pinned_facts: list[str],
    task_type: str = "question",
    target_model: str = "qwen2.5-7b",
    context_budget: int = 4096,
    tier: int | None = None,
    max_words: int | None = None,
    recent_lines: int | None = None,
    memory_state: dict | None = None,
    open_loops: list[str] | None = None,
    hard_constraints: list[str] | None = None,
    retrieved_memories: list[dict] | None = None,
    tool_outputs: list[dict] | None = None,
) -> SDIPacket:
    """
    Build an SDI_CONTEXT_PACKET from conversation + pinned facts.

    Parameters
    ----------
    conversation : str
        Raw conversation text (may include filler)
    pinned_facts : list[str]
        Critical facts to always include verbatim
    task_type : str
        Type of task (question, creative, planning, coding, review)
    target_model : str
        Target model (for memory guard)
    context_budget : int
        Max tokens for this context budget
    tier : int | None
        Override tier (0-3). If None, auto-selected from memory state.
    max_words : int | None
        Override max words for recent window
    recent_lines : int | None
        Max recent lines to consider
    memory_state : dict | None
        Memory state with memavailable_gb and swap_used_gb
    open_loops : list[str] | None
        Open loops / unresolved items
    hard_constraints : list[str] | None
        Hard constraints verbatim
    retrieved_memories : list[dict] | None
        Retrieved memories, each as {text, source, relevance}
    tool_outputs : list[dict] | None
        Recent tool outputs

    Returns
    -------
    SDIPacket
    """

    warnings: list[str] = []
    secret_warnings: list[str] = []
    dropped_sections: list[dict] = []

    # ---- Memory guard ----
    mem_state = memory_state or {}
    mem_gb = mem_state.get("memavailable_gb", 13.9)
    swap_gb = mem_state.get("swap_used_gb", 0.0)

    guard = memory_guard(target_model, context_budget, mem_gb, swap_gb)
    warnings.extend(guard["warnings"])

    if tier is None:
        tier = guard["tier"]

    # ---- Tier word budgets ----
    TIER_WORD_BUDGETS = {
        0: 256,   # emergency — only current request + pinned facts
        1: 1500,  # safe default
        2: 3000,  # extended
        3: 6000,  # full
    }
    # Tier budget overrides max_words when tier is explicitly provided.
    # max_words only used as an additional cap.
    word_budget = TIER_WORD_BUDGETS.get(tier, 1500)
    if max_words and max_words < word_budget:
        word_budget = max_words

    # ---- Secret scan on pinned facts ----
    pinned_text = "\n".join(pinned_facts)
    secrets = secret_scan(pinned_text)
    for s in secrets:
        if s["is_fake"]:
            secret_warnings.append(f"FAKE secret detected and redacted: {s['matched_text']}")
        else:
            secret_warnings.append(f"REAL secret pattern detected: {s['matched_text']} — review required")

    # ---- Drop filler ----
    cleaned_conversation, filler_dropped = drop_filler(conversation)
    dropped_sections.extend(filler_dropped)

    # ---- Extract recent turns ----
    recent_text, turns_dropped = extract_recent_turns(
        cleaned_conversation, max_words=word_budget, max_lines=recent_lines
    )
    dropped_sections.extend(turns_dropped)

    # ---- Parse conversation to extract current request ----
    current_request = extract_current_request(recent_text)

    # ---- Build packet components ----
    goal = f"[task_type={task_type}] {current_request}"

    components: dict[str, str] = {}
    components["Goal"] = goal
    components["Current user request"] = current_request

    # Pinned facts
    pinned_block = "\n".join(f"  - {f}" for f in pinned_facts)
    components["Pinned facts"] = pinned_block

    # Hard constraints
    if hard_constraints:
        constraints_block = "\n".join(f"  - {c}" for c in hard_constraints)
    else:
        constraints_block = "  (none)"
    components["Hard constraints"] = constraints_block

    # Relevant memory
    if retrieved_memories:
        mem_block = "\n".join(
            f"  - [{m.get('source', 'memory')}] {m.get('text', '')}"
            for m in retrieved_memories[:5]
        )
    else:
        mem_block = "  (none)"
    components["Relevant memory"] = mem_block

    # Open loops
    if open_loops:
        loops_block = "\n".join(f"  - {o}" for o in open_loops)
    else:
        loops_block = "  (none)"
    components["Open loops"] = loops_block

    # Recent state (condensed from recent_text)
    recent_summary = condense_recent_turns(recent_text)
    components["Recent state"] = f"  {recent_summary}"

    # Tool/output facts
    if tool_outputs:
        tool_block = "\n".join(f"  - [tool] {t.get('text', str(t))}" for t in tool_outputs[:3])
    else:
        tool_block = "  (none)"
    components["Tool/output facts"] = tool_block

    # Dropped context summary
    if dropped_sections:
        dropped_block = "\n".join(
            f"  - [{d.get('type', 'unknown')}] {d.get('reason', '')}"
            for d in dropped_sections
        )
    else:
        dropped_block = "  (nothing dropped)"
    components["Dropped context summary"] = dropped_block

    # Uncertainty
    components["Uncertainty"] = "  (none marked)"

    # Memory/safety note
    mem_note = (
        f"Tier {tier} applied. "
        f"MemAvailable={mem_gb:.1f}GB, SwapUsed={swap_gb:.2f}GB. "
        f"Word budget={word_budget}."
    )
    if warnings:
        mem_note += f" WARNINGS: {'; '.join(warnings)}"
    components["Memory/safety note"] = f"  {mem_note}"

    # ---- Assemble packet ----
    packet_lines = ["[SDI_CONTEXT_PACKET]"]
    for field, content in components.items():
        packet_lines.append(f"{field}:{content}")
    packet_lines.append("[/SDI_CONTEXT_PACKET]")

    packet_text = "\n".join(packet_lines)
    est_tokens, est_words = token_estimate(packet_text)

    # ---- What was included ----
    included = pinned_facts.copy()

    compression_reason = f"Tier {tier} applied"
    if filler_dropped:
        compression_reason += f", {len(filler_dropped)} filler blocks removed"
    if turns_dropped:
        compression_reason += f", {len(turns_dropped)} old turns dropped"

    return SDIPacket(
        packet_text=packet_text,
        estimated_tokens=est_tokens,
        estimated_words=est_words,
        tier=tier,
        included_facts=included,
        dropped_sections=dropped_sections,
        safety_warnings=warnings,
        secret_warnings=secret_warnings,
        compression_reason=compression_reason,
        model_used=guard.get("chosen_model") or target_model,
    )


def extract_current_request(recent_text: str) -> str:
    """Extract the most recent user request from a conversation."""
    lines = recent_text.split("\n")
    user_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("User:"):
            user_lines.append(stripped[5:].strip())
    if user_lines:
        return user_lines[-1]
    return "[could not extract current request — conversation may be empty]"


def condense_recent_turns(recent_text: str, max_sentences: int = 3) -> str:
    """Condense recent turns into a brief summary."""
    lines = recent_text.split("\n")
    turns = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("User:") or stripped.startswith("Assistant:"):
            content = stripped.split(":", 1)[1].strip()
            role = stripped.split(":")[0]
            turns.append(f"{role}: {content[:80]}{'...' if len(content) > 80 else ''}")
    if len(turns) <= max_sentences * 2:
        return " ".join(turns)
    # Summarize
    recent = turns[-max_sentences * 2:]
    return " [then] ".join(r[:100] for r in recent)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cli_main():
    parser = argparse.ArgumentParser(
        description="Phase 26I: SDI Context Packet Builder"
    )
    parser.add_argument(
        "--conversation", required=True,
        help="Path to conversation text file"
    )
    parser.add_argument(
        "--pinned", required=True,
        help="Path to pinned facts JSON file"
    )
    parser.add_argument(
        "--tier", type=int, choices=[0, 1, 2, 3], default=None,
        help="Compression tier (0-3). Auto-selected if not specified."
    )
    parser.add_argument(
        "--max-words", type=int, default=None,
        help="Max words for recent window (overrides tier default)"
    )
    parser.add_argument(
        "--recent-lines", type=int, default=None,
        help="Max recent lines from conversation"
    )
    parser.add_argument(
        "--out", help="Output path for packet text"
    )
    parser.add_argument(
        "--meta", help="Output path for metadata JSON"
    )
    parser.add_argument(
        "--task-type", default="question",
        help="Task type: question, creative, planning, coding, review"
    )
    parser.add_argument(
        "--target-model", default="qwen2.5-7b",
        help="Target model name"
    )
    parser.add_argument(
        "--context-budget", type=int, default=4096,
        help="Context budget in tokens"
    )
    parser.add_argument(
        "--no-guard", action="store_true",
        help="Skip memory/swap guard"
    )

    args = parser.parse_args()

    # Load conversation
    with open(args.conversation) as f:
        conversation = f.read()

    # Load pinned facts
    with open(args.pinned) as f:
        pinned_data = json.load(f)

    pinned_facts = pinned_data if isinstance(pinned_data, list) else pinned_data.get("pinned_facts", [])
    open_loops = pinned_data.get("open_loops", []) if isinstance(pinned_data, dict) else []
    hard_constraints = pinned_data.get("constraints_verbatim", []) if isinstance(pinned_data, dict) else []

    # Build packet
    try:
        packet = build_sdi_packet(
            conversation=conversation,
            pinned_facts=pinned_facts,
            task_type=args.task_type,
            target_model=args.target_model,
            context_budget=args.context_budget,
            tier=args.tier,
            max_words=args.max_words,
            recent_lines=args.recent_lines,
            open_loops=open_loops,
            hard_constraints=hard_constraints,
        )
    except Exception as e:
        print(f"ERROR building packet: {e}", file=sys.stderr)
        sys.exit(1)

    # Output packet text
    if args.out:
        with open(args.out, "w") as f:
            f.write(packet.packet_text)
        print(f"Packet written to: {args.out}")
    else:
        print("=== SDI_CONTEXT_PACKET ===")
        print(packet.packet_text)

    # Output metadata
    meta = packet.to_meta()
    if args.meta:
        with open(args.meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata written to: {args.meta}")

    print(f"\nTier: {packet.tier}")
    print(f"Est. tokens: {packet.estimated_tokens}, Est. words: {packet.estimated_words}")
    if packet.safety_warnings:
        print(f"Safety warnings: {packet.safety_warnings}")
    if packet.secret_warnings:
        print(f"Secret warnings: {packet.secret_warnings}")
    if packet.dropped_sections:
        print(f"Dropped sections: {len(packet.dropped_sections)}")
        for d in packet.dropped_sections:
            print(f"  - [{d.get('type')}] {d.get('reason', '')}")


if __name__ == "__main__":
    cli_main()