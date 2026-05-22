#!/usr/bin/env python3
"""
Phase 28B: External Context/Memory Probe Harness
================================================
Runs controlled Ollama prompts while externally measuring
RAM, swap, wall time, and output sanity across context sizes.

Design goals:
- No external deps beyond Python stdlib + psutil + curl
- All outputs default to /tmp, not staged to repo
- One run at a time, strict memory guards
- Designed for qwen2.5:0.5B and qwen2.5:3B only (no 7B)

Usage:
  python3 sdi_context_memory_probe.py --model qwen2.5:0.5b \
    --contexts 2048 4096 8192 \
    --prompt-kind medium \
    --out-json /tmp/probe_results.json \
    --out-md   /tmp/probe_results.md

Abort conditions:
  - swap used > 1GB before a run
  - swap delta > 250MB after a run
  - available RAM < 3GB
  - timeout / stale process
  - Ollama unstable
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL      = "http://localhost:11434"
TIMEOUT_S     = 300
TEMP          = 0
MAX_TOKS      = 64
SWAP_GUARD    = 1.0     # GB
RAM_MIN_FREE  = 3.0     # GB
OUT_DEFAULT  = "/tmp"

# ── Prompts ─────────────────────────────────────────────────────────────────
PROMPTS = {
    "tiny": {
        "description": "Simple one-line question",
        "prompt": "Return only the capital of France.",
        "answer_key": "Paris",
    },
    "medium": {
        "description": "~2K token filler + tracking fact + question",
        "prompt": None,  # built by build_medium()
        "answer_key": "SDI-2026-Q2-v011",
    },
    "structured": {
        "description": "Pinned facts + constraint + open loop + filler",
        "prompt": None,  # built by build_structured()
        "answer_key": "SDI-2026-Q2-v011",
    },
}

def build_medium(target_chars=2000):
    base = " DEBUG LOG: Runtime telemetry confirms stable swap utilization. "
    fill = (base * ((target_chars // len(base)) + 2))[:target_chars]
    return (
        f"{fill}\n\n"
        f"SDI Runtime tracking ID: SDI-2026-Q2-v011.\n"
        f"{fill}\n\n"
        f"What is the SDI Runtime tracking ID?"
    )

def build_structured(target_chars=2000):
    base = " DEBUG LOG: Runtime telemetry confirms stable swap utilization across bounded tests. "
    fill = (base * ((target_chars // len(base)) + 2))[:target_chars]
    return (
        "You are a helpful assistant. Answer based on the provided context.\n\n"
        "## System Context\n"
        "Project: NEXUS | Framework: Villnave's Law | Phase: SDI Runtime v0.1.1\n\n"
        "## Recent Conversation\n"
        f"{fill}\n\n"
        "## Pinned Facts\n"
        "SDI Runtime v0.1.1 is a context-selection layer for CPU/RAM-constrained inference.\n"
        "Tracking ID: SDI-2026-Q2-v011.\n\n"
        "## Open Loops\n"
        "Check Villnave's Law canonical reference for update frequency.\n\n"
        "## User Query\n"
        "What is the SDI Runtime tracking ID? What persistence framework does Villnave's Law govern?"
    )

# ── Memory helpers ─────────────────────────────────────────────────────────
def snap():
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return dict(
        ram_avail_gb  = round(vm.available  / 1e9, 2),
        ram_total_gb  = round(vm.total       / 1e9, 1),
        swap_used_gb  = round(swap.used       / 1e9, 3),
        swap_total_gb = round(swap.total      / 1e9, 1),
    )

def check_guard(snap, label="pre"):
    ok = snap["swap_used_gb"] <= SWAP_GUARD and snap["ram_avail_gb"] >= RAM_MIN_FREE
    status = "OK" if ok else "ABORT"
    print(f"  [{label}] RAM={snap['ram_avail_gb']}GB  swap={snap['swap_used_gb']}GB  → {status}")
    return ok

# ── Ollama call ──────────────────────────────────────────────────────────────
def ollama_gen(prompt, model, ctx, maxt=MAX_TOKS):
    payload = dict(
        model   = model,
        prompt  = prompt,
        stream  = False,
        options = dict(temperature=TEMP, num_predict=maxt, num_ctx=ctx),
    )
    t0  = time.time()
    cmd = [
        "curl", "-s", "--max-time", str(TIMEOUT_S),
        "-X", "POST", f"{BASE_URL}/api/generate",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S + 15)
    elapsed = round(time.time() - t0, 3)

    result = dict(elapsed_s=elapsed, prompt_chars=len(prompt),
                  prompt_tokens_est=len(prompt) // 4,
                  curl_returncode=r.returncode, curl_stderr=r.stderr[:200],
                  model=model, ctx=ctx)

    try:
        dj = json.loads(r.stdout)
        result["json_parsed"]      = True
        result["response"]         = dj.get("response", "")
        result["done"]             = dj.get("done")
        result["done_reason"]       = dj.get("done_reason", "")
        result["prompt_eval_count"] = dj.get("prompt_eval_count")
        result["eval_count"]       = dj.get("eval_count")
        result["eval_duration"]    = dj.get("eval_duration")
        result["total_duration"]   = dj.get("total_duration")
        result["http_ok"]          = True
    except json.JSONDecodeError:
        result["json_parsed"] = False
        result["response"]    = ""
        result["http_ok"]     = False

    return result

def score_response(response_text, answer_key):
    return answer_key in response_text

# ── Probe runner ────────────────────────────────────────────────────────────
def run_probe(model, ctx, prompt_kind):
    info = PROMPTS.get(prompt_kind, PROMPTS["medium"])
    if prompt_kind == "medium":
        prompt = build_medium()
    elif prompt_kind == "structured":
        prompt = build_structured()
    else:
        prompt = info["prompt"]

    answer_key = info["answer_key"]
    print(f"\n  [{model}] c={ctx}  kind={prompt_kind}  ({len(prompt)} chars)")
    pre = snap()
    if not check_guard(pre, "pre"):
        return dict(status="ABORT_PRE", pre=pre, post=None, prompt_kind=prompt_kind)

    resp = ollama_gen(prompt, model, ctx)
    post = snap()

    result = dict(
        model=model, ctx=ctx, prompt_kind=prompt_kind,
        status="OK" if resp.get("http_ok") else "FAIL",
        pre=pre, post=post,
        swap_delta_gb=round(post["swap_used_gb"] - pre["swap_used_gb"], 3),
        elapsed_s=resp.get("elapsed_s"),
        response=resp.get("response", ""),
        sane=score_response(resp.get("response", ""), answer_key),
        prompt_eval=resp.get("prompt_eval_count"),
        eval_count=resp.get("eval_count"),
        eval_duration=resp.get("eval_duration"),
        done=resp.get("done"),
        done_reason=resp.get("done_reason"),
        curl_returncode=resp.get("curl_returncode"),
    )

    if resp.get("eval_count"):
        ms_per_tok = round(resp["eval_duration"] / resp["eval_count"] / 1e6, 2) if resp["eval_duration"] else None
        result["ms_per_token"] = ms_per_tok

    swap_ok  = abs(result["swap_delta_gb"]) <= 0.25
    mem_ok   = post["ram_avail_gb"] >= RAM_MIN_FREE
    sane_ok  = result["sane"]
    http_ok  = resp.get("http_ok", False)

    if not swap_ok:
        result["status"] = "ABORT_SWAP"
    elif not mem_ok:
        result["status"] = "ABORT_RAM"
    elif not http_ok:
        result["status"] = "FAIL_HTTP"

    print(f"  [post] RAM={post['ram_avail_gb']}GB  swap={post['swap_used_gb']}GB  Δ={result['swap_delta_gb']}GB")
    print(f"  → status={result['status']}  sane={sane_ok}  eval={resp.get('eval_count','?')}  "
          f"{resp.get('elapsed_s','?')}s  response: {resp.get('response','')[:60].strip()}")
    return result

# ── Main matrix ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Phase 28B: Context/Memory Probe Harness")
    ap.add_argument("--model",       required=True,
                    help="Model name, e.g. qwen2.5:0.5b")
    ap.add_argument("--contexts",     nargs="+", type=int, default=[2048, 4096, 8192],
                    help="Context sizes to probe")
    ap.add_argument("--prompt-kind",  choices=["tiny", "medium", "structured"],
                    default="medium", help="Prompt type")
    ap.add_argument("--out-json",     default=f"{OUT_DEFAULT}/probe_results.json")
    ap.add_argument("--out-md",       default=f"{OUT_DEFAULT}/probe_results.md")
    ap.add_argument("--timeout",      type=int, default=TIMEOUT_S,
                    help="Max seconds per request")
    args = ap.parse_args()

    print(f"=== Phase 28B Context/Memory Probe ===")
    print(f"Model:    {args.model}")
    print(f"Contexts: {args.contexts}")
    print(f"Prompt:   {args.prompt_kind}")
    print(f"Output:   {args.out_json}")

    results = []
    for ctx in args.contexts:
        res = run_probe(args.model, ctx, args.prompt_kind)
        results.append(res)
        if res["status"] != "OK":
            print(f"  ⚠ aborting — status={res['status']}")
            break

    # Write JSON
    out = dict(model=args.model, prompt_kind=args.prompt_kind,
              contexts=args.contexts, results=results)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON → {args.out_json}")

    # Write MD
    md_lines = [f"# Phase 28B: {args.model} Context/Memory Probe",
               f"| Context | Status | Swap Δ | RAM avail | Elapsed | Tokens | sane | Notes |",
               f"|---------|--------|--------|----------|---------|--------|------|-------|"]
    for r in results:
        note = r.get("done_reason","") if r.get("done_reason") else ("OK" if r["status"]=="OK" else r["status"])
        md_lines.append(
            f"| c={r['ctx']} | {r['status']} | {r['swap_delta_gb']}GB | "
            f"{r['pre']['ram_avail_gb']}→{r['post']['ram_avail_gb']}GB | "
            f"{r.get('elapsed_s','?')}s | {r.get('eval_count','?')} | "
            f"{r.get('sane')} | {note} |"
        )
    with open(args.out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"MD   → {args.out_md}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'ctx':<6} {'status':<12} {'Δswap':<8} {'RAM post':<10} {'sane':<5} {'tok':<4} {'elapsed'}")
    for r in results:
        print(f"{r['ctx']:<6} {r['status']:<12} {r['swap_delta_gb']:<8} "
              f"{r['post']['ram_avail_gb']:<10} {r.get('sane',False)!s:<5} "
              f"{r.get('eval_count','?'):<4} {r.get('elapsed_s','?')}s")
    return results

if __name__ == "__main__":
    main()