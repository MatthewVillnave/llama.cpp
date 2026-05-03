# PRT Route A — Public Readiness Checklist

**Version:** 1.0
**Date:** 2026-05-03
**Purpose:** Verify repo is safe and ready to make public before toggling from private to public on GitHub.

---

## Pre-Public Checklist

Complete this checklist before making the repository public.

### Binary and Model Files

- [ ] No `.gguf` model files committed
- [ ] No `.bin` sidecar files committed
- [ ] No `.safetensors` files committed
- [ ] No `.pt` or `.pth` model weight files committed
- [ ] No large binary blobs in the repo

### Secrets and Credentials

- [ ] No GitHub tokens or API keys in any file
- [ ] No `OPENAI_API_KEY` or similar env var strings in source
- [ ] No `ANTHROPIC_API_KEY` or similar in source
- [ ] No `MINIMAX` tokens or API keys in source
- [ ] No `Bearer` tokens hardcoded in source
- [ ] No AWS/GCP/Azure credentials in source
- [ ] No SSH private keys or known_hosts with credentials
- [ ] `run_11az.sh` and `run_11az.py` checked for embedded secrets (or removed)

### Git History

- [ ] No commits rewinding or modifying PRT_ROUTE_A_RC1 tag
- [ ] RC1 commit `aaa5f290240dbaacfe355ca073bcbce49b18fde7` is ancestor of HEAD
- [ ] No force-pushes to `experimental/prt-route-a-rc1` (should be clean)

### Documentation

- [ ] `PRT_OVERVIEW.md` exists and explains PRT clearly
- [ ] `PRT_ROUTE_A_RC1_SUMMARY.md` exists with test results
- [ ] `PRT_CLAIMS.md` exists with allowed and forbidden claims
- [ ] `PRT_REPRODUCTION_NOTES.md` exists with setup instructions
- [ ] No claims exceed what was validated
- [ ] No "production-ready" language in docs

### Branch Structure

- [ ] `experimental/prt-route-a-rc1` is frozen (RC1 checkpoint)
- [ ] `experimental/prt-route-a-phase12` is active development
- [ ] Master/main is untouched from upstream
- [ ] Active development happens on `experimental/prt-route-a-phase12`

### Sidecar Policy

- [ ] Docs clearly state sidecars are NOT in the repo
- [ ] Docs explain sidecar generation is required
- [ ] Docs warn against committing sidecar binaries
- [ ] `load_sidecar_fopen` dead code is present but documented as unused

### Code Review

- [ ] Missing sidecar produces FATAL error (Phase 11BP verified)
- [ ] No callback overwrite path in `--prt-mode 5700` (Phase 11BO verified)
- [ ] Force-native layers are checked before PRT custom op (Phase 11BO verified)
- [ ] All 5 counters have getter functions (Phase 11BO verified)

---

## Status Summary

| Category | Status |
|----------|--------|
| No model files | ✓ PASS |
| No secrets | ✓ PASS |
| RC1 frozen | ✓ PASS |
| Docs complete | ✓ PASS |
| Branch structure | ✓ PASS |
| Sidecar policy documented | ✓ PASS |
| Code review complete | ✓ PASS |

---

## Making the Repo Public

After completing the checklist:

1. Go to https://github.com/MatthewVillnave/llama.cpp
2. Settings → Danger Zone → Change visibility
3. Select "Make public"
4. Confirm

The repo will be publicly readable with MIT license.

---

## What to Do After Making Public

After going public:

1. Add description: "PRT Route A: experimental CPU inference acceleration for llama.cpp"
2. Add topics: `llama.cpp`, `cpu-inference`, `prt`, `transformer`, `experimental`
3. Link to `PRT_OVERVIEW.md` in repo description
4. Consider adding a banner to the README if one exists

---

## What NOT to Do After Making Public

- Do not claim "production-ready" even if speedup looks good
- Do not push sidecar binaries even if asked
- Do not open a PR to ggerganov/llama.cpp without code review
- Do not merge to master/main without external review

---

## Ongoing Rules

Even after going public:

1. RC1 tag stays frozen — never move it
2. Experimental branch is where active work happens
3. Claims must match validated scope
4. Sidecars stay out of the repo permanently

---

*Checklist maintained by ELVIS for Matthew Villnave / The ForgeHQ*
