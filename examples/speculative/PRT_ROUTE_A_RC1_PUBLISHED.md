# PRT_ROUTE_A_RC1 — PUBLISH READY

**Status:** RC1 tagged, experimental branch created, push commands ready

---

## Repo State

| Item | Value |
|------|-------|
| Current branch | `experimental/prt-route-a-rc1` |
| HEAD commit | `aaa5f2902` |
| Tag | `PRT_ROUTE_A_RC1` — matches HEAD |
| Working tree | Dirty (uncommitted junk files — safe to leave) |
| Remote | `origin` → https://github.com/ggerganov/llama.cpp.git |
| Ahead of origin/master | 2 commits |

---

## Push Commands

Run these manually (GitHub auth not available on this machine):

```bash
# Push the experimental branch
git push -u origin experimental/prt-route-a-rc1

# Push the tag
git push origin PRT_ROUTE_A_RC1
```

After pushing, verify with:

```bash
# Verify branch pushed
git ls-remote --heads origin experimental/prt-route-a-rc1

# Verify tag pushed
git ls-remote --tags origin PRT_ROUTE_A_RC1

# Show branches
git branch -r | grep prt

# Show tags
git tag -l | grep PRT_ROUTE_A_RC1
```

Expected: branch and tag both appear in remote listing.

---

## What Was Done

1. ✓ Created `experimental/prt-route-a-rc1` branch from `PRT_ROUTE_A_RC1` tag
2. ✓ Switched to that branch (you are now on it)
3. ✓ Stash-popped working directory back (uncommitted files preserved)
4. ✓ Verified HEAD matches tag
5. ✓ Documented push commands

---

## What Was NOT Done

- ✗ GitHub push (no auth)
- ✗ Master/main merge
- ✗ Origin/master push (2 commits are local only)

---

## Post-Push Verification

After running the push commands, verify:

```bash
# Branch should appear
git ls-remote --heads origin experimental/prt-route-a-rc1
# Expected: 2 lines with refs/heads/experimental/prt-route-a-rc1

# Tag should appear
git ls-remote --tags origin PRT_ROUTE_A_RC1
# Expected: 1 line with refs/tags/PRT_ROUTE_A_RC1
```

---

## Exact Allowed Claim (for PR)

> PRT_ROUTE_A_RC1 is a tagged experimental CPU inference acceleration branch using Route A true FFN_UP graph replacement with L12/L15 native anchors. It achieved ~1.84x average speedup on the controlled mini-suite with clean counters, stable memory, and loud missing-sidecar validation.

## Forbidden Claims (for PR)

- ~~production-ready~~
- ~~universal CPU inference speedup~~
- ~~all prompts validated~~
- ~~full JSON/structured validation complete~~
- ~~pure all36 PRT is safe~~
- ~~no further testing needed~~

---

## Next Phase Recommendation

**Phase 12: Broader Validation**

After push:
1. Full mini-suite re-run on clean machine (validate tag reproducibility)
2. JSON/structured prompts at n=50 on >32GB machine
3. L13+L14 alternative fallback pair documented
4. Performance profiling (where is time spent?)

---

## Commit History in Branch

```
aaa5f2902 PRT Phase 11BP: Route A core — build_ffn hook, force-native API, sidecar validation, custom op
3763be222 PRT Phase 11BM-11BP: RC1 docs — Route A + L12/L15, ~1.84x speedup, sidecar validation
d12cc3d1c CUDA: also store `node->src->data` ptrs for equality check (#21635)
```

---

## Final Verdict

| Item | Status |
|------|--------|
| A. Branch created | ✓ YES |
| B. Tag exists | ✓ YES |
| C. HEAD matches tag | ✓ YES |
| D. Master untouched | ✓ YES |
| E. Push ready | ✓ YES (commands provided) |
| F. Remote verification | PENDING — run push commands first |

---

*Publish ready — awaiting manual git push*
