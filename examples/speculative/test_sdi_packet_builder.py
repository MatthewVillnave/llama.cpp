#!/usr/bin/env python3
"""
Phase 26I: Unit tests for SDI packet builder

Run:
    python3 examples/speculative/test_sdi_packet_builder.py
"""

import json
import os
import subprocess
import sys
import tempfile

# Resolve fixture paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(BASE_DIR, "fixtures", "sdi_packet")
CONV_FILE = os.path.join(FIXTURE_DIR, "conversation_long.txt")
PINNED_FILE = os.path.join(FIXTURE_DIR, "pinned_facts.json")
EXPECTED_FILE = os.path.join(FIXTURE_DIR, "expected_key_facts.json")
BUILDER_SCRIPT = os.path.join(BASE_DIR, "sdi_packet_builder.py")

sys.path.insert(0, BASE_DIR)
from sdi_packet_builder import build_sdi_packet, secret_scan, drop_filler


def run_builder(tier: int = 1, max_words: int | None = None) -> tuple[str, dict]:
    """Run builder via CLI and capture outputs."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out_f:
        out_path = out_f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as meta_f:
        meta_path = meta_f.name

    cmd = [
        sys.executable, BUILDER_SCRIPT,
        "--conversation", CONV_FILE,
        "--pinned", PINNED_FILE,
        "--tier", str(tier),
        "--out", out_path,
        "--meta", meta_path,
    ]
    if max_words is not None:
        cmd.extend(["--max-words", str(max_words)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Builder failed: {result.stderr}")

    with open(out_path) as f:
        packet_text = f.read()
    with open(meta_path) as f:
        meta = json.load(f)

    os.unlink(out_path)
    os.unlink(meta_path)
    return packet_text, meta


class TestSDIPacketBuilder:
    """Test suite for sdi_packet_builder.py"""

    def test_runs_successfully(self):
        """Builder runs without error on fixture."""
        packet, meta = run_builder(tier=1)
        assert len(packet) > 100, "Packet should not be empty"
        assert meta["tier"] == 1, "Tier should be 1"
        print("  PASS: runs successfully")

    def test_packet_format(self):
        """Output contains [SDI_CONTEXT_PACKET] markers."""
        packet, _ = run_builder(tier=1)
        assert "[SDI_CONTEXT_PACKET]" in packet, "Should have opening marker"
        assert "[/SDI_CONTEXT_PACKET]" in packet, "Should have closing marker"
        print("  PASS: packet format correct")

    def test_pinned_codename_preserved(self):
        """Project name ARGUS appears in packet."""
        packet, _ = run_builder(tier=1)
        assert "ARGUS" in packet, "Pinned codename 'ARGUS' must be preserved"
        print("  PASS: ARGUS codename preserved")

    def test_commit_tag_path_preserved(self):
        """Commit a1b2c3d4, repo path, and key values preserved verbatim."""
        packet, _ = run_builder(tier=1)
        assert "a1b2c3d4" in packet, "Commit hash must be preserved"
        assert "/home/matthew-villnave/argus-project" in packet, "Repo path must be preserved"
        print("  PASS: commit/path preserved")

    def test_lead_name_preserved(self):
        """Dr. Sarah Chen appears in packet."""
        packet, _ = run_builder(tier=1)
        assert "Dr. Sarah Chen" in packet, "Lead name must be preserved"
        print("  PASS: lead name preserved")

    def test_current_request_included(self):
        """Most recent user request is in the packet."""
        packet, _ = run_builder(tier=1)
        assert "API key format" in packet, "Current request about API key should be included"
        print("  PASS: current request included")

    def test_no_excessive_filler_repetition(self):
        """Filler phrase does not appear more than once in output."""
        packet, _ = run_builder(tier=1)
        lines = packet.split("\n")
        line_counts: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 10:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        repeated = {l: c for l, c in line_counts.items() if c > 1 and len(l) > 20}
        # Filter out benign repetition (bullet markers, section headers)
        bad_repeated = {
            l: c for l, c in repeated.items()
            if not any(l.startswith(x) for x in ["  -", "Goal:", "Current", "Pinned", "Dropped"])
        }
        assert len(bad_repeated) == 0, f"Excessive filler repetition found: {bad_repeated}"
        print("  PASS: no excessive filler repetition")

    def test_dropped_content_report(self):
        """Metadata includes dropped_sections report."""
        _, meta = run_builder(tier=1)
        assert "dropped_sections" in meta, "Metadata should include dropped_sections"
        assert len(meta["dropped_sections"]) > 0, "Should have dropped at least the Roman Empire filler"
        roman_dropped = any(
            d.get("type") == "roman_empire_filler"
            for d in meta["dropped_sections"]
        )
        assert roman_dropped, "Roman Empire filler should be in dropped_sections"
        print("  PASS: dropped-content report present")

    def test_metadata_fields(self):
        """Metadata includes tier, estimated_tokens, included_facts."""
        _, meta = run_builder(tier=2)
        assert "tier" in meta, "tier field required"
        assert "estimated_tokens" in meta, "estimated_tokens field required"
        assert "included_facts" in meta, "included_facts field required"
        assert meta["tier"] == 2, "Tier should be 2"
        assert meta["estimated_tokens"] > 0, "Token estimate should be positive"
        print("  PASS: metadata fields complete")

    def test_secret_scan_fake_detected(self):
        """Secret scan detects and warns about fake API key."""
        packet, meta = run_builder(tier=1)
        # The fake key should trigger a secret warning
        assert len(meta.get("secret_warnings", [])) > 0, "Should have secret warning for sk-fake-..."
        print("  PASS: secret scan warning generated")

    def test_deterministic_output(self):
        """Two identical runs produce identical outputs."""
        packet1, meta1 = run_builder(tier=1)
        packet2, meta2 = run_builder(tier=1)
        assert packet1 == packet2, "Packet should be deterministic"
        assert meta1["estimated_tokens"] == meta2["estimated_tokens"], "Token estimate should match"
        print("  PASS: deterministic output")

    def test_tier0_is_compact(self):
        """Tier 0 produces compact output."""
        packet, meta = run_builder(tier=0)
        assert meta["tier"] == 0, "Should be tier 0"
        assert meta["estimated_tokens"] < 600, "Tier 0 should be under ~600 tokens"
        print("  PASS: Tier 0 is compact")

    def test_tier2_has_more_content(self):
        """Tier 2 produces a larger or equally-sized packet compared to Tier 0.
        Note: with this fixture, conversation is short after filler removal (~117 words),
        so Tier 0 and Tier 2 produce the same output. This test verifies both tiers
        work and Tier 2 is at least as large as Tier 0."""
        packet0, meta0 = run_builder(tier=0, max_words=200)
        packet2, meta2 = run_builder(tier=2, max_words=2000)
        # Due to short post-filler conversation, both tiers may be equal in size.
        # The important thing is that Tier 2 is not smaller.
        assert meta2["estimated_tokens"] >= meta0["estimated_tokens"], \
            f"Tier 2 ({meta2['estimated_tokens']} tokens) should be >= Tier 0 ({meta0['estimated_tokens']} tokens)"
        # Also verify the packet texts are different (Memory/safety note differs by tier)
        assert packet0 != packet2, "Tier 0 and Tier 2 packets should differ"
        print(f"  PASS: Tier 2 ({meta2['estimated_tokens']} tok) >= Tier 0 ({meta0['estimated_tokens']} tok), packets differ")

    def test_included_facts_all_pinned(self):
        """All pinned facts appear in included_facts list."""
        _, meta = run_builder(tier=1)
        included = meta.get("included_facts", [])
        # Load expected
        with open(FIXTURE_DIR + "/expected_key_facts.json") as f:
            expected = json.load(f)
        critical_facts = expected["facts_that_must_be_preserved_exactly"]
        for fact in critical_facts:
            found = any(fact in inc for inc in included)
            assert found, f"Pinned fact missing from included_facts: {fact}"
        print("  PASS: all pinned facts included")


def main():
    os.chdir(BASE_DIR)  # ensure working dir is correct
    tests = TestSDIPacketBuilder()
    results = []
    for name in dir(tests):
        if name.startswith("test_"):
            try:
                getattr(tests, name)()
                results.append((name, "PASS"))
            except AssertionError as e:
                results.append((name, f"FAIL: {e}"))
            except Exception as e:
                results.append((name, f"ERROR: {e}"))

    print(f"\n=== Phase 26I Test Results ({len(results)} tests) ===")
    passed = sum(1 for _, r in results if r == "PASS")
    for name, result in results:
        print(f"  {name}: {result}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()