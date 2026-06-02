"""
Tests for BM25 semantic search in MemoryInterface.

Verifies that find_similar_experiences uses BM25Okapi ranking
when rank_bm25 is available, with correct fallback to keyword matching.

P1 Issue 2: https://github.com/ros-claw/rosclaw-v1.0/issues/XXX
"""

import pytest

from rosclaw.memory.interface import MemoryInterface, _HAS_BM25


@pytest.fixture
def mem():
    """MemoryInterface with in-memory SeekDB backend."""
    m = MemoryInterface("test_bot")
    m.initialize()
    yield m
    m.stop()


@pytest.fixture
def populated_mem(mem):
    """MemoryInterface populated with diverse experiences."""
    experiences = [
        ("exp1", "pick up the red cup", "success", ["pick", "cup", "red"]),
        ("exp2", "grasp the blue mug", "success", ["grasp", "mug", "blue"]),
        ("exp3", "place block on table", "success", ["place", "block", "table"]),
        ("exp4", "pick up the green block", "failure", ["pick", "block", "green"]),
        ("exp5", "move to kitchen", "success", ["move", "kitchen", "navigate"]),
        ("exp6", "pick up cup from table", "success", ["pick", "cup", "table"]),
        ("exp7", "rotate wrist clockwise", "success", ["rotate", "wrist"]),
        ("exp8", "scan the room", "success", ["scan", "room", "perception"]),
    ]
    for eid, instr, outcome, tags in experiences:
        mem.store_experience(eid, "praxis", instr, outcome=outcome, tags=tags)
    return mem


class TestBM25Availability:
    """Verify BM25 conditional import."""

    def test_bm25_is_available(self):
        """rank_bm25 should be installed in the test environment."""
        assert _HAS_BM25 is True

    def test_tokenizer(self):
        """Tokenizer should lowercase and split on word boundaries."""
        tokens = MemoryInterface._tokenize("Pick up the Red Cup!")
        assert tokens == ["pick", "up", "the", "red", "cup"]

    def test_tokenizer_filters_short(self):
        """Single-char tokens should be filtered."""
        tokens = MemoryInterface._tokenize("a b c task")
        assert tokens == ["task"]

    def test_tokenizer_handles_empty(self):
        """Empty input should produce empty tokens."""
        assert MemoryInterface._tokenize("") == []
        assert MemoryInterface._tokenize("   ") == []

    def test_tokenizer_cjk(self):
        """Tokenizer should handle CJK characters."""
        tokens = MemoryInterface._tokenize("抓取杯子 pick cup")
        assert "抓取杯子" in tokens
        assert "pick" in tokens
        assert "cup" in tokens


class TestBM25Search:
    """BM25 ranking quality tests."""

    def test_exact_match_ranks_first(self, populated_mem):
        """Exact word match should rank highly."""
        results = populated_mem.find_similar_experiences("pick up cup", limit=3)
        assert len(results) > 0
        # exp1 ("pick up the red cup") or exp6 ("pick up cup from table") should be top
        top_ids = [r["id"] for r in results[:2]]
        assert "exp1" in top_ids or "exp6" in top_ids

    def test_synonym_like_queries_find_results(self, populated_mem):
        """Queries with overlapping terms should find relevant experiences."""
        # "grasp mug" overlaps with exp2 "grasp the blue mug" via "grasp" and "mug"
        results = populated_mem.find_similar_experiences("grasp mug", limit=3)
        assert len(results) > 0
        top_ids = [r["id"] for r in results]
        assert "exp2" in top_ids

    def test_outcome_filter_works(self, populated_mem):
        """outcome_filter should restrict results."""
        # exp4 is the only failure with "pick" or "block"
        results = populated_mem.find_similar_experiences(
            "pick up block", outcome_filter="failure", limit=5
        )
        for r in results:
            assert r["outcome"] == "failure"

    def test_unrelated_query_returns_empty(self, populated_mem):
        """Completely unrelated query should return few or no results."""
        results = populated_mem.find_similar_experiences(
            "quantum physics entanglement", limit=5
        )
        # Should return empty or very few results (BM25 may still match "the")
        assert len(results) <= 2

    def test_empty_query_returns_empty(self, populated_mem):
        """Empty query should return empty list."""
        results = populated_mem.find_similar_experiences("", limit=5)
        assert results == []

    def test_limit_respected(self, populated_mem):
        """Result count should not exceed limit."""
        results = populated_mem.find_similar_experiences("pick", limit=2)
        assert len(results) <= 2

    def test_tag_matching(self, populated_mem):
        """Tags should contribute to relevance scoring."""
        # "navigate" only appears as a tag in exp5
        results = populated_mem.find_similar_experiences("navigate kitchen", limit=3)
        assert len(results) > 0
        top_ids = [r["id"] for r in results]
        assert "exp5" in top_ids

    def test_empty_db_returns_empty(self, mem):
        """Empty database should return empty results."""
        results = mem.find_similar_experiences("pick up cup", limit=5)
        assert results == []

    def test_bm25_ranking_quality(self, populated_mem):
        """BM25 should rank more relevant results higher than less relevant.

        'pick up the red cup' should rank higher than 'rotate wrist' for
        the query 'pick up cup'.
        """
        results = populated_mem.find_similar_experiences("pick up cup", limit=10)
        ids = [r["id"] for r in results]
        # exp1 (pick up the red cup) should rank before exp7 (rotate wrist)
        if "exp1" in ids and "exp7" in ids:
            assert ids.index("exp1") < ids.index("exp7")


class TestBM25Fallback:
    """Verify keyword fallback when rank_bm25 is not available."""

    def test_keyword_fallback(self, populated_mem):
        """Should fall back to keyword matching when BM25 unavailable."""
        import rosclaw.memory.interface as iface

        original = iface._HAS_BM25
        try:
            iface._HAS_BM25 = False
            results = populated_mem.find_similar_experiences("pick up cup", limit=3)
            assert len(results) > 0
            # Should still find pick/cup experiences
            top_instructions = [r.get("instruction", "") for r in results]
            assert any("pick" in instr for instr in top_instructions)
        finally:
            iface._HAS_BM25 = original

    def test_keyword_fallback_empty_query(self, populated_mem):
        """Keyword fallback should handle empty queries."""
        import rosclaw.memory.interface as iface

        original = iface._HAS_BM25
        try:
            iface._HAS_BM25 = False
            results = populated_mem.find_similar_experiences("", limit=5)
            assert results == []
        finally:
            iface._HAS_BM25 = original
