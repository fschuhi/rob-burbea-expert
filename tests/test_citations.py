"""
Tests for citation utilities.
"""

import pytest
from src.citations import resolve_references, ensure_bold_citations


class TestResolveReferences:
    """Tests for parsing citation tags from text."""

    def test_single_citation(self):
        """Single citation [1] should return [1]."""
        text = "The practice [1] is described."
        assert resolve_references(text) == [1]

    def test_multiple_citations(self):
        """Multiple citations [1], [2] should return [1, 2]."""
        text = "See [1] and [2] for details."
        assert resolve_references(text) == [1, 2]

    def test_comma_separated(self):
        """Comma-separated [1, 2, 3] should return [1, 2, 3]."""
        text = "References [1, 2, 3] discuss this."
        assert resolve_references(text) == [1, 2, 3]

    def test_range_notation(self):
        """Range [1-3] should expand to [1, 2, 3]."""
        text = "See [1-3] for comprehensive coverage."
        assert resolve_references(text) == [1, 2, 3]

    def test_mixed_notation(self):
        """Mixed [1, 3-5, 7] should expand correctly."""
        text = "References [1, 3-5, 7] are relevant."
        assert resolve_references(text) == [1, 3, 4, 5, 7]

    def test_duplicates_removed(self):
        """Duplicate references should be deduplicated."""
        text = "See [1] and also [1] again."
        assert resolve_references(text) == [1]

    def test_unsorted_input(self):
        """Citations should be returned in sorted order."""
        text = "References [5, 2, 8, 1]."
        assert resolve_references(text) == [1, 2, 5, 8]

    def test_no_citations(self):
        """Text without citations should return empty list."""
        text = "This text has no citations."
        assert resolve_references(text) == []

    def test_invalid_ranges_ignored(self):
        """Huge ranges (likely errors) should be ignored."""
        text = "Invalid [1-100] range."
        assert resolve_references(text) == []

    def test_malformed_citations_ignored(self):
        """Malformed citations should not crash."""
        text = "Bad [abc], [1-], [-3], [1--2] citations but valid [5]."
        assert resolve_references(text) == [5]

    def test_whitespace_handled(self):
        """Whitespace in citations should be handled."""
        text = "Citations [ 1 , 2 , 3 ] with spaces."
        assert resolve_references(text) == [1, 2, 3]

    def test_multiple_brackets(self):
        """Multiple separate bracket groups should all be parsed."""
        text = "First [1] and second [2] and third [3-4]."
        assert resolve_references(text) == [1, 2, 3, 4]


class TestEnsureBoldCitations:
    """Tests for bold citation formatting."""

    def test_wraps_plain_citation(self):
        """Plain [1] should be wrapped."""
        text = "The practice [1] is described."
        expected = "The practice **[1]** is described."
        assert ensure_bold_citations(text) == expected

    def test_multiple_citations(self):
        """Multiple plain citations should all be wrapped."""
        text = "See [1] and [2] for details."
        expected = "See **[1]** and **[2]** for details."
        assert ensure_bold_citations(text) == expected

    def test_preserves_existing_bold(self):
        """Already bold **[1]** should not be double-wrapped."""
        text = "Already **[1]** bold."
        expected = "Already **[1]** bold."
        assert ensure_bold_citations(text) == expected

    def test_mixed_bold_and_plain(self):
        """Mix of bold and plain should only wrap plain."""
        text = "Plain [1] and bold **[2]** citations."
        expected = "Plain **[1]** and bold **[2]** citations."
        assert ensure_bold_citations(text) == expected

    def test_no_citations(self):
        """Text without citations should be unchanged."""
        text = "This text has no citations."
        assert ensure_bold_citations(text) == text

    def test_multidigit_citations(self):
        """Multi-digit citations like [12] should work."""
        text = "Reference [12] is important."
        expected = "Reference **[12]** is important."
        assert ensure_bold_citations(text) == expected

    def test_adjacent_citations(self):
        """Adjacent citations [1][2] should both be wrapped."""
        text = "Multiple [1][2][3] in a row."
        expected = "Multiple **[1]****[2]****[3]** in a row."
        assert ensure_bold_citations(text) == expected

    def test_citation_at_boundaries(self):
        """Citations at start/end of string should work."""
        text = "[1] Start and end [2]"
        expected = "**[1]** Start and end **[2]**"
        assert ensure_bold_citations(text) == expected
