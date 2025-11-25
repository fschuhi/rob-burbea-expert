"""
Citation utilities for parsing and formatting references in RAG responses.

This module provides tools for:
- Extracting citation numbers from generated text
- Formatting citations for display (e.g., bold styling for Streamlit)
"""

import re


def resolve_references(text: str) -> list[int]:
    """
    Robustly parses citation tags like [1], [1, 2], [1-3] from text.

    Supports:
    - Single citations: [1]
    - Comma-separated: [1, 2, 3]
    - Ranges: [1-3] expands to [1, 2, 3]
    - Mixed: [1, 3-5, 7] expands to [1, 3, 4, 5, 7]

    Args:
        text: Generated text that may contain citation markers

    Returns:
        Sorted list of unique citation IDs (integers)

    Examples:
        >>> resolve_references("The practice [1] involves energy body [2].")
        [1, 2]
        >>> resolve_references("References [1-3] discuss this.")
        [1, 2, 3]
        >>> resolve_references("See [1, 3-5, 7] for details.")
        [1, 3, 4, 5, 7]
    """
    raw_matches = re.findall(r"\[([\d,\s\-]+)\]", text)
    unique_ids = set()

    for match in raw_matches:
        parts = match.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Handle range notation [N-M]
                try:
                    start, end = map(int, part.split("-"))
                    # Safety: Don't expand huge ranges (likely parsing error)
                    if end - start < 20:
                        unique_ids.update(range(start, end + 1))
                except ValueError:
                    continue
            else:
                # Handle single number
                try:
                    unique_ids.add(int(part))
                except ValueError:
                    continue

    return sorted(list(unique_ids))


def ensure_bold_citations(text: str) -> str:
    """
    Wraps citation tags [N] in bold markers if not already bold.

    This is useful for Streamlit markdown rendering where **[1]** displays
    as bold gold text (via CSS), but [1] displays as plain text.

    Args:
        text: Text containing citation markers

    Returns:
        Text with citations wrapped in ** markers

    Examples:
        >>> ensure_bold_citations("The practice [1] is described.")
        'The practice **[1]** is described.'
        >>> ensure_bold_citations("Already **[1]** bold.")
        'Already **[1]** bold.'
    """
    # Replace [N] with **[N]** unless already wrapped
    # Negative lookbehind (?<!\*\*) ensures we don't double-wrap
    # Negative lookahead (?!\*\*) ensures we don't wrap if already bold
    return re.sub(r"(?<!\*\*)\[(\d+)\](?!\*\*)", r"**[\1]**", text)
