"""Unit tests for the Rust chunker."""

import pytest

fast_chunker = pytest.importorskip(
    "fast_chunker",
    reason="fast_chunker package not available",
)

# Skip if the Rust extension isn't exposed correctly.
if not hasattr(fast_chunker, "chunk_text"):
    pytest.skip(
        "Rust chunk_text function not available",
        allow_module_level=True,
    )


def test_chunk_text_basic():
    """Verify text is split into valid chunks."""

    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = fast_chunker.chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) >= 2
    assert chunks[0].startswith("a")
    assert chunks[1][0] in chunks[0]


def test_chunk_text_overlap_guard():
    """Verify chunk overlap values are validated."""

    with pytest.raises(ValueError):
        fast_chunker.chunk_text(
            "abc",
            chunk_size=10,
            chunk_overlap=10,
        )
