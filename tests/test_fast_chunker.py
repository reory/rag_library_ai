# Unit tests for the Rust chunker, with a skip if the module isn’t available.


import fast_chunker
import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("fast_chunker", reason="fast_chunker wheel not available")
    is None,
    reason="fast_chunker not importable",
)


def test_chunk_text_basic():
    import fast_chunker

    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = fast_chunker.chunk_text(text, chunk_size=10, chunk_overlap=2)

    # Should produce overlapping chunks
    assert len(chunks) >= 2
    assert chunks[0].startswith("a")
    assert chunks[1][0] in chunks[0]  # overlap


def test_chunk_text_overlap_guard():

    with pytest.raises(Exception):
        fast_chunker.chunk_text("abc", chunk_size=10, chunk_overlap=10)
