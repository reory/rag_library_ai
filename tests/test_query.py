# Test the CLI-style query pipeline by mocking Chroma, embeddings, and LLM

# tests/test_query.py
import query


def test_format_docs(sample_docs):
    text = query.format_docs(sample_docs)
    assert "Python lists are mutable" in text
    assert "Tuples are immutable" in text
    assert "\n\n" in text


def test_ask_the_books(
    monkeypatch, fake_chroma_class, fake_embeddings, fake_llm_class, capsys
):
    # Monkeypatch dependencies
    monkeypatch.setattr(
        query, "HuggingFaceEmbeddings", lambda model_name: fake_embeddings
    )
    monkeypatch.setattr(query, "Chroma", fake_chroma_class)
    monkeypatch.setattr(query, "ChatGoogleGenerativeAI", fake_llm_class)

    query.ask_the_books("What is a list in Python?")

    captured = capsys.readouterr()
    assert "🔎 Searching for: What is a list in Python?" in captured.out
    assert "🤖 AI RESPONSE" in captured.out
    assert "FAKE_LLM_RESPONSE" in captured.out
