# Focus on format_docs and load_llm from app.py,
# mocking heavy dependencies so system does not hit real Chroma or Gemini.

import app


def test_format_docs_app(sample_docs):
    text = app.format_docs(sample_docs)
    assert "Python lists are mutable" in text
    assert "Tuples are immutable" in text


def test_load_llm_builds_chain(
    monkeypatch, fake_chroma_class, fake_embeddings, fake_llm_class
):
    # Monkeypatch heavy deps
    monkeypatch.setattr(
        app, "HuggingFaceEmbeddings", lambda model_name: fake_embeddings
    )
    monkeypatch.setattr(app, "Chroma", fake_chroma_class)
    monkeypatch.setattr(app, "ChatGoogleGenerativeAI", fake_llm_class)

    chain = app.load_llm()

    # The returned object should be callable via .invoke in the real app
    assert chain is not None

    # Just assert that it returns *something* string-like
    result = chain.invoke("What is a tuple?")
    assert isinstance(result, str)
    assert result  # non-empty
