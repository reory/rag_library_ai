# Mock out PyPDFLoader, fast_chunker.chunk_text,
# HuggingFaceEmbeddings, and Chroma so ingestion is fast and deterministic.

# tests/test_ingest.py
import types

import ingest


def test_run_ingestion_happy_path(
    monkeypatch, tmp_path, fake_chroma_class, fake_embeddings
):
    # Create fake DATA_PATH with a dummy PDF filename
    data_dir = tmp_path / "data" / "books"
    data_dir.mkdir(parents=True)
    (data_dir / "dummy.pdf").write_text("PDF CONTENT IGNORED")

    monkeypatch.setattr(ingest, "DATA_PATH", str(data_dir))
    monkeypatch.setattr(ingest, "DB_PATH", str(tmp_path / "vectorstore" / "db"))

    # Fake PyPDFLoader to return simple docs
    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            Doc = types.SimpleNamespace
            return [
                Doc(page_content="Some Python content", metadata={"source": self.path})
            ]

    monkeypatch.setattr(ingest, "PyPDFLoader", FakeLoader)

    # Fake fast_chunker
    class FakeFastChunker:
        @staticmethod
        def chunk_text(text, chunk_size, chunk_overlap):
            return [text[:10], text[10:20]]

    monkeypatch.setattr(ingest, "fast_chunker", FakeFastChunker)

    # Fake embeddings + Chroma
    monkeypatch.setattr(
        ingest, "HuggingFaceEmbeddings", lambda model_name: fake_embeddings
    )
    monkeypatch.setattr(ingest, "Chroma", fake_chroma_class)

    # Run
    ingest.run_ingestion()

    # Verify DB path created
    assert (tmp_path / "vectorstore" / "db").exists()


def test_run_ingestion_missing_data_folder(monkeypatch, tmp_path, capsys):
    # Point DATA_PATH to a non-existent folder
    missing_dir = tmp_path / "nope"
    monkeypatch.setattr(ingest, "DATA_PATH", str(missing_dir))

    ingest.run_ingestion()
    captured = capsys.readouterr()
    assert "Error: Please create" in captured.out
