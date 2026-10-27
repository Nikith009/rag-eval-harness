from rageval.chunking import (
    chunk_documents,
    fixed_chunks,
    recursive_chunks,
    sentence_chunks,
    word_count,
)
from rageval.config import ChunkerConfig
from rageval.documents import Document


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_fixed_chunks_overlap():
    chunks = fixed_chunks(words(25), size=10, overlap=3)
    assert [c.split()[0] for c in chunks] == ["w0", "w7", "w14", "w21"]
    # consecutive chunks share exactly `overlap` words
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "w24"


def test_fixed_chunks_short_text_is_single_chunk():
    assert fixed_chunks("just a few words", size=10, overlap=2) == ["just a few words"]
    assert fixed_chunks("   ", size=10, overlap=2) == []


def test_sentence_chunks_windows():
    text = "One. Two! Three? Four. Five."
    assert sentence_chunks(text, n=2, overlap=0) == ["One. Two!", "Three? Four.", "Five."]
    assert sentence_chunks(text, n=3, overlap=1) == ["One. Two! Three?", "Three? Four. Five."]


def test_recursive_chunks_respect_size_and_prefer_paragraphs():
    para_a = "Alpha sentence one. Alpha sentence two."
    para_b = "Beta " + words(40) + "."
    text = f"{para_a}\n\n{para_b}"
    chunks = recursive_chunks(text, size=20, overlap=0)

    assert chunks[0].startswith("Alpha sentence one.")
    assert all(word_count(c) <= 20 for c in chunks)
    # nothing is lost
    assert sum(word_count(c) for c in chunks) == word_count(text)


def test_recursive_chunks_overlap_repeats_tail():
    sentences = " ".join(f"Sentence number {i} is here." for i in range(12))
    chunks = recursive_chunks(sentences, size=15, overlap=5)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.split(". ")[-1]
        assert b.startswith(last_sentence.rstrip("."))


def test_chunk_documents_assigns_ids_and_positions():
    docs = [Document("a", words(30)), Document("b", words(15))]
    cfg = ChunkerConfig(name="f10", strategy="fixed", size=10, overlap=0)
    chunks = chunk_documents(docs, cfg)

    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert [c.doc_id for c in chunks] == ["a", "a", "a", "b", "b"]
    assert [c.position for c in chunks] == [0, 1, 2, 0, 1]
