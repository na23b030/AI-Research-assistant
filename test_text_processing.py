from backend.text_processing import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", chunk_size_words=5, overlap_words=1) == []


def test_short_text_returns_one_chunk():
    assert chunk_text("one two three", chunk_size_words=5, overlap_words=1) == ["one two three"]


def test_chunks_overlap_by_requested_word_count():
    words = "one two three four five six seven eight nine" 
    chunks = chunk_text(words, chunk_size_words=4, overlap_words=1)
    assert chunks == ["one two three four", "four five six seven", "seven eight nine"]


def test_chunk_parameters_are_validated():
    try:
        chunk_text("some text", chunk_size_words=3, overlap_words=3)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
