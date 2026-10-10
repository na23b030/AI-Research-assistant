from backend.evaluation import evaluate_retrieval


def test_retrieval_evaluation_metrics_with_stubbed_embeddings_and_retrieval(monkeypatch):
    import backend.evaluation as evaluation_module

    monkeypatch.setattr(evaluation_module, "embed_texts", lambda questions: [[1.0], [2.0]])

    def fake_retrieve(vector, top_k):
        if vector == [1.0]:
            return [
                {"document_id": 10},
                {"document_id": 10},
                {"document_id": 20},
            ][:top_k]
        return [{"document_id": 30}, {"document_id": 40}][:top_k]

    monkeypatch.setattr(evaluation_module, "retrieve_chunks", fake_retrieve)
    result = evaluate_retrieval(
        [
            {"question": "Q1", "relevant_document_ids": [10]},
            {"question": "Q2", "relevant_document_ids": [99]},
        ],
        top_k=5,
    )

    assert result["examples_evaluated"] == 2
    assert result["hit_rate_at_k"] == 0.5
    assert result["mean_recall_at_k"] == 0.5
    assert result["mean_reciprocal_rank_at_k"] == 0.5


def test_retrieval_evaluation_rejects_empty_dataset():
    try:
        evaluate_retrieval([], top_k=5)
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected empty evaluation set to raise ValueError")
