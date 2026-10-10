"""Small, transparent document-level retrieval evaluation utilities."""
from __future__ import annotations

from typing import Any

from backend.llm_client import embed_texts
from backend.retrieval import retrieve_chunks


def evaluate_retrieval(
    examples: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate whether expected source documents appear in retrieved results.

    Metrics are document-level proxies. A stronger later benchmark should label
    relevant passages and separately evaluate answer correctness/faithfulness.
    """
    if not examples:
        raise ValueError("Provide at least one evaluation example.")
    questions = [str(example["question"]).strip() for example in examples]
    embeddings = embed_texts(questions)

    details: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    hits: list[float] = []

    for example, vector in zip(examples, embeddings):
        expected = {int(doc_id) for doc_id in example["relevant_document_ids"]}
        retrieved_chunks = retrieve_chunks(vector, top_k=top_k)

        # Repeated passages from one source count as one document for these metrics.
        ranked_document_ids: list[int] = []
        for chunk in retrieved_chunks:
            doc_id = int(chunk["document_id"])
            if doc_id not in ranked_document_ids:
                ranked_document_ids.append(doc_id)

        found = expected.intersection(ranked_document_ids)
        recall = len(found) / len(expected) if expected else 0.0
        reciprocal_rank = 0.0
        for rank, document_id in enumerate(ranked_document_ids, start=1):
            if document_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        hits.append(1.0 if found else 0.0)
        details.append(
            {
                "question": example["question"],
                "expected_document_ids": sorted(expected),
                "retrieved_document_ids": ranked_document_ids,
                "relevant_documents_found": sorted(found),
                "recall_at_k": round(recall, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
            }
        )

    count = len(examples)
    return {
        "examples_evaluated": count,
        "top_k": top_k,
        "hit_rate_at_k": round(sum(hits) / count, 4),
        "mean_recall_at_k": round(sum(recalls) / count, 4),
        "mean_reciprocal_rank_at_k": round(sum(reciprocal_ranks) / count, 4),
        "details": details,
        "metric_note": (
            "These metrics evaluate source-document retrieval, not whether the generated answer is correct. "
            "For multiple relevant documents, Recall@k measures the fraction found."
        ),
    }
