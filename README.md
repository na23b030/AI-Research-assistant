# RAG-Powered AI Document Intelligence and Research Assistant

A document question-answering application that uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from uploaded files and generate answers with source references. The project includes a FastAPI backend, a Streamlit interface, and a local SQLite database.

## Features

- Upload and index PDF, TXT, and Markdown files.
- Extract text and split it into overlapping chunks.
- Create text embeddings with the OpenAI Embeddings API.
- Retrieve relevant passages using cosine similarity and configurable top-k search.
- Generate answers from retrieved context using the OpenAI Responses API.
- Display source passages, citation labels, similarity scores, token usage, and response latency.
- Store document metadata, chunks, and embeddings in SQLite.
- Detect duplicate document content and remove indexed documents.
- Evaluate retrieval using Hit Rate@k, Recall@k, and Mean Reciprocal Rank (MRR).
- Expose REST endpoints with request validation and interactive API documentation.

## Technology Stack

- **Language:** Python
- **Backend/API:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit
- **LLM and embeddings:** OpenAI API
- **Data storage:** SQLite
- **Document processing:** pypdf
- **Numerical operations:** NumPy
- **Testing:** pytest

## Architecture

```text
PDF / TXT / Markdown
        |
        v
Text extraction and normalization
        |
        v
Overlapping text chunking
        |
        v
OpenAI embedding model
        |
        v
SQLite: document metadata, chunks, embeddings

User question
        |
        v
Question embedding
        |
        v
Cosine-similarity search and top-k retrieval
        |
        v
Retrieved passages + question + prompt instructions
        |
        v
OpenAI Responses API
        |
        v
Answer with source labels and inspectable evidence
```

The backend handles document ingestion, retrieval, answer generation, persistence, and evaluation. The Streamlit frontend communicates with the backend through HTTP requests and JSON responses.

## Project Structure

```text
grounded_ai_research_assistant/
├── backend/
│   ├── config.py             # Environment and application configuration
│   ├── database.py           # SQLite persistence
│   ├── evaluation.py         # Retrieval evaluation metrics
│   ├── llm_client.py         # OpenAI embeddings and generation calls
│   ├── main.py               # FastAPI routes and request handling
│   ├── rag.py                # RAG pipeline and context construction
│   ├── retrieval.py          # Cosine-similarity retrieval
│   ├── schemas.py             # API request schemas and validation
│   └── text_processing.py    # File extraction and text chunking
├── frontend/
│   └── app.py                # Streamlit user interface
├── sample_docs/
│   └── ai_systems_notes.txt  # Sample document for testing
├── tests/
│   ├── test_evaluation.py
│   └── test_text_processing.py
├── .env.example              # Environment variable template
├── .gitignore
├── pytest.ini
├── requirements.txt
├── run_backend.bat
├── run_frontend.bat
└── README.md
```

## Requirements

- Python 3.10 or newer
- An OpenAI API key with API access and available quota
- Internet access for embedding and answer-generation requests

OpenAI API usage may incur charges. API billing is separate from a ChatGPT subscription. Review current API pricing before processing large documents or running repeated evaluations.

## Setup on Windows

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set the API key:

```text
OPENAI_API_KEY=your_api_key_here
```

The default models are configured as follows:

```text
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Model names can be changed in `.env` to models available to the configured API account. Never commit `.env` or expose the API key in source code, screenshots, or public repositories. The `.gitignore` file excludes `.env`, virtual environments, and local database files.

## Run the Application

### 1. Start the backend

In the first terminal, with the virtual environment activated:

```powershell
uvicorn backend.main:app --reload
```

The backend runs at `http://127.0.0.1:8000` by default.

- Health endpoint: `http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

The SQLite database is initialized automatically and stored at `data/research_assistant.sqlite3` by default.

### 2. Start the frontend

Open a second terminal in the project folder and activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

### 3. Try the application

1. Upload `sample_docs/ai_systems_notes.txt` from the project folder.
2. Ask: `What is RAG, and what are its limitations?`
3. Expand the retrieved-evidence section to inspect the passages and similarity scores.
4. In the retrieval evaluation section, create labelled question/document examples and run the metrics.

## REST API

The API uses standard HTTP methods and returns JSON for its application endpoints. FastAPI provides interactive request and response schemas at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns application and model configuration status |
| `GET` | `/api/documents` | Lists indexed documents |
| `POST` | `/api/documents/upload` | Uploads, processes, embeds, and stores documents using multipart form data |
| `DELETE` | `/api/documents/{document_id}` | Deletes a document and its chunks |
| `POST` | `/api/chat` | Retrieves evidence and generates an answer |
| `POST` | `/api/evaluate/retrieval` | Measures source-document retrieval against labelled examples |

Example request body for `POST /api/chat`:

```json
{
  "question": "What are the limitations of RAG?",
  "top_k": 5,
  "history": []
}
```

Common response status codes include `200` for success, `400` for a request that cannot be processed, `404` for a missing document, `413` for an upload limit violation, `422` for request validation errors, `502` for a model-provider request failure, and `503` when the API key is not configured.

## Retrieval Evaluation

The evaluation endpoint accepts questions and the IDs of documents labelled as relevant. It embeds each question, retrieves the top-k passages, and checks whether the expected source documents appear in the results.

- **Hit Rate@k:** proportion of questions for which at least one relevant document was retrieved.
- **Recall@k:** proportion of labelled relevant documents retrieved for each question, averaged over the evaluation set.
- **MRR:** average reciprocal rank of the first relevant retrieved document.

These metrics evaluate source-document retrieval. They do not independently establish that an answer generated by the LLM is correct or fully supported by the cited evidence.

## Tests

Run the tests from the project root with the virtual environment activated:

```powershell
python -m pytest -q
```

The unit tests cover text chunking and retrieval metric calculations without making OpenAI API calls.

## Scope and Limitations

- Retrieval uses an exact cosine-similarity scan over the local SQLite data. This is suitable for a small dataset but is not designed for large-scale vector search.
- PDF extraction requires selectable text; OCR for scanned or image-only PDFs is not included.
- Retrieval evaluation operates at the document level. Passage-level relevance and generated-answer evaluation are not implemented in this version.
- The application is intended for local demonstration. Authentication, multi-user isolation, production-grade rate limiting, and managed storage are not included.
- Retrieved passages are treated as untrusted input in the generation prompt, but prompt instructions alone do not eliminate prompt-injection risk.
- Source passages should be checked when validating an answer; generated text and citation labels can still be incorrect.

## Potential Extensions

Possible next steps include hybrid BM25 and vector retrieval, a reranker, passage-level and answer-level evaluation, query rewriting, controlled tool-using research workflows, tracing, and a dedicated vector database such as pgvector or Qdrant.
