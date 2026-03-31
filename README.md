# Legal RAG Azure

Production-style legal RAG starter for Azure using FastAPI, LangGraph, Azure OpenAI, Azure AI Search, and Azure Database for PostgreSQL.

## What This Project Does

- Discovers legal PDFs from the current VS Code workspace.
- Detects ER diagram image files for schema reference and documentation.
- Extracts page-wise PDF text, chunks it, infers chunk types, embeds chunks with Azure OpenAI, and stores metadata in PostgreSQL.
- Indexes chunk documents into Azure AI Search for hybrid retrieval.
- Uses LangGraph to route questions across SQL lookup, semantic retrieval, or both.
- Returns grounded answers with chunk-level citations.

The current workspace already contains legal PDFs and ER diagrams, and the code is designed around that assumption.

## Implemented Schema

The SQLAlchemy/Alembic schema implements the ER guidance with these tables:

- `courts`
- `judges`
- `statutes`
- `statute_versions`
- `cases`
- `case_chunks`
- `case_precedents`
- `case_statute_references`

`cases` is the system of record anchor. Azure AI Search is the primary retrieval layer, while PostgreSQL holds canonical metadata and relationship tables.

## Workspace Discovery

PDF discovery order:

1. `PDF_DATA_DIR` if provided
2. `./data` if present
3. `./cases` if present
4. Recursive scan from `WORKSPACE_ROOT`

ER diagram images are discovered by scanning for image files whose names contain `er` or `diagram`.

## Setup

### 1. Create a Virtual Environment

```powershell
cd legal-rag-azure
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```powershell
Copy-Item .env.example .env
```

Fill in at least:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_BATCH_SIZE`
- `AZURE_OPENAI_MAX_RETRIES`
- `AZURE_OPENAI_RETRY_BASE_SECONDS`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_SSLMODE`
- `WORKSPACE_ROOT`
- `PDF_DATA_DIR`

Set `WORKSPACE_ROOT` to the workspace folder that contains the legal PDFs. In this repository, using `..` from `legal-rag-azure` points back to the DBMS project workspace where the PDFs and ER diagrams currently live.

For Azure Database for PostgreSQL, set `POSTGRES_SSLMODE=require`. If your environment requires an explicit CA certificate path, also set `POSTGRES_SSLROOTCERT`.

### 4. Start PostgreSQL

```powershell
docker compose up -d
```

### 5. Run Migrations

```powershell
alembic upgrade head
```

### 6. Create the Azure AI Search Index

```powershell
python scripts/create_search_index.py
```

### 7. Inspect Workspace Files

```powershell
python scripts/inspect_workspace.py
```

### 8. Ingest PDFs From the Workspace

```powershell
python scripts/ingest_all_pdfs.py
```

If you want to target a specific folder, set `PDF_DATA_DIR` or call the API with `pdf_dir`.

### 9. Run FastAPI

```powershell
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /health`
- `GET /workspace/files`
- `POST /ingest/workspace`
- `POST /query`

### Example Ingest Request

```json
{
  "pdf_dir": "../cases"
}
```

### Example Query Request

```json
{
  "question": "Summarize the reasoning in Sanjay Chandra vs CBI and cite the relevant pages."
}
```

### Query Response Shape

```json
{
  "answer": "Grounded answer here",
  "citations": [
    {
      "chunk_id": "abc123",
      "case_id": "case123",
      "title": "Sanjay Chandra vs CBI",
      "page_number": 4,
      "chunk_type": "ruling",
      "citation": "Sanjay Chandra vs CBI | page 4 | chunk abc123"
    }
  ],
  "retrieved_chunks_summary": [
    {
      "chunk_id": "abc123",
      "title": "Sanjay Chandra vs CBI",
      "page_number": 4,
      "chunk_type": "ruling",
      "score": 2.81
    }
  ],
  "route_used": "retriever"
}
```

## LangGraph Flow

The workflow includes:

- `classify_intent`
- `sql_lookup`
- `retrieve_chunks`
- `maybe_rewrite_query`
- `retrieve_chunks_retry`
- `grade_retrieved_docs`
- `generate_answer`
- `format_citations`

Routing behavior:

- Structured metadata questions route to SQL.
- Semantic similarity, summarization, dissent, or reasoning questions route to Azure AI Search retrieval.
- Mixed questions use both.

Low-result retrieval triggers query rewriting before a retry. Final answers are constrained to SQL rows and retrieved chunks.

## Notes On Legal Metadata Extraction

The first version uses practical parsing heuristics for:

- case title
- court name
- judge name
- judgment date
- case status

If a field cannot be inferred, the code stores `NULL` and logs a warning. This keeps ingestion robust without blocking the pipeline.

## Current Practical Limitations / TODOs

- Precedent graph extraction is scaffolded in the schema but not fully populated yet.
- Statute version extraction from case text is not fully automated yet.
- SQL lookup currently uses focused heuristics rather than full text-to-SQL generation.
- Retrieval grading uses a combination of search scores and lightweight LLM validation.
- Some scanned PDFs may require OCR if `pypdf` extraction is weak.

## File Highlights

- `app/ingestion/workspace_discovery.py`: workspace PDF and ER diagram discovery
- `app/ingestion/parser.py`: PDF parsing and metadata extraction
- `app/ingestion/chunker.py`: section-aware chunking and chunk type inference
- `app/retrievers/azure_search.py`: Azure AI Search schema + hybrid retrieval
- `app/retrievers/sql_lookup.py`: structured lookup tool for LangGraph
- `app/graph/workflow.py`: graph orchestration and routing
- `app/services/ingest_service.py`: end-to-end ingestion service
- `app/services/query_service.py`: end-to-end query service

## Azure Setup You Must Complete Manually

- Provision Azure OpenAI and create chat + embedding deployments.
- Provision Azure AI Search and supply endpoint + admin key.
- Provision Azure Database for PostgreSQL.
- Ensure the embedding deployment dimension matches `AZURE_SEARCH_VECTOR_DIMENSIONS`.
- Run `alembic upgrade head`.
- Run `python scripts/create_search_index.py`.

## ER Diagram Interpretation

The ER diagrams in the workspace were used as design guidance. Where the diagram image text was ambiguous, the implementation chose a production-friendly interpretation aligned with the entity list you supplied.
