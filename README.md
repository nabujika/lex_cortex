# Lex Cortex

`Lex Cortex` is a Legal Tech Retrieval-Augmented Generation (RAG) backend for **The Legal Brain**.

It uses:

- **FastAPI** for the API layer
- **LangGraph** for orchestration
- **Azure OpenAI** for chat + embeddings
- **PostgreSQL + pgvector** for relational + vector retrieval
- **Hybrid RAG** with semantic search followed by SQL enrichment

## Architecture

The system follows this flow:

1. Legal PDFs are ingested.
2. Judgment PDFs are parsed, chunked, embedded, and stored in PostgreSQL.
3. Statute PDFs are stored as versioned statute records.
4. A LangGraph workflow handles query analysis, vector retrieval, SQL joins, and answer generation.
5. The API returns:
   - grounded answer
   - extracted filters
   - cases used for the answer

### LangGraph pipeline

- `analyze_query`
- `vector_search`
- `sql_enrich`
- `merge_results`
- `generate_answer`

## Data Model

Core tables implemented:

- `court`
- `judge`
- `"case"`
- `statute`
- `case_chunk`
- `statute_version`
- `case_precedent`
- `case_statute`
- `embedding_store`

Notes:

- Embeddings are stored against `case_chunk`, as required by the ER model.
- Statutes are stored in `statute` + `statute_version` and used during SQL enrichment.
- `case_precedent` support is included for precedent traversal.

## Project Structure

```text
app.py
db.py
ingest.py
rag_graph.py
requirements.txt
.env.example
README.md
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your Azure credentials.

Important settings:

- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DIMENSIONS`
- `POSTGRES_HOST`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`

Important:

- `AZURE_OPENAI_EMBEDDING_DIMENSIONS` must match the vector dimension used in the database schema.
- For `text-embedding-3-large`, this project explicitly requests the configured dimension size from Azure.

## Azure Prerequisites

Before running the app:

1. Create an Azure OpenAI resource.
2. Deploy:
   - one chat model, such as `gpt-4o-mini`
   - one embedding model, such as `text-embedding-3-large`
3. Create an Azure Database for PostgreSQL Flexible Server.
4. In PostgreSQL server parameters, allowlist the `vector` extension by adding it to `azure.extensions`.

If `vector` is not allowlisted, `/admin/init-db` will fail.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## Initialize the Database

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/admin/init-db
```

You can also inspect the generated schema:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/schema
```

## Ingest Legal PDFs

Place judgment PDFs and statute PDFs in a directory, then call:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/admin/ingest `
  -ContentType "application/json" `
  -Body '{"directory":"c:\\Users\\Lenovo\\Desktop\\dbs"}'
```

Ingestion behavior:

- statute PDFs are loaded first
- case PDFs are chunked at roughly `450` tokens
- overlap is `50` tokens
- embeddings are generated via Azure OpenAI
- chunks go to `case_chunk`
- vectors go to `embedding_store`

## Query the RAG System

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"query":"What has the court said about bail in economic offences?","top_k":8}'
```

Response shape:

```json
{
  "answer": "Grounded legal answer...",
  "cases_used": [
    {
      "title": "Sanjay Chandra vs Cbi",
      "judge": "Unknown Judge",
      "court": "Unknown Court"
    }
  ],
  "filters": {
    "judge": null,
    "court": null,
    "statute": null,
    "date_from": null,
    "date_to": null,
    "as_of_date": null,
    "needs_precedents": true,
    "answer_style": null
  }
}
```

## API Endpoints

- `GET /health` - service healthcheck
- `GET /schema` - schema metadata and SQL
- `POST /admin/init-db` - initialize PostgreSQL schema
- `POST /admin/ingest` - ingest legal PDFs
- `POST /query` - run the hybrid RAG pipeline

## Notes on Repository Contents

This repo intentionally excludes:

- `.env`
- `.venv`
- server logs
- local PDF corpus

Those are ignored via `.gitignore` to avoid leaking secrets or uploading source documents unintentionally.

## Deployment

For Azure App Service, a typical startup command is:

```text
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

Set the same environment variables from `.env` in App Service configuration.

## Current Status

The backend currently supports:

- end-to-end ingestion
- pgvector similarity search
- SQL enrichment across `case`, `judge`, `court`, and `statute`
- answer generation with case citations
- precedent lookup support
- temporal statute filtering support

## Next Improvements

Good next steps for this repo:

- improve court and judge metadata extraction from PDF headers
- populate `case_precedent` automatically from cited-case parsing
- strengthen statute linking beyond title matching
- add tests and migration tooling
