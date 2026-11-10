# Hybrid RAG + GraphRAG for Technical Manuals

A question-answering system for complex industrial manuals. It combines **dense + sparse hybrid retrieval** in Qdrant with **knowledge-graph expansion** in Neo4j, then uses a LangGraph workflow to generate and verify answers against retrieved manual evidence.

The system is hybrid by design: vector retrieval is the primary retrieval path, while graph retrieval adds entity-, relationship-, and community-based context where it is useful.

## Project overview

The project turns an authorized technical-manual PDF into searchable structured data and two complementary indexes:

- **Vector index (Qdrant):** dense semantic embeddings plus sparse BM25 retrieval, fused for broad lexical and semantic recall.
- **Knowledge graph (Neo4j):** entities, relations, steps, warnings, and chunk communities that support graph expansion.
- **Answer workflow (LangGraph):** query reformulation, intent classification, retrieval, answer generation, grounding verification, and a bounded retry.
- **Evaluation:** a gold-set batch harness that reports reliability, intent classification, retrieval ranking, grounding/safety diagnostics, and workflow metrics.

## Architecture

```text
Technical manual PDF
       |
       v
PDF text + table extraction
       |
       v
Section-aware chunking
       |
       v
Chunk enrichment (steps and warnings) + table attachment
       |
       v
Entity and relation extraction
       |
       +----------------------------+
       |                            |
       v                            v
Qdrant dense + BM25 index      Neo4j knowledge graph
       |                            |
       +-------------+--------------+
                     |
                     v
          Hybrid retrieval pipeline
      (vector search + graph expansion)
                     |
                     v
          LangGraph assistant workflow
  (reformulate -> classify -> retrieve -> answer
                -> verify -> bounded retry)
                     |
                     v
           Evidence-grounded response
```

## Technology

- Python 3.11
- Groq API for LLM extraction, classification, generation, and verification
- Qdrant for dense and sparse vector retrieval
- Neo4j for graph storage and traversal
- Sentence Transformers and FastEmbed for embeddings
- LangGraph for orchestration
- pdfplumber for PDF extraction
- Docker Compose for local infrastructure

## Repository files

The current checkout keeps Python scripts and JSON artifacts at the repository root. The following is a guide to the main files and generated data.

### Python source files

| File | Responsibility |
|---|---|
| `extract_pdf_text_and_tables.py` | Extracts text and tables from the source PDF. |
| `chunk_manual.py` | Splits extracted text into section-aware chunks. |
| `enrich_chunks.py` | Enriches chunks with procedural steps and safety warnings. |
| `attach_tables_to_chunks.py` | Associates extracted tables with their relevant chunks. |
| `extract_entities_and_relations.py` | Extracts and resolves entities and graph relationships. |
| `index_qdrant.py` | Creates the dense + sparse Qdrant collection and indexes chunks. |
| `import_neo4j.py` | Imports entities, relations, and chunk communities into Neo4j. |
| `retrieval.py` | Implements hybrid vector retrieval, graph expansion, ranking, and context assembly. |
| `run_assistant.py` | Runs the interactive conversational assistant. |
| `evaluation.py` | Runs the gold-set evaluation workflow and manages evaluation caching. |
| `evaluate_metrics.py` | Computes evaluation metrics from the evaluation cache. |

### JSON data files

| File | Contents |
|---|---|
| `chunks.json` | Section-aware chunks produced by the chunking stage. |
| `all_chunks.json` | Enriched chunks used by retrieval and indexing. |
| `tables.json` | Extracted table data. |
| `entities.json` | Resolved entity records. |
| `relations.json` | Entity relationship records. |
| `steps.json` | Extracted procedural steps. |
| `warnings.json` | Extracted safety warnings. |
| `gold_set.json` | Evaluation questions and relevance/intent labels. |
| `metrics_cache.json` | Cached extraction or processing metrics, if present. |

### Configuration and reports

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the Python application image. |
| `docker-compose.yml` | Defines Qdrant, Neo4j, bootstrap, and assistant services. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Environment-variable template. |
| `EVALUATION_REPORT.md` | Curated evaluation results for the recorded run. |

## Run locally with Docker

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + the Docker Compose plugin (Linux)
- A Groq API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Anaskhoursa/Advanced-Corporate-GraphRAG-for-a-Complex-Technical-Manual.git
cd Advanced-Corporate-GraphRAG-for-a-Complex-Technical-Manual
```

### 2. Configure secrets

```bash
cp .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

- `GROQ_API_KEY`: your Groq API key
- `NEO4J_PASSWORD`: a strong, unique password

Do not commit `.env` or share credentials. The Compose services publish their database ports on localhost only.

### 3. Start Qdrant and Neo4j

```bash
docker compose up -d qdrant neo4j
```

Wait until both services report healthy:

```bash
docker compose ps
```

Local service endpoints:

- Qdrant dashboard/API: http://localhost:6333/dashboard
- Neo4j Browser: http://localhost:7474

### 4. Build the indexes

Run the one-shot bootstrap profile after the databases are healthy:

```bash
docker compose --profile bootstrap run --rm bootstrap
```

This indexes `all_chunks.json` in Qdrant and imports `entities.json` and `relations.json` into Neo4j. The Neo4j import uses the clean-wipe option as configured by the bootstrap command, so it replaces the existing graph data.

### 5. Start the assistant

```bash
docker compose run --rm app
```

Ask questions about the technical manual in the interactive terminal. Exit with the command shown by the CLI (or Ctrl+C).

### 6. Stop the services

```bash
docker compose down
```

Persistent database data is stored in Docker volumes. To delete the stored database data as well:

```bash
docker compose down -v
```

This permanently removes the local Qdrant and Neo4j volumes.

## Run without Docker for Python

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create and configure `.env` as described above. Start Qdrant and Neo4j separately, then run:

```bash
python run_assistant.py
```

For a full local rebuild, run the extraction and indexing stages in order (the source PDF must be supplied separately):

```bash
python extract_pdf_text_and_tables.py
python chunk_manual.py
python enrich_chunks.py
python attach_tables_to_chunks.py
python extract_entities_and_relations.py
python index_qdrant.py
python import_neo4j.py --wipe
```

## Evaluation

The checked-in `EVALUATION_REPORT.md` summarizes one completed evaluation run. To run the evaluation harness again, use the evaluation script's CLI help to inspect available options:

```bash
python evaluation.py --help
```

The evaluation pipeline uses a gold set and can cache completed runs. The report includes system reliability, intent classification, retrieval ranking, grounding/safety diagnostics, and workflow rates. These measurements should be interpreted in the context of the evaluation set and verifier design.

## Manual-specific configuration

The source PDF is not included. The current JSON artifacts and parts of the processing pipeline were produced for the HL Series pump manual. In particular, `TABLE_MAP` in `attach_tables_to_chunks.py` is manual-specific and must be reviewed and adapted before using a different manual.

Only process and redistribute manuals that you are authorized to use.

## Security notes

- Keep API keys and passwords in an untracked `.env` file.
- Use a unique, strong Neo4j password.
- Do not expose Qdrant or Neo4j ports publicly without appropriate network controls and authentication.
- Treat retrieved manual content as source evidence; the LLM verifier is a grounding aid, not independent proof of factual correctness.
