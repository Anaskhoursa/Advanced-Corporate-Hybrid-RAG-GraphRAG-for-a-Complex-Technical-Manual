# Hybrid RAG + GraphRAG for Technical Manuals

A technical-manual question-answering system combining dense and sparse (BM25) retrieval in Qdrant with knowledge-graph expansion in Neo4j. A LangGraph workflow handles query reformulation, intent classification, retrieval, answer generation, grounding verification, and bounded retry.

## Repository layout

- `scripts/`: all Python source code.
- `data/`: JSON inputs, extracted artifacts, graph records, and evaluation data.
- Root: Docker configuration, dependency list, environment template, and documentation.

## Source files

| File | Purpose |
|---|---|
| `scripts/extract_pdf_text_and_tables.py` | Extract PDF text and tables. |
| `scripts/chunk_manual.py` | Create section-aware chunks. |
| `scripts/enrich_chunks.py` | Extract procedural steps and warnings from chunks. |
| `scripts/attach_tables_to_chunks.py` | Attach manually mapped tables to chunks. |
| `scripts/extract_entities_and_relations.py` | Extract entities, relations, steps, and warnings. |
| `scripts/index_qdrant.py` | Build the dense + sparse Qdrant index. |
| `scripts/import_neo4j.py` | Load graph entities and relations into Neo4j. |
| `scripts/retrieval.py` | Hybrid vector retrieval and graph expansion. |
| `scripts/run_assistant.py` | Interactive LangGraph assistant. |
| `scripts/evaluation.py` | Run the gold-set evaluation. |
| `scripts/evaluate_metrics.py` | Calculate evaluation metrics from cached runs. |

## Data files

| File | Contents |
|---|---|
| `data/chunks.json` | Section-aware chunks. |
| `data/all_chunks.json` | Enriched chunks used for indexing and retrieval. |
| `data/tables.json` | Extracted PDF tables. |
| `data/entities.json` | Extracted/resolved entities. |
| `data/relations.json` | Graph relationships. |
| `data/steps.json` | Procedural steps. |
| `data/warnings.json` | Safety warnings. |
| `data/gold_set.json` | Evaluation questions and labels. |
| `data/metrics_cache.json` | Cached processing metrics. |

## Run with Docker Compose

Requirements: Docker Engine/Desktop with Compose, Git, and a Groq API key.

```bash
git clone https://github.com/Anaskhoursa/Advanced-Corporate-GraphRAG-for-a-Complex-Technical-Manual.git
cd Advanced-Corporate-GraphRAG-for-a-Complex-Technical-Manual
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`, `NEO4J_USER`, and a strong `NEO4J_PASSWORD`.

Start databases, build indexes, and launch the assistant:

```bash
docker compose up -d qdrant neo4j
docker compose ps
docker compose --profile bootstrap run --rm bootstrap
docker compose run --rm app
```

Qdrant dashboard: http://localhost:6333/dashboard  
Neo4j Browser: http://localhost:7474

Stop services with `docker compose down`. Remove persisted database volumes only when intended: `docker compose down -v`.

The Docker image copies `scripts/` and `data/` into the same named directories under `/app`. Compose invokes Python entry points from `scripts/`; Python scripts resolve JSON artifacts under `data/`.

## Run locally without Docker

Use Python 3.11, create a virtual environment, and install dependencies:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
python scripts/run_assistant.py
```

Run Qdrant and Neo4j separately and configure their connection variables in `.env`.

## Rebuild data and indexes

The original source PDF is not included; provide it at `manuals/hl_series_iom.pdf` before extraction. Run pipeline steps from the repository root:

```bash
python scripts/extract_pdf_text_and_tables.py
python scripts/chunk_manual.py
python scripts/enrich_chunks.py
python scripts/attach_tables_to_chunks.py
python scripts/extract_entities_and_relations.py
python scripts/index_qdrant.py
python scripts/import_neo4j.py --wipe
```

The extraction pipeline writes generated JSON artifacts to `data/`. The table mapping in `scripts/attach_tables_to_chunks.py` is specific to the HL Series pump manual and must be adapted for another manual.

## Evaluation

See `EVALUATION_REPORT.md` for the recorded evaluation summary. Run `python scripts/evaluation.py --help` for evaluation options. Evaluation outputs/caches are written under `data/` unless an explicit output directory is supplied.

## Security and scope

Keep credentials in the untracked `.env` file. Database ports are bound to localhost by Compose. Only process manuals you are authorized to use. A verifier pass is a grounding diagnostic, not proof that an answer is correct.
