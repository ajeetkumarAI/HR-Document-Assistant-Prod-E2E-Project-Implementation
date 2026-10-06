# 02: Architecture Walkthrough (Whiteboard Script)

Answers to: *"Explain your architecture end to end"*, *"Sketch a production RAG chatbot"*, *"Walk through everything that
happens when a user submits a query, and where the maximum latency is"*, *"Explain RAG end to end"*.

The full diagrams (Mermaid + images) are in the main [README §2](../README.md#2-architecture).

---

## How to draw it in 60 seconds

Draw four layers top to bottom and talk while drawing:

```
  [ Streamlit UI / any client ]
              │ HTTP + API key
  [ FastAPI: auth · rate limit · validation · request id ]
         │ questions                      │ uploads
  [ RAG pipeline ]                 [ Ingestion pipeline ]
     │       │        │                    │
  [Cache] [OpenAI] [Qdrant: dense + sparse + metadata]  ── traces ──► [LangSmith]
```

Say: *"Two pipelines share one vector database. Ingestion writes, the RAG pipeline reads. The API layer handles
security and limits; the pipelines don't know about HTTP, so they're easy to test."*

---

## Ingestion: what happens when a document is added

| # | Step | Code | Why |
|---|---|---|---|
| 1 | Parse PDF / DOCX / HTML / CSV / MD; headings → Markdown `#` | `src/ingestion/loader.py` | One heading rule for every format |
| 2 | Metadata from front-matter / sidecar YAML / upload form; normalise (lower-case, dates → `effective_ts` int) | `src/ingestion/metadata.py` | Exact-match filters need consistent values |
| 3 | `doc_id` = hash of file name; `checksum` = SHA-256 of bytes. Same checksum → **skip** | `ingestion_pipeline.py` | Re-runs cost nothing; idempotent |
| 4 | Split by headings; sections > 400 tokens split on paragraph → line → sentence → word with 60-token overlap; tiny pieces merged | `src/chunking/chunker.py` | Keep each policy clause whole; overlap keeps context at boundaries |
| 5 | Prepend `Document: Leave Policy / Section: Sick Leave` to the text **that is embedded** | chunker | "It cannot be carried forward" alone has no subject; the header gives it one |
| 6 | Dense vector: OpenAI `text-embedding-3-small` (1,536 dims), batched 96, cached | `src/embeddings/embedder.py` | Meaning match |
| 7 | Sparse vector: BM25 term weights; Qdrant computes IDF | `src/embeddings/sparse.py` | Exact terms: "POSH", "LTA", policy codes |
| 8 | Delete the old version's chunks, then upsert | `vector_store.py` | No orphaned outdated text |
| 9 | Bump the corpus version → all cached answers invalid | `utils/cache.py` | Never serve a stale policy |

---

## Query: what happens on every question (with real timings)

Real numbers measured in the UI for "How many sick leaves do I get per year?":

| # | Stage | Fresh question | Repeated question | Code |
|---|---|---|---|---|
| 0 | API key → role, rate limit (60/min per key) | <1 ms | <1 ms | `src/api/dependencies.py` |
| 1 | Guardrails: empty, >2,000 chars, injection patterns | 0.02 ms | 0.03 ms | `src/guardrails/guards.py` |
| 2 | **Exact cache** (normalised question, scoped by role + filters + corpus version) | miss | **hit, 0.15 ms → done** | `src/utils/cache.py` |
| 3 | Condense, *only for real follow-ups* | skipped | - | `rag_pipeline.py` |
| 4 | Embed question (cached) | ~655 ms | - | `embedder.py` |
| 5 | Semantic cache (cosine ≥ 0.95 to a past question) | miss | - | `cache.py` |
| 6 | Hybrid search: dense top 40 + BM25 top 40, same filter, **RRF** → top 20 | ~12 ms | - | `vector_store.py` |
| 7 | Rerank top 20 → best 5 (OpenAI scores 0–10, JSON schema) | **~2,600 ms** | - | `reranker.py` |
| 8 | Build context: numbered blocks `[1]…[5]`, 6,000-token budget | <1 ms | - | `prompt_templates.py` |
| 9 | Generate with `gpt-6-luna` (retry → fallback `gpt-5.4-mini`) | ~1,850 ms | - | `llm_client.py` |
| 10 | Validate citations, cache, memory, metrics, trace | <1 ms | - | `rag_pipeline.py` |
| | **Total** | **~6–10 s** | **~0.5–1 ms** | |

### "Where is the maximum latency?"
> "In LLM calls, not in search. Qdrant hybrid search takes about 12 ms; the reranker (~2.6 s) and generation
> (~1.9 s) dominate, plus ~0.7 s for the first embedding. That's why the biggest wins are (1) caching, which removes all of it
> for repeated questions; (2) skipping unnecessary LLM calls, like condensing questions that are already complete;
> (3) a faster reranker (Cohere / cross-encoder, or grading fewer passages); and (4) streaming, so the user sees the
> first words after the rerank instead of waiting for the full answer."

### "Why not just send the whole PDF to the LLM (huge context windows)?"
> "Cost, latency and accuracy. Sending 3,000 PDFs per question is impossible, and even one long PDF costs thousands of
> tokens per question, every time. Models also lose detail in the middle of very long contexts. RAG sends ~5 relevant
> passages (~700 input tokens in my runs), it's faster, it's auditable through citations, and access control is enforced per chunk.
> Long context is useful for one-off analysis of a single document, not for a high-traffic Q&A service."

---

## Key design decisions (say the trade-off)

| Decision | Why | Trade-off |
|---|---|---|
| Hybrid dense + BM25, RRF inside Qdrant | HR queries mix meaning and exact terms | Extra sparse vector per chunk |
| Filters **inside** HNSW search, not after | Post-filtering can return fewer than k results | Needs payload indexes (server mode) |
| Access control as a retrieval filter | The LLM can't leak what it never sees | Roles must be maintained in metadata |
| Rerank 20 → 5 | Search is broad, the reranker is precise | ~2.6 s per fresh question |
| Corpus-versioned cache | Policy updates can't serve stale answers | Whole cache invalidated on any change |
| Deterministic ids (uuid5) | Re-ingest overwrites, never duplicates | Renaming a file creates a new document |
| Static system prompt + `prompt_cache_key` | OpenAI prompt caching reuses the prefix | Per-request data must go in the user message |
| Own retry policy (SDK retries off) | One place to tune and log; no retry on 4xx | More code to own |
| Fail-open reranker | An outage lowers quality instead of failing the request | Quality silently drops; monitor `rerank` errors |
| Thin Streamlit client over REST | Any frontend (Teams, React) can reuse the API | Two processes to run |

---

## Tech stack, and why each piece

| Layer | Choice | Why this over alternatives |
|---|---|---|
| API | FastAPI + Pydantic | Async-capable, automatic validation and OpenAPI docs, typed request/response models |
| Vector DB | Qdrant | Native hybrid (dense + sparse) with server-side RRF, filterable HNSW, payload indexes, local mode for dev |
| Embeddings | OpenAI `text-embedding-3-small` | Strong quality/price, adjustable `dimensions` |
| LLM | `gpt-6-luna` (+ `gpt-5.4-mini` fallback) | Fast and cheap for reading-heavy RAG; reasoning effort tunable |
| Reranker | OpenAI LLM grading (Cohere optional) | No extra vendor; structured JSON output |
| Cache | In-memory TTL-LRU, Redis for multi-worker | Simple locally; shared and persistent in production |
| Tracing | LangSmith | Nested traces of retriever / rerank / LLM, feedback, dataset experiments |
| Metrics | Prometheus | Standard; per-stage latency histograms |
| UI | Streamlit | Fast to build; streams via Server-Sent Events from the API |
| Deploy | Docker + compose (API, UI, Qdrant, Redis) | Same setup everywhere; non-root image with a healthcheck |
