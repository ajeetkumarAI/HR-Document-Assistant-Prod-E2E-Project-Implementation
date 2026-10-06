# 05: Production Engineering

What separates a POC from a production GenAI system, with how this project does each part.

---

## POC vs production: what's different?

| Concern | POC | Production (this project) |
|---|---|---|
| Config | Hard-coded | Typed settings: `config.yaml` + `.env`, validated at startup (`src/utils/config.py`) |
| Failures | Crash | Retry with backoff, fallback model, fail-open reranker, timeouts |
| Cost | Ignored | Embedding + exact + semantic cache, prompt caching, rate limits, token metrics |
| Security | None | API keys → roles, access control in retrieval, guardrails, PII masking, secrets in `.env` |
| Data updates | Rebuild everything | Idempotent ingestion: checksum skip, versioned replace, cache invalidation |
| Observability | `print()` | JSON logs + request IDs, Prometheus metrics, LangSmith traces + feedback |
| Quality | Eyeballing | 45 tests + golden-set evaluation |
| Deployment | Notebook | Docker (non-root, healthcheck), compose with Qdrant + Redis, `/health` & `/ready` |

---

## Reliability

### Design a retry mechanism with backoff. What about 400 / 401?
**In this project:** `with_retry()` in `src/utils/helpers.py` (tenacity):
- up to **4 attempts**, waits ~0.5 s → 1 s → 2 s… (doubling, max 20 s) **plus random jitter** so many clients don't retry
  at the same moment;
- retries **only temporary errors**: 429 rate limit, 5xx, timeouts, connection errors (`_is_retryable`);
- **never retries 400/401/404**: a bad request or wrong key will fail identically, so retrying only adds delay;
- SDK retries are turned off (`max_retries=0`) so attempts don't multiply (4 × 3 = 12 calls).
Code to write live: [`09_coding_round.md`](09_coding_round.md#1-retry-with-exponential-backoff).

### Making retries safe for thousands of concurrent users
Jitter (avoids synchronised retry storms), a cap on attempts and on total wait, respect for the `Retry-After` header, a
**circuit breaker** (stop calling a failing service for a cool-down period), a global rate limiter in front of the provider,
and queueing or load shedding. *Honest note:* this project has jitter, caps and fallback; a circuit breaker is the next addition.

### How do you handle model failure / model availability?
Retries → **fallback model** (`llm.fallback_model`; the response shows `used_fallback_model: true`) → if everything fails,
HTTP 502 with a request ID. Streaming falls back only before the first token. Monitor `rag_llm_errors_total`. For
higher availability: multiple providers or regions behind a gateway.

### How would you handle a model rate limit?
Client side: backoff + jitter on 429, respect `Retry-After`, fallback model, cache, smaller prompts. Capacity side: request
a higher tier, spread load across deployments/regions, batch embeddings (this project batches 96 texts per call).

### Fail-open vs fail-closed
Reranker, query rewriting and Redis cache **fail open** (degrade quality or speed, keep serving). Auth, access
control and guardrails **fail closed** (refuse). Choose by "what's worse: a slower or slightly worse answer, or a
security breach?"

---

## Caching

### Implement caching for repeated queries. Limitations of an in-memory dict? Would you cache everything?
**In this project** (`src/utils/cache.py`):
- **Exact cache:** key = normalised question, scoped by role + filters + **corpus version**. Hit → ~1 ms, 0 tokens.
- **Semantic cache:** cosine ≥ 0.95 to a past question in the same scope.
- **Embedding cache:** never embed the same text twice (saves cost on re-ingestion and repeated questions).
- Memory backend = TTL + LRU (`cachetools.TTLCache`, 5,000 entries, 24 h); **Redis** for multiple workers.

**Limits of a plain dict:** grows forever (no TTL/LRU) → memory leak; not shared across workers/servers; lost on restart; not
thread-safe without a lock; no invalidation.

**Don't cache everything:** never cache "not found" answers (the document may be uploaded later); never share across
roles (an admin's answer could leak confidential data to an employee, hence role in the scope); invalidate on document
change (corpus version); be careful with personalised or time-sensitive answers; follow-ups must be condensed to a
standalone question before using it as a key.

### Why did the cache not work in conversations at first? → Story 2 in [`01_project_pitch.md`](01_project_pitch.md)

---

## Rate limiting

### What is rate limiting? How is it implemented here?
Limiting requests per client per time window to protect cost and capacity. **In this project:** a **token bucket** per
API key (`TokenBucketLimiter`): 60 tokens, refilled at 1/second, allowing short bursts; empty bucket → HTTP 429. It's in-process;
with several servers move it to **Redis** or the API gateway (Kong, Azure APIM, Cloud Armor).

---

## Concurrency and Python

### Why async in an AI app? 3 independent API calls: sequential or concurrent?
LLM apps are **I/O-bound** (waiting on OpenAI, the vector DB, Redis). Concurrency overlaps the waiting: three 1-second calls
take ~1 s concurrently (`asyncio.gather`) vs ~3 s sequentially. **In this project:** FastAPI runs the blocking pipeline
in its thread pool (handlers are plain `def`), uploads use `run_in_threadpool`. Running dense and sparse search
concurrently is already done server-side in Qdrant (prefetch).

### Concurrency vs parallelism; GIL; multithreading vs multiprocessing
- **Concurrency** = dealing with many tasks by interleaving them (one core). **Parallelism** = running at the same time (many cores).
- **GIL** = only one thread executes Python bytecode at a time. Threads are still fine for **I/O-bound** work (the GIL is
  released while waiting on the network); for **CPU-bound** work (parsing thousands of PDFs, local embeddings) use
  **multiprocessing** or native libraries that release the GIL.
- Python 3.13 has an optional free-threaded build without the GIL.

### Why FastAPI for GenAI backends? Where does Pydantic fit?
Async support, high performance, **automatic validation** and OpenAPI/Swagger docs from **Pydantic** models, dependency
injection (this project: auth + rate limit as `Depends`), streaming responses (SSE). **In this project:** `schemas.py`
defines request/response models; invalid bodies get a consistent 422; settings are Pydantic too.

### Generators and streaming LLM responses; decorators vs generators
A **generator** `yield`s values one at a time, which is perfect for streaming tokens to the user as they arrive (this
project: `rag_pipeline.stream()` yields `sources` → `token`… → `done` events, served as Server-Sent Events). A
**decorator** wraps a function to add behaviour (this project: `@traceable` for tracing, `with_retry` for retries).

### Process large data without running out of memory
Stream and batch instead of loading everything: generators, `batched()` (this project embeds 96 and upserts 128 at a time),
chunked file reading, pandas `chunksize`, and capped upload reads (`f.read(limit + 1)` rejects >25 MB without loading
a huge file).

---

## Observability

### What do you monitor in production?
- **Latency per stage** (p50/p95): `rag_stage_seconds{stage=…}`.
- **Tokens and cost** by model: `rag_llm_tokens_total`.
- **Errors:** LLM errors, fallback usage, 5xx rate, reranker failures.
- **Cache hit rate:** `rag_cache_events_total`.
- **Quality signals:** "no context" answers, 👍/👎 feedback, guardrail blocks.
- **Traffic:** requests per key, 429s.
**In this project:** all exposed at `/metrics`; each request is traced in LangSmith; logs carry `request_id`.

### What is observability? What would you log?
Being able to answer "why did this request behave this way?" from outside. Log a request ID, role, cache status, stages
and timings, model, tokens, citations count and errors; **mask PII** (emails, phones, PAN, Aadhaar, API keys) as
`logger.py` does. Don't log raw secrets.

### Debug a high-latency RAG pipeline / "production suddenly slow"
1. Look at per-stage timings (`timings_ms` / Prometheus) and find the slow stage.
2. LLM stages slow → provider latency, longer prompts (context budget), retries happening (check logs), fallback in use.
3. Retrieval slow → index not built (local mode is brute force), missing payload indexes, `ef_search` too high, huge `top_k`.
4. Everything slow → CPU/memory saturation, thread pool exhausted, Redis down (falling back), rate limiting.
5. Check what changed: deploys, data volume, traffic pattern, cache hit rate dropped (e.g., cache cleared by re-ingestion).

### Responses got worse after a deployment
Diff the prompt version, model, chunking and retrieval settings; re-run the golden set on both versions; check if re-ingestion
changed the chunks; roll back (config switch or previous image) while investigating. Prevent it with golden-set
gating in CI, canary releases and prompt versioning.

---

## Cost

### Budget was $2,000/month, now $18,000 and rising. Audit and optimise.
1. **Measure:** tokens per request by stage and model (`rag_llm_tokens_total`, LangSmith), requests per user or key, cache hit rate.
2. **Find the leak:** runaway retries, a bot or abusive user, huge contexts, resending chat history every turn (Story 2),
   an expensive model used for simple steps, reranking every request.
3. **Fix:** caching (exact + semantic), rate limits per key, cap context (top_n, token budget), condense instead of resending
   history, a cheaper model for utility steps (rerank, rewrite), prompt caching (static system prompt), smaller embedding
   `dimensions`, batching, and an alert when spend goes over budget.

### Accuracy is excellent but cost very high / cheap but accuracy poor
Find the cheapest configuration that passes your quality bar on the golden set: route simple questions to a cheap
model and hard ones to a strong model; cache; rerank to fewer passages. If accuracy is poor: fix retrieval first
(cheapest lever), then the prompt, then a stronger model only where needed.

---

## Scaling

### 10,000 → 5,000,000 documents: latency crashed, costs spiked. Re-architect.
- Qdrant **server/cluster** with sharding and replicas; HNSW tuned; **int8 quantization** (4× RAM saving); vectors/HNSW on disk.
- **Payload indexes** on filter fields; route by department/tenant (filter or separate collections).
- **Hierarchical retrieval:** first find documents (summary index), then chunks.
- Async **ingestion workers + a queue** (only changed files, as checksums do here); embed in batches; cache embeddings.
- Smaller `dimensions` (e.g., 512) to cut storage and speed up search; rerank fewer candidates.

### Corporate chatbot for 10,000 employees; usage grows 10×
Stateless API replicas behind a load balancer (Cloud Run / Kubernetes autoscaling), **Redis** for cache, sessions and rate
limits, a Qdrant cluster, higher provider quota plus a fallback model, streaming responses, and async ingestion workers.
The biggest win is the cache, since HR questions repeat heavily. Load-test and watch p95 per stage.

### Real-time responses
Streaming (SSE), caching, skipping unnecessary LLM calls (condense only for follow-ups), a fast model, fewer rerank candidates,
keeping the client connection warm, and colocating services in one region.

### Millions of documents: async processing, duplicates, a 20-minute job, failures
Queue (Pub/Sub, SQS, Kafka) → workers; **idempotency** via deterministic IDs and checksums (this project) so retries never
duplicate; status tracking per document (`indexed/updated/skipped/failed`, as the ingestion report does); dead-letter
queue for repeated failures; return `202 Accepted` + a job ID for long jobs instead of keeping the HTTP request open.

### What is idempotency?
Doing an operation twice has the same effect as doing it once. **In this project:** re-ingesting the same file is a no-op
(checksum), and changed files overwrite by deterministic `chunk_id`, so there are never duplicates.

---

## Deployment and LLMOps

### How would you deploy this safely? (Cloud Run vs GKE)
Docker image (already in the repo) → registry → **Cloud Run** / Azure Container Apps / ECS for a stateless API with
autoscaling; Qdrant Cloud or a managed cluster; Redis (Memorystore / Azure Cache); secrets in Secret Manager / Key Vault;
`/ready` as the readiness probe. Choose **GKE / Kubernetes** when you need GPUs for self-hosted models, sidecars, or
fine-grained networking.

### Data must never leave the enterprise environment
Use models inside the company boundary: **Azure OpenAI / Vertex AI / Bedrock** in the company's tenant with a private endpoint
and no data retention, or **self-hosted open models** (Llama/Mistral via vLLM on GPUs) plus open embeddings. Qdrant
self-hosted in the VPC; no public egress; encryption at rest and in transit. **In this project:** `OPENAI_BASE_URL` points
the client at Azure OpenAI or a gateway without code changes; `store=False` stops the provider persisting responses.

### MLOps vs LLMOps; prompt versioning; canary; shadow testing; rollback; CI/CD
- **LLMOps** adds prompt and version management, evaluation sets, token cost tracking and guardrails to MLOps.
- **Prompt versioning:** `PROMPT_VERSION` in code, logged and attached to every trace so quality can be compared across versions.
- **Canary:** send 5% of traffic to the new version and compare metrics before full rollout. **Shadow:** run the new version
  on real traffic without showing users its answers, and compare.
- **Rollback:** previous image tag or config flag (model/prompt in `config.yaml`).
- **CI/CD:** this repo's GitHub Actions runs ruff + 45 tests on every push; add golden-set evaluation as a release gate.
- **Testing GenAI is different:** outputs are non-deterministic, so tests use fakes for logic, golden sets with tolerant
  metrics for quality, and regression tracking over time.
