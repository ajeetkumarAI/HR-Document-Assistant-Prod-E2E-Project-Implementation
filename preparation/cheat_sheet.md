# ⭐ Cheat Sheet (read 10 minutes before the interview)

## The one-liner
> "A production RAG HR assistant: hybrid search in Qdrant, OpenAI reranking and generation, role-based access inside retrieval,
> versioned caching, retries + fallback, LangSmith tracing, FastAPI + Streamlit, Docker, 45 tests."

## Numbers to remember

| What | Value |
|---|---|
| Chunk size / overlap / minimum | **400 / 60 / 30 tokens**, split by headings first |
| Embeddings | OpenAI `text-embedding-3-small`, **1,536** dims, batch 96, cached |
| Retrieval | Hybrid dense + BM25, **RRF**, each branch top **40** → fused top **20** |
| Rerank | OpenAI scores 0–10 → best **5**; fail-open |
| HNSW | `m=16`, `ef_construct=200`, `ef_search=128`; optional int8 quantization (4× less RAM) |
| Context budget | **6,000** tokens; answer max **1,024** tokens |
| LLM | `gpt-6-luna` (reasoning effort low) → fallback `gpt-5.4-mini` |
| Retries | **4** attempts, 0.5 s → doubling, max 20 s, jitter; only 429 / 5xx / timeouts |
| Rate limit | **60** requests/min per API key (token bucket) |
| Cache | exact + semantic (cosine **≥ 0.95**) + embedding; TTL **24 h**; 5,000 entries; Redis option |
| Chat memory | last **6** turns, 1 h TTL; follow-ups condensed, history not resent |
| Guardrails | ≤ **2,000** chars, injection patterns, citation validation, PII masking in logs |
| Roles | employee → public · manager → + manager · hr_admin → + confidential |
| Latency (fresh) | ~6–10 s total: rerank ~2.6 s, generate ~1.9 s, embed ~0.7 s, **search ~12 ms** |
| Latency (repeat) | **~1 ms, 0 tokens** (exact cache) |
| Corpus | 7 sample policies → **31 chunks** |
| Tests | **45** offline tests + golden set (16 questions: hit@k, MRR, keyword recall, refusal, p50/p95) |

## Keywords to use naturally
hybrid search · Reciprocal Rank Fusion · filterable HNSW · payload indexes · contextual chunk headers · reranking (recall then
precision) · access control at retrieval time · corpus-versioned cache · semantic cache · prompt caching · idempotent ingestion ·
fail-open vs fail-closed · exponential backoff with jitter · model fallback · structured outputs · groundedness · golden set ·
LLM-as-judge · request ID correlation · per-stage latency · least privilege · indirect prompt injection.

## Five things that make this production-grade (if asked "what makes it production-grade?")
1. **Security in data access**, not the prompt: role filters inside the vector search.
2. **Reliability:** retries on temporary errors only, fallback model, fail-open reranker, timeouts, rate limits.
3. **Cost and latency:** three cache layers, cache checked before any LLM call, condense only for follow-ups.
4. **Data lifecycle:** checksum skip, versioned replace, automatic cache invalidation.
5. **Observability and quality:** request IDs, metrics, LangSmith traces + feedback, tests, golden-set evaluation.

## Honest gaps (say them before they ask)
Not an agent (deterministic pipeline by design) · rate limiter is per process (move to Redis/gateway) · no circuit breaker yet ·
pattern-based injection detection (add a classifier) · static API keys (use SSO in a company) · MMR/near-duplicate removal not yet
in the pipeline · uploads are synchronous (use a queue for bulk).

## If you blank on a question
"Let me think about it in terms of my project…", then map it: *retrieval? generation? reliability? security? cost?*
Every GenAI question lands in one of those five boxes.
