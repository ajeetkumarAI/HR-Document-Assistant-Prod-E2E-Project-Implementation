# 01: Project Pitch, Contribution and Stories

Answers to: *"Tell me about yourself / your project"*, *"Explain your current project"*, *"What was your exact
contribution?"*, *"Biggest technical challenge?"*, *"Tell me about a failure"*, *"How did you evaluate it?"*

---

## 30-second pitch (prescreening calls, recruiters)

> "I built a production-grade RAG assistant that answers employee questions about HR policies, such as leave, travel
> and work-from-home rules, strictly from company documents, with citations. It uses hybrid search in Qdrant,
> OpenAI reranking and generation, role-based access so employees never see confidential documents, caching that
> cuts repeated questions from about 10 seconds to 1 millisecond, and full tracing in LangSmith. It has a FastAPI
> backend, a Streamlit UI, Docker deployment and 45 automated tests."

## 2-minute pitch (start of a technical round)

> **Problem.** Employees kept asking HR the same policy questions. The answers live in long PDFs that change
> regularly, and some documents (salary bands, performance guidelines) must only be visible to certain roles.
>
> **Solution.** A RAG system with two pipelines.
> - **Ingestion:** documents (PDF, DOCX, Markdown, CSV…) are parsed, split by heading into ~400-token chunks with
>   60-token overlap, and each chunk is stored in Qdrant with a meaning vector (OpenAI embeddings), a keyword vector
>   (BM25) and metadata: category, department, effective date and access level. Re-ingestion is idempotent: unchanged files are
>   skipped by checksum, changed files replace their old chunks.
> - **Query:** guardrails check the question → exact and semantic cache → hybrid search with metadata filters and
>   the caller's access level applied *inside* the search → an OpenAI model reranks the top 20 to the best 5 →
>   `gpt-6-luna` answers with numbered citations → citations are validated → the answer is cached.
>
> **Production concerns.** Retries with exponential backoff only on temporary errors, a fallback model, rate
> limiting per API key, PII masking in logs, Prometheus metrics, LangSmith traces, and a golden-set evaluation script.
>
> **Result.** Correct, cited answers. Repeated questions come back in ~1 ms with zero tokens, and an employee asking
> about salary bands gets "I couldn't find this" while an HR admin gets the answer.

## 5-minute version

Use the 2-minute pitch, then walk the query flow from [`02_architecture_walkthrough.md`](02_architecture_walkthrough.md),
then close with **one challenge** and **one result** from the stories below.

---

## "What was your exact contribution?"

Say "I", and be specific:

- **Designed the retrieval:** hybrid dense + BM25 with Reciprocal Rank Fusion in Qdrant, structure-aware chunking with a
  context header ("Document: Leave Policy / Section: Sick Leave") prepended before embedding.
- **Built access control into retrieval**, not the prompt: the role's allowed `access_level`s are added to every
  Qdrant filter, so confidential chunks are never retrieved for an employee.
- **Built the reliability layer:** retry policy (`with_retry`), model fallback, fail-open reranker, timeouts, rate limiting.
- **Built the caching layer:** embedding cache, exact + semantic answer cache, versioned by corpus so a policy update can
  never return a stale answer.
- **Built observability:** JSON logs with request IDs and PII masking, Prometheus metrics, LangSmith tracing with
  user feedback.
- **Wrote the tests and evaluation:** 45 offline tests (fake LLM, in-memory Qdrant) and a golden-question evaluation
  (hit@k, MRR, keyword recall, latency).

---

## STAR stories (real issues found and fixed in this project)

Use these for *"biggest challenge"*, *"a failure"*, *"a mistake you made"*, *"debugging"* questions.

### Story 1: Keyword search silently missed plural forms (a bug I caught in my own code)
- **Situation:** Hybrid search combines meaning and keyword matches. While documenting the BM25 tokenizer I traced
  what it did to "leave" vs "leaves".
- **Task:** Make sure keyword matching treats different forms of a word as the same.
- **Action:** Found that "leaves" became `leav` but "leave" stayed `leave`, so they never matched. I rewrote the
  stemmer with ordered rules (`ies→y`, `sses/xes/ches→strip es`, plural `s`, then drop a final silent `e`), tested
  it on 13 word pairs, and added a regression test (`test_stemmer_aligns_word_forms`).
- **Result:** "leave/leaves", "employee/employees" and "approve/approved/approving" now match, and the index is rebuilt with
  `python -m scripts.ingest --force`.
- **Lesson:** Read your own preprocessing output. A silent mismatch like this lowers recall without any error.

### Story 2: The cache wasn't used inside conversations (found from real latency numbers)
- **Situation:** In the UI, asking the *same* question twice still took ~6.6 s and ~800 tokens.
- **Task:** Find why the cache didn't help.
- **Action:** Read the per-stage `timings_ms`. `condense` took 3,283 ms. The pipeline skipped the cache whenever chat
  history existed, always ran an LLM "condense" call, and resent the whole history to the answer model. I
  (1) moved the exact-cache check before any LLM call, (2) added a cheap follow-up detector so condense only runs for
  real follow-ups ("can I carry *them* forward?"), (3) stopped resending history, and (4) made cached answers report 0 tokens.
- **Result:** Repeated question: **10.01 s → 1 ms, 746 tokens → 0**. New self-contained questions save the 1.5–3.3 s condense call.
- **Lesson:** Instrument every stage. The timing breakdown pointed straight at the cause.

### Story 3: Dev server restarted itself in an endless loop
- **Situation:** Running the API in dev mode logged "1 change detected" three times a second.
- **Action:** The auto-reloader watched the whole project, including `logs/app.log`. Every restart wrote a log line,
  which triggered another restart. Fixed by watching only `src/` (`reload_dirs` in `main.py`).
- **Lesson:** Side effects of observability (log files) can interact with tooling in unexpected ways.

### Story 4: Worked in tests, failed on the target machine (Python version)
- **Situation:** Tests passed on Python 3.13 but failed on a 3.10 machine with `cannot import name 'UTC'`.
- **Action:** `datetime.UTC` only exists from 3.11. Replaced it with `timezone.utc`, set `requires-python >=3.10`
  and ruff's `target-version = py310` so the linter prevents it from happening again.
- **Lesson:** Pin and lint for the oldest Python you support.

### Story 5: Reranker slowed every query (trade-off decision)
- **Situation:** Reranking with an LLM took ~2–2.6 s, the largest stage for new questions.
- **Decision:** Kept it, because answer quality matters more for HR policy, but made it **fail-open**, ran it with `reasoning_effort="none"` and
  capped each passage at 1,500 characters. Options to cut further: grade 10 instead of 20 passages, or switch to
  Cohere Rerank (`reranker.provider: cohere`) which is purpose-built and faster.

---

## "How did you evaluate it?"

- **Offline:** `scripts/evaluate.py` runs a golden set (`data/eval/golden_qa.jsonl`, 16 questions with role, expected
  source and expected facts) and reports **hit@k**, **MRR**, **answer keyword recall**, **refusal accuracy** (questions
  the role shouldn't be able to answer must be refused) and **p50/p95 latency**. `--langsmith` uploads it as a
  LangSmith experiment so prompt/model changes can be compared side by side.
- **Automated tests:** 45 tests covering chunking, metadata, hybrid search, filters, access control, cache,
  guardrails, retries/fallback and the API.
- **Online:** LangSmith traces per request, 👍/👎 feedback sent to LangSmith with the `run_id`, Prometheus metrics for
  latency per stage, cache hit rate, tokens, errors and "no context" answers.

## "Tell me about a failure in your GenAI project"
Use Story 2 (performance) or Story 1 (silent quality bug). Both show you **measured, found the root cause, fixed it,
and added a test**.
