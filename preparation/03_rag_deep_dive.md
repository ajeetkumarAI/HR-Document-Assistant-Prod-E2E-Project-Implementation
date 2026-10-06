# 03: RAG Deep Dive

Each answer: **short answer** → **in this project** → **trade-off / follow-up**. Read the short answers out loud.

---

## Fundamentals

### What is RAG? Explain it to a CEO.
**Technical:** Retrieval-Augmented Generation: retrieve the most relevant passages from your own documents, put them in the
prompt, and have the LLM answer only from them, with citations.
**CEO version:** "It's an open-book exam. Instead of answering from memory, the AI looks up the right page of our policy
handbook first and shows you the page it used."

### RAG vs fine-tuning? When would you use both? When NOT to use RAG?
- **RAG** changes what the model *knows at question time*: for facts that change (policies updated weekly), need
  citations, or need per-user access control. **Fine-tuning** changes how the model *behaves*: tone, format,
  domain jargon, a narrow classification task.
- **Both:** fine-tune for style or format (e.g., always answer in the HR team's template) and use RAG for the facts.
- **Don't use RAG** when the answer doesn't depend on private or up-to-date documents (general writing, maths), when the
  task is pure transformation of the given input (summarise this email), or when the data is structured and a SQL
  query answers it exactly.
- **In this project:** HR policies change and differ by role, so RAG is the right choice: no retraining when a policy changes, just
  re-ingest.

### Explain embeddings without using the word "vector".
> "An embedding is a list of numbers that acts like GPS coordinates for meaning. Sentences that mean similar things get
> coordinates close together, even with different words: 'time off when I'm unwell' lands next to 'sick leave policy'.
> Search becomes 'find the passages whose coordinates are nearest to the question's coordinates'."

### What is cosine similarity?
The cosine of the angle between two embeddings: 1 = same direction (same meaning), 0 = unrelated. It ignores length,
so a long and a short text about the same thing still match. **In this project:** Qdrant `distance: cosine`; the semantic
cache reuses an answer when cosine ≥ **0.95**.

### Tokenization vs chunking?
Tokenization splits text into the model's units (~4 characters each), the units LLMs read and bill. Chunking splits a
*document* into retrievable passages. **In this project** chunk size is measured **in tokens** (400) via `tiktoken`, so
chunks fit predictably in the 6,000-token context budget.

### Can a RAG system work without a vector database?
Yes. BM25 keyword search alone (Elasticsearch / OpenSearch), Postgres full-text search, or `pgvector` inside
Postgres all work. A vector DB adds fast approximate nearest-neighbour search, filtering and scale. **In this project** Qdrant holds
**both** dense and BM25 sparse vectors, so one system does both.

### SQL database vs vector database? Can PostgreSQL do vector search?
SQL: exact matches, joins, transactions. Vector DB: similarity search over embeddings, plus metadata filters. Postgres
with `pgvector` (HNSW/IVF indexes) is a good choice when you already run Postgres and have up to a few million vectors.

---

## Chunking

### What chunk size do you use, and why overlap?
- **400 tokens, 60 overlap, minimum 30** (`config.yaml → chunking`). HR clauses are short; 400 tokens holds a full
  sub-section like "Sick Leave" with its conditions. Smaller chunks retrieve more precisely; larger chunks carry more context.
- **Overlap** repeats the last ~60 tokens of one chunk at the start of the next, so a rule that spans a boundary
  ("...requires a certificate. | It must be uploaded within 5 days") is still complete in at least one chunk.

### Chunking strategies, and the parameters you pass
| Strategy | How | When |
|---|---|---|
| Fixed size (characters/tokens) | Cut every N tokens | Quick baseline; breaks sentences |
| Recursive | Try paragraph → line → sentence → word | General text |
| **Structure-aware** (this project) | Split by headings first, then recursive | Policies, manuals, docs with headings |
| Semantic | Split where embedding similarity drops | Unstructured prose |
| Parent-child | Retrieve small chunks, return the larger parent | Need precision *and* context |
| Table/row-aware | Keep rows together with headers | CSVs, tables (this project: each CSV row → `column: value` lines) |

Parameters: `chunk_size`, `chunk_overlap`, `separators`, `min_chunk_size`, and whether to add a context header.

### Is fixed-character chunking optimal for all documents?
No. It cuts mid-sentence and mid-table and ignores structure. **In this project**: headings define sections, tables
become `a | b | c` rows (DOCX) or `column: value` lines (CSV), and tiny pieces are merged into their neighbour.

### Why add a context header to chunks? (strong differentiator)
A chunk like "It cannot be carried forward" has no subject. Prepending `Document: Leave Policy / Section: Sick Leave` to the
**embedded** text (not the text shown to the LLM) makes it findable. This is also called "contextual chunk headers".

---

## Retrieval

### What is hybrid search? Why use it?
Run **dense** (meaning) and **sparse/BM25** (exact words) search and merge the results. Dense handles paraphrases;
BM25 handles acronyms, codes and names ("POSH", "Form 16", "B3"). **In this project** Qdrant runs both with the same
filter and merges with **Reciprocal Rank Fusion**: `score = Σ 1/(60 + rank)`. It uses ranks rather than scores
because cosine and BM25 scores are on different scales.

### What is HNSW? What do m / ef_construct / ef_search do?
Hierarchical Navigable Small World: a graph where each vector links to its nearest neighbours, so search hops through
the graph instead of comparing with every vector. **In this project:** `m=16` (links per node, more = better recall
and more RAM), `ef_construct=200` (effort when building), `ef_search=128` (effort per query, the recall/latency knob).
Optional int8 **scalar quantization** cuts RAM by 4× for large corpora.

### What is metadata filtering, and why is it critical for enterprise RAG?
Restricting search to chunks whose metadata matches: category, department, date range, access level. It gives accuracy
(only current travel policies) and **security** (only documents your role may see). **In this project:** filters are
applied **inside** the HNSW search (filterable HNSW + payload indexes), not after, so you still get the full top-k
results. `effective_ts` is stored as an integer (`20260101`) for fast range filters.

### What is reranking? Why retrieve 20 and keep 5?
Search compares the question and each chunk *separately* (fast, rough). A reranker reads the question and each passage
*together* (slow, precise). So: broad recall with search (top 20), precision with the reranker (best 5). **In this
project:** an OpenAI model scores each passage 0–10 with a strict JSON schema; Cohere Rerank is an option. It's
fail-open: on error the search order is kept.

### Top 5 retrieved chunks are near-duplicates. How do you fix it?
Near-duplicates waste context slots. Fixes: **MMR** (Maximal Marginal Relevance) to penalise similarity to
already-picked chunks; dedupe by cosine > 0.95 between candidates; dedupe at ingestion by content hash; collapse by
`doc_id` (max N chunks per document). *Honest note:* this project dedupes at ingestion (checksum, deterministic ids) and
relies on the reranker. MMR is the next step (see [`09_coding_round.md`](09_coding_round.md) for the code).

### User queries diverge from document wording. How do you improve retrieval?
Query rewriting (this project condenses follow-ups), **multi-query** (generate 3 paraphrases and merge), **HyDE**
(embed a hypothetical answer), hybrid search (this project), synonyms or a domain glossary, and the context headers on chunks.

### What is query rewriting? Give an example.
"Can I carry them forward?" (after a sick-leave answer) → "Can I carry forward unused sick leave to the next year?".
**In this project:** `_condense()` does this with the utility model, **only** when `looks_like_follow_up()` detects a
reference word ("them", "that", "what about…"), which saves 1.5–3 s on self-contained questions.

### Multi-query retrieval, contextual compression, parent-child and hierarchical retrieval?
- **Multi-query:** several rephrasings, union of results. Better recall, more cost.
- **Contextual compression:** trim retrieved chunks to only the relevant sentences before sending them to the LLM.
- **Parent-child:** index small chunks for precise matching, return their parent section for context.
- **Hierarchical:** first find the right document (summaries), then search within it. Scales to millions of documents.
*In this project:* structure-aware chunks plus the section path give much of parent-child's benefit; the others are
extensions.

### What if no relevant document is retrieved?
**In this project:** if nothing survives filters and reranking, the pipeline returns a fixed *"I couldn't find this in
the HR documents… contact HR"* **without calling the LLM**, and those answers are never cached (the document might be
uploaded later). The system prompt also tells the model to say so when the context doesn't contain the answer.

### How do you handle tables, scanned PDFs, and PDFs with images?
- **Tables:** keep rows intact with their headers (this project: DOCX tables → pipe rows, CSV → `column: value`); for
  complex tables use a table-aware parser (e.g., Docling, Unstructured) or a multimodal model to convert them to Markdown.
- **Scanned PDFs:** no text layer. This project detects that and raises *"Run OCR first, e.g. `ocrmypdf`"*.
- **Images and diagrams:** caption them with a multimodal model at ingestion and index the caption.

### Restricted vs unrestricted data during retrieval?
**In this project:** every chunk has `access_level` ∈ {public, manager, confidential}; `role_access` maps roles to levels;
the filter `access_level IN allowed` is added to **every** search before user filters, and the role comes from the API
key, never from the request body. A typo in metadata falls back to `public`, so mark sensitive files carefully.

---

## Citations and hallucination

### How do you implement citations?
Number the passages in the prompt (`[1] Source: Leave Policy | Section: … | Effective: …`), instruct the model to cite
`[n]`, then **validate**: `extract_citations()` removes any `[n]` that doesn't match a real passage (e.g., `[7]` when 5
were sent) and the API returns source, section, page, effective date and rerank score for each citation.

### "The answer is correct but the citation is wrong."
Check whether the model cited a neighbouring passage (duplicates or overlap), whether passage numbering changed after
context-budget truncation, and whether the citation instruction is clear. Fixes: dedupe near-identical passages, ask the model to
quote the supporting sentence, add a citation-accuracy check to evaluation (LLM-as-judge: "does passage [n] support this
claim?").

### Why do LLMs hallucinate? Can hallucination be eliminated completely?
LLMs predict likely next tokens; they don't look facts up. Hallucination comes from missing or wrong context, ambiguous
questions, high temperature, long noisy contexts, and pressure to always answer. **It can't be eliminated, only
reduced and detected.** Say that honestly to clients.

### How do you reduce hallucination in RAG? (with weak retrieval)
1. Better retrieval: hybrid, rerank, filters, context headers.
2. A grounding prompt: "answer only from context; say you couldn't find it" (this project's `SYSTEM_PROMPT`).
3. A **refusal path** when retrieval is empty or weak (this project) and an optional `reranker.score_threshold`.
4. Citations + validation, so every claim is traceable.
5. Low reasoning effort / temperature for factual answers.
6. Evaluation: groundedness / faithfulness checks (LLM-as-judge, RAGAS) on a golden set.

### Diagnose a chatbot confidently returning facts not in the knowledge base
Take the `request_id` → LangSmith trace → check: (a) **what was retrieved** (the right chunk? wrong version? missing
because of filters?), (b) **what was sent** (did the context contain the fact?), (c) **what the model did** (ignored the
context or invented details?). If retrieval failed, fix chunking/filters/hybrid; if the model ignored the context, tighten
the prompt, lower the temperature, add a groundedness check, or switch model. Add the case to the golden set so it
can't regress.

### "Retrieved chunks are correct but the answer is still wrong or incomplete."
The problem is in generation: context too long (truncated by the budget?), the answer is spread across chunks the model
didn't combine, conflicting versions (this project puts `Effective:` in each block and tells the model to prefer the newest),
the prompt is unclear, or the model is too weak. Inspect the exact prompt in LangSmith, then iterate on the prompt or model
against the golden set.

---

## Evaluation

### How do you evaluate a RAG system?
Two layers:
- **Retrieval:** Precision@K, Recall@K, **hit@k** (was the right source retrieved?), **MRR** (how high it ranked).
- **Generation:** faithfulness/groundedness (are claims supported by the context?), answer relevance, correctness against the
  expected facts, citation accuracy, refusal accuracy.
- **System:** latency p50/p95, cost per query, cache hit rate, error rate.
**In this project:** `scripts/evaluate.py` computes hit@k, MRR, keyword recall, refusal accuracy, p50/p95; `--langsmith`
runs it as an experiment.

### Precision@K, Recall@K, MRR
- **Precision@K:** of the K retrieved, what fraction are relevant.
- **Recall@K:** of all relevant chunks, what fraction are in the top K.
- **MRR:** average of 1/rank of the first relevant result (1.0 = always first).

### What is a golden dataset? How would you build one for RAG?
A fixed set of questions with expected answers or sources, used to catch regressions. Build it from real user
questions (logs), HR FAQs and edge cases (follow-ups, access-restricted questions, unanswerable questions); store the expected
source + key facts. **In this project:** `data/eval/golden_qa.jsonl` (16 cases, including one that must be refused for an employee).

### What is LLM-as-a-judge?
Using a strong LLM with a rubric to grade outputs (groundedness, relevance, tone). Cheap and scalable but biased; calibrate
it against human labels and keep the rubric fixed. Useful for comparing prompt versions on the golden set.

### Offline vs production evaluation
Offline: golden set before each release (prompt/model/chunking changes). Production: LangSmith traces, user 👍/👎, "no
answer" rate, cache hit rate, latency and cost dashboards, plus sampled human review.

### Retrieval drift / RAG poisoning? → see [`06_security_and_guardrails.md`](06_security_and_guardrails.md)
