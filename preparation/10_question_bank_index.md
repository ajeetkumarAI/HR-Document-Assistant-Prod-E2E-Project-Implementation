# 10: Question Bank Index (company → question → where to find the answer)

Source: [Company-Specific Gen AI & Agentic AI Interview Question Bank](https://github.com/ajeetkumarAI/Company-Specific-Gen-AI-Agentic-AI-Interview-Question-Answer-Bank).

**How to read this:** `03 › hybrid search` means *file `03_rag_deep_dive.md`, the question heading about hybrid search*.
🧩 = answer uses this project's code or numbers directly · 📖 = concept answer · 🎭 = scenario/behavioural answer.

| Code | File |
|---|---|
| 01 | [`01_project_pitch.md`](01_project_pitch.md) |
| 02 | [`02_architecture_walkthrough.md`](02_architecture_walkthrough.md) |
| 03 | [`03_rag_deep_dive.md`](03_rag_deep_dive.md) |
| 04 | [`04_llm_fundamentals.md`](04_llm_fundamentals.md) |
| 05 | [`05_production_engineering.md`](05_production_engineering.md) |
| 06 | [`06_security_and_guardrails.md`](06_security_and_guardrails.md) |
| 07 | [`07_agentic_ai.md`](07_agentic_ai.md) |
| 08 | [`08_scenario_and_behavioral.md`](08_scenario_and_behavioral.md) |
| 09 | [`09_coding_round.md`](09_coding_round.md) |

---

## Deloitte

| # | Question | Answer |
|---|---|---|
| 1 | GenAI project you worked on recently | 🧩 01 › 2-minute pitch |
| 2 | Why not send the whole PDF instead of RAG? | 🧩 02 › "Why not just send the whole PDF" |
| 3 | Chunk size and overlap | 🧩 03 › "What chunk size do you use" |
| 4 | Embeddings without the word "vector" | 📖 03 › "Explain embeddings without using the word vector" |
| 5 | Chatbot giving wrong answers: debugging approach | 🧩 03 › "Diagnose a chatbot…", 🎭 08 › first scenario |
| 6 | Can hallucinations be removed completely? | 📖 03 › "Why do LLMs hallucinate" |
| 7 | Everything behind a query + where max latency is | 🧩 02 › Query table + "Where is the maximum latency?" |
| 8 | Temperature 0 vs 1.2 | 📖 04 › Temperature |
| 9 | Data must never leave the enterprise | 🧩 05 › "Data must never leave the enterprise environment" |
| 10 | Corporate document chatbot for 10,000 employees, scale 10× | 🧩 05 › "Corporate chatbot for 10,000 employees" |
| 11 | Group a dict of nested tuples by tuple index | 🧩 09 › section 7 (`group_by_tuple_index`) |
| 12 | Execution sequence when an agent receives a query | 📖 07 › agent loop / ReAct + "How I would extend this project" |
| 13 | Text-to-SQL: prevent destructive or hallucinated SQL | 📖 07 › Text-to-SQL |
| 14 | Framework selection for agentic projects | 📖 07 › chain vs agent, LangGraph; choose by control, observability, team skills, client stack |
| 15 | Team lagging, deadline at risk | 🎭 08 › "Your delivery team is lagging" |

## EPAM

| # | Question | Answer |
|---|---|---|
| 1 | Manage incident data for fast, low-latency RAG retrieval | 🧩 03 › metadata filtering + HNSW; 05 › scaling (payload indexes, filters on date/severity) |
| 2 | Run 3 agents in parallel, isolate failures | 📖 07 › "Run 3 independent agents in parallel" |
| 3 | Retry with backoff for agents | 🧩 05 › retry; 07 › retry for agents; 09 › section 1 |
| 4 | Chain vs agent | 📖 07 › Chain vs agent |
| 5 | State in LangGraph | 📖 07 › LangGraph state |
| 6 | Why FastAPI; Pydantic | 🧩 05 › "Why FastAPI for GenAI backends" |
| 7 | What is MCP | 📖 07 › MCP |
| 8 | Temperature, top-p, top-k | 📖 04 › Top-p vs top-k |
| 9 | Words with multiple meanings ("bank") | 📖 04 › multiple meanings |
| 10 | Detect and handle hallucinations in agents | 📖 03 › reduce hallucination; 07 › observability |

## EY (both files)

| # | Question | Answer |
|---|---|---|
| 1 | RAG chatbot hallucinating; client upset: diagnose and fix today | 🧩 03 › "Diagnose a chatbot…", 🎭 08 › first scenario |
| 2 | 10k → 5M documents; latency and cost | 🧩 05 › "10,000 → 5,000,000 documents" |
| 3 | $2k → $18k/month | 🧩 05 › Cost |
| 4 | HR + Legal policy tool, 3,000 PDFs weekly, strict department access | 🧩 **06 › "Design HR + Legal policy search"** (this project *is* this system) |
| 5 | Overlapping Legal + HR policy without duplication | 🧩 06 › "relevant to both Legal and HR" |
| 6 | Client insists on 100% accuracy | 🎭 08 › "100% accuracy" |
| 7 | Scope changes mid-project | 🎭 08 › Scope changes |
| 8 | Technical disagreements with teammates | 🎭 08 › Technical disagreement |
| 9 | Staying up to date with AI | 🎭 08 › stay up to date |

## Infosys (3 files)

| # | Question | Answer |
|---|---|---|
| 1 | Tell me about yourself and current work | 🧩 01 › pitches |
| 2 | What is agentic AI in your application? | 📖 07 › opening note (be honest: RAG, not agent) + "How I would extend" |
| 3 | Who decides which tool to call; wrong tool? | 📖 07 › tool calling, wrong tool |
| 4–5 | MCP; function calling vs MCP | 📖 07 › MCP |
| 6 | Fine-tuning vs RAG; both? | 📖 03 › RAG vs fine-tuning |
| 7 | How do you know the app is getting better (offline vs production evaluation)? | 🧩 01 › "How did you evaluate it?", 03 › Evaluation |
| 8 | Async; 3 independent APIs: sequential or concurrent? | 📖 05 › "Why async" |
| 9 | Concurrency vs parallelism; GIL | 📖 05 › GIL |
| 10 | Agent with 10 tools calls 5 for a simple question | 📖 07 › "10 tools" |
| 11 | Trust the LLM to delete a customer record? | 📖 07 › "Should the LLM decide permissions?" |
| 12 | Project deep dive: problem, stack, deployment, audience, document types | 🧩 01 › 2-minute pitch; 02 › tech stack table |
| 13 | Document types and volume vs cloud cost and architecture | 🧩 05 › scaling + cost; 03 › tables and scanned PDFs |
| 14 | End-to-end pipeline, loaders, chunking | 🧩 02 › Ingestion table; 03 › chunking strategies |
| 15 | System vs user prompt | 🧩 04 › System prompt vs user prompt |
| 16–17 | Multithreading vs multiprocessing; GIL | 📖 05 › GIL |
| 18 | Why LLMs hallucinate; which parameters | 📖 03 › hallucination; 04 › temperature |
| 19 | Evaluate and monitor LLM/RAG in production | 🧩 05 › What do you monitor; 03 › Evaluation |
| 20 | Integrate a GenAI feature into a legacy microservice | 📖 05 › deployment: separate service behind a REST API (like this project's FastAPI + thin clients), feature flag, canary, fallback to old behaviour |
| 21 | Architect wants one giant prompt; you prefer RAG | 🎭 08 › "senior architect wants one giant prompt" |
| 22 | Git conflicts after a teammate merged | 🎭 08 › Git conflicts |
| 23 | Present hybrid RAG + semantic cache + reranking to engineers vs client | 🎭 08 › "Present your hybrid RAG…" |

## PwC (2 files)

| # | Question | Answer |
|---|---|---|
| 1 | POC vs production architecture | 🧩 05 › POC vs production table |
| 2 | Document ingestion + cost optimisation for PDFs | 🧩 02 › Ingestion (checksum skip, batching, embedding cache); 05 › Cost |
| 3 | Tokenization vs chunking | 📖 03 › Tokenization vs chunking |
| 4 | Vectors and tokens; AI without a vector DB? | 📖 03 › "without a vector database" |
| 5 | Retrieval process; role of top-k / top-p | 🧩 02 › Query table; 04 › top-k sampling vs retrieval top-k |
| 6–7 | RAG vs agentic AI; how agents decide | 📖 07 › concepts |
| 8 | Scoring LLM/RAG outputs; fine-tuning steps | 📖 03 › Evaluation; 04 › PEFT/LoRA (fine-tuning: data prep → base model → LoRA → train → evaluate on held-out set) |
| 9 | Sketch a production RAG chatbot end to end | 🧩 02 › "How to draw it in 60 seconds" + README diagrams |
| 10 | Draw an agentic workflow | 📖 07 › "How I would extend this project into an agent" |
| 11 | Why classic ML (loss, regression) still matters | 📖 Embeddings, rerankers and fine-tuning all optimise loss functions; evaluation uses precision/recall; many "AI" problems are better solved with classic ML |
| 12 | Detect outliers before embedding | 📖 Statistical (IQR, z-score) for numeric data; for text: very short/long docs, duplicates, embedding-space outliers (far from all clusters), garbage OCR |
| 13 | Pandas filter Marks < 30 → Fail | 🧩 09 › section 6 |

## TCS (2 files)

| # | Question | Answer |
|---|---|---|
| 1 | Architecture end to end | 🧩 02 |
| 2 | Why RAG over standalone LLM or fine-tuning | 📖 03 › RAG vs fine-tuning |
| 3 | Chunking strategy you used and why | 🧩 03 › chunking |
| 4 | Precision@K, Recall@K, MRR | 🧩 03 › Evaluation (`scripts/evaluate.py` computes hit@k and MRR) |
| 5 | System vs user prompt | 🧩 04 |
| 6 | Temperature, top-p, top-k | 📖 04 |
| 7 | Detect and reduce hallucinations | 🧩 03 › reduce hallucination |
| 8 | Agentic AI vs GenAI | 📖 07 |
| 9 | Agent with external tools | 📖 07 › "How I would extend" |
| 10 | Prevent infinite loops | 📖 07 › loops |
| 11–12 | Multithreading vs multiprocessing; GIL | 📖 05 › GIL |
| 13 | Debug high latency in RAG | 🧩 05 › "Debug a high-latency RAG pipeline"; 02 › latency table |
| 14 | Metrics to monitor | 🧩 05 › What do you monitor |
| 15 | Transformer vs RNN | 📖 04 |
| 16 | Decoder-only vs encoder-decoder | 📖 04 |
| 17 | Evaluating hallucination; metrics | 📖 03 › Evaluation (faithfulness/groundedness, LLM-as-judge) |
| 18 | Restricted vs unrestricted data in retrieval | 🧩 03 › restricted vs unrestricted; 06 |
| 19 | FAISS and PEFT | 📖 04 |
| 20 | Hosting an LLM for latency under traffic spikes | 📖 05 › scaling + real-time; self-hosted: vLLM with continuous batching, autoscaling GPU pods, quantisation, prompt caching |
| 21 | Decorators vs generators; streaming | 🧩 05 › generators; 09 › Python concepts |
| 22 | Large data without OOM | 🧩 05 › "Process large data without running out of memory" |
| 23 | Chunking strategies and parameters | 🧩 03 › chunking table |
| 24 | Function calling: when is it triggered | 📖 07 › tool calling |
| 25 | MCP and why it was introduced | 📖 07 › MCP |

## Wipro (2 files)

| # | Question | Answer |
|---|---|---|
| 1 | Queries diverge from document wording | 🧩 03 › "User queries diverge" |
| 2 | Correct chunks but wrong answer | 🧩 03 › "Retrieved chunks are correct but…" |
| 3 | Prompt injection | 🧩 06 |
| 4 | Long-term conversation memory without blowing the context | 🧩 07 › agent memory (this project: 6 turns, condense to a standalone question, history not resent) |
| 5, 7–9 | Retry decorator; 400/401?; thousands of users | 🧩 09 › section 1; 05 › retries at scale |
| 6, 10–12 | LLM cache; limits; cache everything? | 🧩 09 › section 2; 05 › Caching |
| 13–15 | Chunking with overlap; why overlap; fixed-size optimal? | 🧩 09 › section 5; 03 › chunking |
| 16–18 | Top 20 → top 5; near-duplicates | 🧩 09 › section 3 |
| 19–20 | Malformed JSON; keeps failing | 🧩 09 › section 4; 04 › structured output |

## Unknown company (MCP architecture)

| # | Question | Answer |
|---|---|---|
| 1–4 | 500k-row DB resource vs tool; 300 tools / 40k tokens; deletion safety gates; MCP server timeout | 📖 07 › "MCP production scenarios" |

---

## Google (Cloud AI Engineer: interview1 has 300 questions, plus 2 prescreening files)

Grouped by topic so you can study by block.

| Questions | Topic | Answer |
|---|---|---|
| 1–7, prescreen 1–10, 44, 61–68 | You, your project, contribution, challenge, evaluation, failure | 🧩 01 (pitches, contribution, STAR stories, evaluation) |
| 8–17, 111–117 | LLM, token, Transformer, attention, QKV, temperature, top-p, hallucination, pre-training, RLHF, foundation and multimodal models | 📖 04 |
| 18–30, 131–150 | RAG vs fine-tuning, RAG end to end, embeddings, cosine, chunking, hybrid, reranking, improving RAG, evaluation, query rewriting, multi-query, compression, metadata filtering, parent-child, tables, scanned PDFs, citations, unauthorized docs, poisoning, drift, no retrieval, grounded generation, measuring hallucination | 🧩 03 (+ 06 for poisoning and unauthorized documents) |
| 31–40, 151–165, 241–260, 291–299 | Agents, tool calling, LangGraph, loops, multi-agent, MCP, planning, state, HITL, observability, ReAct, planner/supervisor, memory, excessive agency, indirect injection, exfiltration | 📖 07 (+ 06 for security) |
| 41–54, 201–230 | Gemini, Vertex AI, ADK, Cloud Run vs GKE, IAM, Secret Manager, Cloud Storage vs Cloud SQL, BigQuery, Pub/Sub, Eventarc, VPC, Artifact Registry, grounding on GCP | See "Mapping this project to Google Cloud" below |
| 55–60, 186–190, 231–240 | Design: enterprise chatbot, RAG for 1M users, high traffic, latency, cost, model failure, document platform, 10M documents, async, idempotency, failed processing, **enterprise HR assistant**, multi-tenancy, banking assistant, real-time, streaming | 🧩 05 (scaling, cost, reliability) + 06 (multi-tenant isolation) + **this whole project for Q231** |
| 61–70, 251–260 | Prompt injection, securing RAG/agents, least privilege, HITL, OWASP | 🧩 06 |
| 71–80 | Overfitting, precision/recall, F1, data leakage, cross-validation, drift, batch vs online | 📖 see "Classic ML quick answers" below |
| 81–90, 271–290 | Python and DSA | 🧩 09 |
| 91–110, 191–200 | Customer scenarios and behavioural | 🎭 08 |
| 118–130 | Structured output, invalid output, context engineering, overflow, semantic and prompt caching, choosing and comparing LLMs, LLM-as-judge | 🧩 04 (+ 09 › section 4) |
| 166–185, 261–270 | MLOps/LLMOps, prompt versioning, canary, monitoring, token costs, evaluation pipeline, FastAPI, REST, rate limiting, SQL vs vector DB, pgvector, indexes, CI/CD, golden dataset, regression tests, shadow testing, rollback | 🧩 05 (+ 03 for golden dataset, + 03 › SQL vs vector DB) |
| 300 | Your overall AI engineering approach | 🧩 08 › "Why should we hire you?" + 01 › contribution |
| Prescreen 26–31, 52–60 | Clouds, languages, location, notice period, salary, offers | Personal: prepare your own honest answers |

### Mapping this project to Google Cloud (for GCP-specific questions)

| This project | On GCP |
|---|---|
| OpenAI `gpt-6-luna` / embeddings | **Gemini** via **Vertex AI** (`gemini-*` models, `text-embedding-*`), same RAG design; swap the client in `llm_client.py` / `embedder.py` |
| Docker image + FastAPI | **Cloud Run** (stateless, autoscaling, scale-to-zero); **GKE** if you need GPUs/self-hosted models or sidecars |
| Images | **Artifact Registry** |
| `.env` secrets | **Secret Manager**, mounted as env vars; never in code |
| API keys → roles | **IAM** + Identity-Aware Proxy / Identity Platform for users; **service accounts** with least privilege for the app |
| Qdrant | Qdrant on GKE/Cloud, or **Vertex AI Vector Search**, or **AlloyDB/Cloud SQL + pgvector** |
| Raw documents in `data/raw` | **Cloud Storage** bucket; upload event → **Eventarc / Pub/Sub** → ingestion worker on Cloud Run Jobs |
| Redis cache | **Memorystore** |
| Logs, metrics | **Cloud Logging / Monitoring** (JSON logs work as structured logs) |
| Private networking | **VPC**, Private Service Connect, VPC Service Controls (data never leaves the perimeter) |
| Analytics on usage and feedback | **BigQuery** |
| Grounding | Vertex AI grounding with your data store or Google Search; this project does its own grounding with citations |

**Cloud Run vs GKE:** Cloud Run = simplest, per-request scaling, no cluster to manage; limits on request time, no GPUs in all
regions, less control. GKE = full Kubernetes control (GPUs, daemons, custom networking), more operations work. **This API fits
Cloud Run.**

### Classic ML quick answers (Google 71–80)

| Question | Answer |
|---|---|
| Overfitting / reduce it | Model memorises training data, fails on new data. Use more data, regularisation, dropout, simpler model, early stopping, cross-validation |
| Precision vs recall; when is recall more important? | Precision = of predicted positives, how many are right; recall = of actual positives, how many we found. Recall matters when missing a case is costly (fraud, disease). **In RAG, retrieval favours recall (top 20), the reranker restores precision (best 5)** |
| F1 | Harmonic mean of precision and recall |
| Data leakage | Information from the test set or the future leaks into training; inflated scores |
| Cross-validation | Rotate train/validation splits (k-fold) for a robust estimate |
| Classification vs regression | Predict a category vs a number |
| Batch vs online inference | Precompute on a schedule vs answer per request (this API is online; ingestion is batch) |
| Model drift | Input or relationship changes over time → monitor and retrain; for RAG, monitor retrieval quality (see 06 › retrieval drift) |

---

## NLP real-time questions

| Questions | Answer |
|---|---|
| Cleaning, numbers, punctuation, unidecode, live string cleaning | 🧩 09 › section 8; 04 › Classic NLP table |
| Stemming vs lemmatization | 🧩 04 › Classic NLP (this project's BM25 stemmer) |
| BoW, TF-IDF, n-grams, term-document matrix, Word2Vec, CBOW, Skip-gram, GloVe, averaging vectors | 📖 04 › Classic NLP |
| RNN, LSTM, Transformer, self vs multi-head attention, positional embeddings, encoder vs decoder, masked decoder, autoregressive | 📖 04 › Model basics |
| Sentence Transformers; building RAG; NLP search system | 🧩 03 + 02 (this project is an NLP search system) |
| Stopwords: when to remove, inference time | 🧩 04 › Classic NLP |
| Class imbalance, improving metrics | 📖 04 › Classic NLP |
