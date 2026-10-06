# Interview Preparation Kit: HR Document Assistant (Production RAG)

Use this folder to **explain this project end to end in a GenAI / Agentic AI interview** and to answer the questions in the
[Company-Specific Gen AI & Agentic AI Interview Question Bank](https://github.com/ajeetkumarAI/Company-Specific-Gen-AI-Agentic-AI-Interview-Question-Answer-Bank)
(Deloitte, EPAM, EY, Google, Infosys, PwC, TCS, Wipro, and more).

Every answer is tied to **real code, real settings and real measured numbers from this repository**, so you can say
*"in my project I did X, here's why, and here's the trade-off"* instead of reciting theory.

---

## How to use this folder

| Order | File | What you get | Time |
|---|---|---|---|
| 1 | [`01_project_pitch.md`](01_project_pitch.md) | 30-second, 2-minute and 5-minute project pitch, your contribution, challenges, failures (STAR stories) | 45 min |
| 2 | [`02_architecture_walkthrough.md`](02_architecture_walkthrough.md) | How to draw the architecture on a whiteboard, what happens on every query, where latency comes from (with real numbers) | 1 h |
| 3 | [`03_rag_deep_dive.md`](03_rag_deep_dive.md) | Chunking, embeddings, hybrid search, HNSW, reranking, metadata filtering, citations, hallucination, evaluation | 2 h |
| 4 | [`04_llm_fundamentals.md`](04_llm_fundamentals.md) | Tokens, Transformer, attention, temperature / top-p / top-k, prompts, structured output, fine-tuning vs RAG | 1 h |
| 5 | [`05_production_engineering.md`](05_production_engineering.md) | Retries, fallback, rate limiting, caching, async/GIL, FastAPI, streaming, observability, cost, scaling | 2 h |
| 6 | [`06_security_and_guardrails.md`](06_security_and_guardrails.md) | Prompt injection, RAG poisoning, access control, PII, secrets, OWASP LLM Top 10 | 45 min |
| 7 | [`07_agentic_ai.md`](07_agentic_ai.md) | Agents, tool calling, MCP, LangGraph, loops, human-in-the-loop, and how to extend this project into an agent | 1.5 h |
| 8 | [`08_scenario_and_behavioral.md`](08_scenario_and_behavioral.md) | "The client says…", "Production is slow…", disagreements, deadlines, explaining to non-technical people | 1 h |
| 9 | [`09_coding_round.md`](09_coding_round.md) | Retry decorator, LLM cache, top-k dedupe, JSON repair, chunker, pandas, SQL, DSA from the bank, with solutions | 1.5 h |
| 10 | [`10_question_bank_index.md`](10_question_bank_index.md) | **Every company's questions mapped to the answer in this folder** | reference |
| 11 | [`11_live_demo_script.md`](11_live_demo_script.md) | A 5-minute live demo that proves each feature | 20 min |
| ⭐ | [`cheat_sheet.md`](cheat_sheet.md) | One page of numbers and keywords to read right before the interview | 10 min |

### Suggested plans

- **1 day before:** `cheat_sheet` → `01` → `02` → `08` → practise the demo (`11`).
- **3 days:** Day 1: `01`, `02`, `03`. Day 2: `04`, `05`, `06`. Day 3: `07`, `08`, `09`, then the company index (`10`) for the company you're interviewing with.
- **Interviewing at a specific company:** open `10_question_bank_index.md`, find the company, and practise each question out loud using the linked answer.

### Rules for answering well

1. **Lead with the decision, then the reason, then the trade-off.** "I used hybrid search because HR queries mix meaning and exact terms like 'POSH'; the cost is one extra sparse index."
2. **Quote a number.** "Repeated questions drop from ~10 s to ~1 ms with 0 tokens." Numbers make it real.
3. **Name the file.** "That's in `src/utils/helpers.py`, `with_retry`." It shows you built it.
4. **Be honest about gaps.** This project is a production RAG system, **not** an agent, and it runs on OpenAI rather than a specific cloud. Each file shows how to answer those questions honestly and how you would extend the project.

---

## The project in one paragraph

An HR policy assistant: employees ask questions in a Streamlit chat; a FastAPI backend answers **only from company
policy documents**, with **citations**. Documents are chunked by heading, embedded with OpenAI `text-embedding-3-small`,
and stored in **Qdrant** with both dense (meaning) and BM25 sparse (keyword) vectors. Each question goes through
guardrails → cache → hybrid search with **metadata filters and role-based access** → **OpenAI reranking** →
`gpt-6-luna` generation with **retries and a fallback model** → citation validation. Everything is traced in
**LangSmith**, logged as JSON with request IDs, exposed as Prometheus metrics, and covered by **45 offline tests**.
