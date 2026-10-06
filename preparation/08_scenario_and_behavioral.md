# 08: Scenario and Behavioural Questions

Structure every answer the same way: **clarify → diagnose with data → act → prevent recurrence**. For behavioural questions use
**STAR** (Situation, Task, Action, Result) and end with what you learned. Real stories from this project are in
[`01_project_pitch.md`](01_project_pitch.md#star-stories-real-issues-found-and-fixed-in-this-project).

---

## Client and production scenarios

### "Your chatbot gives wrong answers / hallucinates. The client is escalating."
1. **Contain:** acknowledge it, collect failing examples, and if it's severe, temporarily raise the refusal threshold or add a disclaimer.
2. **Diagnose per example** using the trace (`request_id` → LangSmith): was the right chunk **retrieved**? Was it in the
   **prompt**? Did the model **ignore** it or invent details? Was the source document itself outdated?
3. **Fix the layer that failed:** retrieval (chunking, hybrid, filters, rerank), the prompt (grounding rules, refusal), the model,
   or the data (re-ingest the corrected policy, which also invalidates the cache automatically).
4. **Prevent:** add every failing case to the golden set; run it before each release; monitor 👎 feedback and "no answer" rate.

### "The client wants 100% accuracy / it must never make a mistake / 99.9% from an LLM." (explain without jargon)
> "No system, human or AI, is right 100% of the time, and I won't promise that. What I can promise is that it's
> **right in a measurable way and safe when it's unsure**: every answer shows the exact policy paragraph it came from, so
> anyone can check it in one click; when it can't find the answer it says so and points to HR instead of guessing; we test it
> on a fixed set of real questions before every change and publish the score; and for high-stakes topics we can require a human
> to confirm. It's like a new HR assistant who always shows you the handbook page and says 'let me check with HR' when unsure."

### "Accuracy is only 70%. What do you do?"
Define accuracy first: on which questions, measured how? Build or extend the golden set, then break errors down by type:
retrieval miss, wrong version, generation error, unanswerable question. Fix the biggest bucket first, usually retrieval
(chunking, hybrid search, reranking, metadata). Re-measure after each change, one change at a time.

### "Production suddenly became slow" / "Responses got worse after deployment"
→ [`05_production_engineering.md`](05_production_engineering.md#observability) (per-stage timings, what changed, rollback).

### "Costs went from $2k to $18k/month" → [`05_production_engineering.md`](05_production_engineering.md#cost)

### "Customer wants to use an LLM for everything" / "wants multi-agent because it sounds advanced"
> "Let's use the simplest thing that meets the requirement. Some steps are better as normal code or SQL: cheaper,
> deterministic, testable. In my HR assistant only answering and reranking use the LLM; access control, caching and
> filtering are plain code. I'd start with one pipeline, measure, and add agents only where steps can't be predicted."

### "Customer wants an AI agent to make financial transactions / refunds automatically"
Yes with guardrails: the agent **proposes**, a human or policy engine **approves** above a limit; least-privilege tools,
amount caps, idempotency keys, a full audit trail, reversibility, and a pilot on low-risk cases first. → [`07_agentic_ai.md`](07_agentic_ai.md)

### "Data must never leave the enterprise environment" → [`05_production_engineering.md`](05_production_engineering.md#data-must-never-leave-the-enterprise-environment)

### "A customer disagrees with your architecture" / "Two teams disagree about the architecture"
Understand their constraints first (cost, skills, compliance, timeline). Make the comparison concrete: a small spike or benchmark
on **their** data (quality, latency, cost), written trade-offs, and a decision record. Commit to the decision once it's made, and agree
a checkpoint to revisit with data.

### "Your senior architect wants one giant prompt with a huge context window; you prefer modular RAG" (Infosys)
> "I'd respect the experience and test it rather than argue. I'd take 30 real questions and run both: the long-context
> prompt and the RAG pipeline. Then compare accuracy, cost per query and latency. In my HR assistant RAG sends ~700 input tokens
> per question with citations; a full-document prompt would send tens of thousands every time. If long context wins on quality,
> we could still use it for rare, complex questions and RAG for the common ones. The data decides, not opinions."

### "Customer asks for a feature the platform doesn't support"
Clarify the underlying need, offer the closest supported option or a workaround, be transparent about limits, and log a feature
request with the business case. Don't promise unsupported features.

### "Customer wants an AI solution in 2 weeks; you estimate 6"
Explain what drives the 6 weeks, then offer options: a **narrower scope in 2 weeks** (e.g., one document set, no access
control, internal pilot), a phased plan, or more resources. Never silently cut testing or security.

### "Explain RAG to a CEO" / "an AI agent to a non-technical customer"
- **RAG:** "an open-book exam: the AI looks up the right page of *our* handbook first, then answers and shows the page."
- **Agent:** "a junior assistant who can use tools: it can look things up, check your leave balance and raise a ticket. It
  decides the steps itself, but asks for approval before anything important."

### Present your hybrid RAG with semantic cache and reranking to (A) engineers and (B) a non-technical client (Infosys)
- **Engineers:** architecture diagram, why hybrid (BM25 for acronyms), RRF, rerank 20→5, cache keys scoped by role +
  corpus version, metrics (cache hit rate, p95 per stage), trade-offs and what's next.
- **Client:** "Answers come from your policies, with the source shown. Repeated questions answer instantly and cost nothing.
  Employees only ever see documents they're allowed to see. Here's the accuracy score on 50 real questions."

---

## Team and delivery

### Your delivery team is lagging and risking a deadline (Deloitte)
Find the real blockers (1:1s, board review), re-plan by priority (must / should / could), remove blockers or reassign work, raise the risk
**early** with stakeholders along with options (scope, date, resources), and track daily until back on plan. Afterwards: retrospective.

### Scope changes mid-project
Assess the impact (effort, risk, timeline), present the trade-off, get agreement on what moves out or what date changes, and record it.
Small changes go into the backlog; large ones need re-planning.

### Teammate merged first; your branch has complex Git conflicts (Infosys)
```bash
git fetch origin
git rebase origin/main          # or: git merge origin/main
# resolve each conflict: understand BOTH changes, don't just pick "mine"
pytest && ruff check .          # make sure the combined code still works
git rebase --continue
git push --force-with-lease     # safe force-push: refuses if someone else pushed meanwhile
```
Talk to the teammate about the overlapping files; prevent it next time with smaller PRs and frequent rebasing.

### Technical disagreement with a teammate / your manager
Disagree with data, not opinions: propose a quick experiment, write down the trade-offs, listen for constraints you don't know about.
If the decision goes the other way, commit fully and document the risk.

### How do you stay up to date with AI?
Official release notes and docs (model providers, LangChain/LangGraph, Qdrant), papers on arXiv for topics I use,
hands-on rebuilding (this project: hybrid search, LLM reranking, semantic cache), and teaching or writing about it, which
forces deep understanding.

### How do you learn a new technology quickly?
Read the official quickstart → build a tiny end-to-end example → apply it to a real problem → read the deeper docs for the parts I
use → write tests. Example: Qdrant's sparse vectors with server-side IDF and RRF fusion, learned while building this project.

### What do you do when you don't know something?
Say so, explain how I'd find out, and follow up. Guessing in front of a client is worse than "let me verify and get back to you
today".

### Ambiguous requirements
Ask what decision the output supports and who uses it, write down assumptions, build a thin slice quickly and show it, then
iterate. Example: "HR assistant" became concrete once we defined roles and which documents each could see.

---

## Classic behavioural prompts → which story to use

| Question | Story from `01_project_pitch.md` |
|---|---|
| Biggest technical challenge | Story 2 (cache inside conversations: 10 s → 1 ms) |
| A failure / mistake you made | Story 1 (stemmer bug I introduced) or Story 4 (Python version) |
| Difficult problem you solved | Story 2 or Story 3 (endless reload loop) |
| Trade-off decision | Story 5 (keep LLM reranker, fail-open, options to speed up) |
| Took ownership beyond your role | Wrote tests, evaluation and docs (README, architecture, interview kit) beyond the features |
| Negative feedback | Example: "answers were slow and the token display was confusing". I measured, fixed, and showed the before/after numbers |
| Worked with ambiguity | Defining access levels and roles for HR documents |
| Strength | Turning prototypes into reliable, measurable systems (retries, caching, tracing, tests) |
| Area of improvement | Pick something real and show the action, e.g. "deeper Kubernetes; I'm deploying this project to a cluster" |

## "Why should we hire you?"
> "I don't stop at a working demo. I build what makes GenAI usable in a company: retrieval that respects permissions,
> reliability when the LLM provider fails, caching that cuts cost, tracing to debug any answer, and tests to keep it correct.
> My HR assistant shows all of that end to end, and I can explain every trade-off in it."
