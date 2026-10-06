# 07: Agentic AI

> **Be honest in interviews:** this project is a **production RAG pipeline, not an agent**. The flow is fixed in code:
> guardrails → cache → retrieve → rerank → generate. The LLM never decides which step runs next. That's a strength for
> an HR assistant (predictable, auditable, cheap). Use this file to (1) explain agent concepts correctly and (2) describe
> **how you would extend this project into an agent**, which interviewers love.

---

## Concepts

### What is an AI agent? Agentic AI vs standard GenAI?
An agent is an LLM in a **loop** that decides which **tool** to call next based on observations, until a goal is met. Standard
GenAI / RAG is a **fixed pipeline** where code decides the steps. Agentic = the model plans and chooses actions.

### Chain vs agent; RAG vs agentic RAG
- **Chain:** predefined sequence (this project). Predictable, testable, cheap.
- **Agent:** dynamic sequence chosen by the LLM. Flexible, but slower, costlier and harder to test.
- **Agentic RAG:** the agent decides *whether* and *how* to retrieve: rewrite the query, search again with different filters, check
  another source, compare answers, then respond.

### When would you use an agent instead of a chain? When not?
Use an agent when the steps can't be known in advance (multi-source research, troubleshooting, tasks that need actions
like creating a ticket). Use a chain when the flow is known. **Don't** use multi-agent "because it sounds advanced": every
extra LLM decision adds latency, cost and failure modes.

### What is tool / function calling? When does the model trigger it?
You describe tools to the model (name, description, JSON schema of parameters). When the model decides a tool is needed, it
returns a structured tool call instead of text; **your code** executes it and sends the result back; the model continues.
The model triggers it based on the question + tool descriptions (and `tool_choice` settings).

### MCP (Model Context Protocol): what is it, why was it introduced, and how does it differ from function calling?
An open standard for connecting AI apps to tools and data. An **MCP server** exposes **tools** (actions), **resources** (readable data)
and **prompts** through one protocol; any MCP-compatible client can use them. **Function calling** is the model's ability to emit a
tool call; **MCP** standardises *how tools are discovered and invoked* across apps, so you write an integration once instead of
per framework. *Example for this project:* expose `search_hr_policies(question, filters)` as an MCP tool so Claude/Copilot-style
assistants can use the HR knowledge base.

### What is LangGraph? LangChain vs LangGraph? How is state managed?
- **LangChain:** building blocks (models, retrievers, prompts) and simple chains.
- **LangGraph:** builds agents as a **graph**: nodes (steps) + edges (including conditional ones) + a **shared
  typed state** (e.g., a `TypedDict` with messages, retrieved docs, retry count). Each node returns updates that are
  merged into the state (reducers, e.g., append to messages). **Checkpointers** persist state per thread → memory,
  resume and human-in-the-loop interrupts.

### Agent loop, ReAct, planner-executor, supervisor
- **ReAct:** Reason → Act (tool) → Observe → repeat.
- **Planner-executor:** one step makes a plan, an executor runs each step; re-plan on failure.
- **Supervisor / orchestrator:** a coordinator routes work to specialist agents and combines the results.

### Prevent infinite agent loops
Max iterations / recursion limit (LangGraph `recursion_limit`), token and time budgets, detecting a repeated identical tool call,
a "give up and answer with what you have" node, and alerts on runs that hit the limit.

### Agent with 10 tools calls 5 for a simple question. How do you control it?
Route first (a cheap classifier: does this even need tools?), expose only the relevant tools per step or intent, write clear tool
descriptions with "when NOT to use", cap tool calls per turn, set `tool_choice` where the step is known, and add few-shot examples.
Evaluate tool-selection accuracy on a golden set.

### The model selects the wrong tool / calls the right answer with the wrong API
Validate tool arguments against the schema, check pre-conditions in code, return clear errors so the model can correct itself, log tool
choice in traces, add failing cases to an evaluation set, improve descriptions or split tools, and restrict tools per user permission.

### Should the LLM decide permissions? Delete a customer record? Refund an order? Financial transactions?
**No.** Permissions live in code. For destructive or financial actions: **human-in-the-loop** approval (LangGraph `interrupt`),
dry-run / preview, limits (amount caps), idempotency keys, full audit logs, and the ability to roll back. The agent *proposes*; a human or
policy engine *approves*.

### Text-to-SQL agent: prevent destructive or hallucinated SQL
Read-only DB user (least privilege) · allow-list `SELECT` only and parse the SQL (sqlglot) to reject DDL/DML · a curated view / semantic
layer instead of raw tables · row limits and timeouts · validate table/column names against the schema · run `EXPLAIN` first · human
approval for anything else · log every query.

### Agent memory: short-term vs long-term, risks, stale information
Short-term = conversation state (this project: last 6 turns, 1 h TTL, condensed into a standalone question). Long-term = stored
facts/preferences (vector store or DB). Risks: privacy, stale or wrong facts persisting, injection stored in memory, cost. Mitigate with
TTLs, timestamps and source on each memory, re-verification against the source of truth, and user-visible / deletable memory.

### Agent observability: what would you log? Debugging wrong answers
Every step: input, chosen tool + arguments, tool output, tokens, latency, errors, final answer, with a trace ID (LangSmith shows this
as a tree). Debug by replaying the trace: was the plan wrong, the tool choice, the tool output, or the final synthesis?

### Single vs multi-agent; how agents communicate; disadvantages
Multi-agent splits responsibilities (researcher, writer, reviewer) and communicates through shared state or messages via an
orchestrator. Downsides: more LLM calls (cost, latency), harder debugging, error compounding, coordination failures. Start with
one agent and good tools.

### Run 3 independent agents in parallel and isolate failures (EPAM)
Run them concurrently (`asyncio.gather(..., return_exceptions=True)` or LangGraph parallel branches), each with its own
**timeout** and **retry with backoff**; a failed branch returns a typed error result instead of raising, so the aggregator continues with
partial results and reports what failed. Add a circuit breaker per downstream dependency.

### Retry with backoff for agent execution (EPAM)
Same policy as this project's `with_retry` (transient errors only, exponential backoff + jitter, max attempts), applied
**per tool call** (not the whole run, to avoid repeating side effects), with **idempotency keys** for actions so a retry never
double-executes (e.g., a refund).

### Excessive agency; least privilege for agents; data exfiltration
Excessive agency = an agent with more tools, permissions or autonomy than it needs. Give each tool the minimum scope, separate read and write
tools, require approval for writes, block outbound tools (email, HTTP) from sending retrieved confidential content (output
filtering / DLP), and never put secrets in the prompt.

### Governing 100 agents across an enterprise
Central registry (owner, purpose, tools, data access, model), a shared gateway (auth, rate limits, logging, cost attribution),
standard evaluation and red-teaming before release, policy-as-code for tool permissions, kill switches, and periodic access reviews.

### MCP production scenarios (from the bank)
- **500k-row DB: Resource or Tool?** A **Tool** with parameters (query/filter/limit). Resources are for reading small, addressable content; never
  dump 500k rows into context.
- **300 tools, 40k tokens of schemas:** dynamic tool selection (retrieve relevant tools per query with embeddings), group tools behind
  a router, shorten schemas, separate agents per domain.
- **MCP tool can delete data:** a separate write server with human approval, dry-run, scoped credentials, audit, and an allow-list.
- **MCP server times out mid-conversation:** timeouts + retries with backoff, circuit breaker, mark the tool unavailable, let the
  agent continue with other tools or tell the user, never hang the whole run.

---

## How I would extend this project into an agent (great closing answer)

> "Today it's a deterministic RAG pipeline, on purpose. The next step would be a **LangGraph agent** with this pipeline as
> one tool:
> 1. `search_hr_policies(question, filters)`: the existing hybrid search + rerank (read-only).
> 2. `get_leave_balance(employee_id)`: HRMS API, read-only, scoped to the signed-in user.
> 3. `create_hr_ticket(summary)`: a write action, so it needs **human confirmation** (LangGraph interrupt) and an idempotency key.
>
> The state would hold messages, retrieved docs and a tool-call count; a router node sends simple policy questions straight to the
> existing pipeline (fast, cheap) and only uses the agent loop for multi-step requests like *'how many sick days do I have left, and can I
> carry them forward?'*. A recursion limit, per-tool timeouts and retries, permissions in code (not the prompt) and LangSmith
> traces for every step complete it. I'd expose `search_hr_policies` as an **MCP tool** so other assistants could reuse it."
