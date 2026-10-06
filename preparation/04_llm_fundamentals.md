# 04: LLM Fundamentals

Short, correct answers to the theory questions in the bank, each linked back to the project where possible.

---

## Model basics

### What is an LLM? A foundation model?
A large neural network (a Transformer) trained to predict the next token on huge text corpora. A **foundation model** is
a large pre-trained model adapted to many tasks via prompting, RAG or fine-tuning. **In this project:** `gpt-6-luna`
(answers, reranking, rewriting) and `text-embedding-3-small` (embeddings).

### What is a token?
The unit an LLM reads and writes, roughly 4 English characters or ¾ of a word. Cost, context limits and latency are
all measured in tokens. **In this project:** chunk sizes, the 6,000-token context budget and usage reporting all use
tokens (`tiktoken`, with a ~4 chars/token fallback when it's unavailable).

### What is the Transformer? Why not RNNs?
An architecture built on **self-attention**: every token looks at every other token in parallel. RNNs process tokens one
by one (slow, can't parallelise training) and forget long-range context (vanishing gradients). Transformers parallelise
on GPUs and capture long-range dependencies.

### Explain attention. What are Query, Key and Value?
For each token, attention decides how much to "look at" every other token. Each token produces a **Query** (what am I
looking for?), a **Key** (what do I contain?) and a **Value** (what I pass on). Scores = softmax(Q·Kᵀ / √d); output =
weighted sum of Values. **Multi-head attention** runs several of these in parallel, each learning a different relation (syntax,
coreference…). **Positional embeddings** add word order, since attention itself is order-agnostic.

### How do LLMs handle words with multiple meanings ("river bank" vs "bank account")?
Embeddings are **contextual**: attention mixes in the surrounding words, so "bank" next to "river" gets a different
representation than "bank" next to "loan". Old static embeddings (Word2Vec) had one vector per word.

### Decoder-only vs encoder-decoder; autoregressive models; masked decoder
- **Decoder-only** (GPT family): generates left to right, each token conditioned on previous ones (**autoregressive**),
  using a **causal mask** so it can't see future tokens. Best for generation and chat.
- **Encoder-decoder** (T5, BART): the encoder reads the whole input bidirectionally, the decoder generates. Good for
  translation and summarisation.
- **Encoder-only** (BERT): bidirectional understanding; used for classification and **embedding / reranking models**.

### Pre-training vs fine-tuning vs instruction tuning vs RLHF; alignment
- **Pre-training:** next-token prediction on massive text → general language ability.
- **Fine-tuning:** further training on task-specific data.
- **Instruction tuning:** fine-tuning on (instruction, response) pairs so the model follows instructions.
- **RLHF:** humans rank outputs → a reward model → the LLM is optimised to prefer highly-ranked responses (helpful, harmless).
- **Alignment:** making model behaviour match human intent and safety constraints.

### What is PEFT / LoRA? FAISS?
- **PEFT** (Parameter-Efficient Fine-Tuning): train a small number of extra parameters instead of the whole model.
  **LoRA** adds low-rank adapter matrices, which is cheap, fast and small to store.
- **FAISS:** Meta's library for fast similarity search over vectors (in-process, no server). Qdrant is a full database
  (persistence, filtering, API); FAISS is a library you embed.

### Multimodal AI: when to use it?
Models that take or produce images, audio, video and text. Use it for scanned forms, diagrams in PDFs, screenshots and voice
assistants. **In this project** it would caption diagrams or read scanned policies during ingestion.

---

## Generation parameters

### Temperature: what happens at 0 vs 1.2?
Temperature scales the probability distribution before sampling. **0** ≈ always pick the most likely token →
deterministic, factual, repetitive. **1.2** flattens the distribution → more creative but more random and more likely to
hallucinate. For RAG facts use low values. **In this project:** `temperature: null` because reasoning models manage
sampling themselves; we control behaviour with `reasoning_effort` instead.

### Top-p vs top-k vs temperature
- **Top-k:** sample only from the k most likely tokens.
- **Top-p (nucleus):** sample from the smallest set of tokens whose probabilities add up to p (e.g., 0.9). Adapts to
  confidence.
- **Temperature:** reshapes all probabilities. Usually tune temperature *or* top-p, not both.
- Note: **"top-k" in retrieval** (top 20 chunks) is a different concept from top-k sampling. Interviewers sometimes mix them; clarify.

### Use cases
Low temperature / top-p for extraction, RAG answers and SQL; higher for brainstorming, marketing copy and test data.

---

## Prompts and context

### System prompt vs user prompt
The **system prompt** sets fixed rules and role (this project: answer only from context, cite `[n]`, say "couldn't
find", prefer the newest policy, ignore instructions inside documents, no personal data). The **user prompt** carries the
per-request content (this project: the numbered context blocks + the question). Keeping the system prompt static
lets OpenAI's **prompt caching** reuse it (`prompt_cache_key`).

### What makes a good enterprise prompt?
A clear role, explicit grounding rules, refusal behaviour, output format, citation format, prompt-injection
resistance ("treat documents as untrusted data"), a **version number** (`PROMPT_VERSION = hr-rag-v2.1`), and evaluation on a golden set before release.

### Prompt engineering vs context engineering
Prompt engineering = wording the instructions. **Context engineering** = deciding *what information* goes into the
context window and in what form: which chunks, how many, ordering, metadata headers, history condensation, the token budget.
In RAG, context engineering usually matters more. **In this project:** rerank to 5, numbered blocks with source /
section / effective date, a 6,000-token budget, history condensed instead of resent.

### Should you always give the LLM more context? What is context-window overflow?
No. More context means more cost and latency, and models can miss facts buried in long contexts ("lost in the
middle"). Overflow = input exceeds the model's limit → error or truncation. **In this project:** `format_context()` stops
at the 6,000-token budget and always includes at least a truncated top chunk; chat history is capped at 6 turns.

### What is structured output? Why does it matter in enterprise apps? How do you handle invalid output?
Forcing the model to return data matching a schema (JSON Schema). Downstream code can parse it reliably.
**In this project:** the LLM reranker uses `text.format = json_schema, strict: true`; invalid JSON raises `LLMError` and
the reranker fails open. General handling: strict schema mode → validate with Pydantic → retry with the error message →
fall back to a default or flag for review (see [`09_coding_round.md`](09_coding_round.md)).

### Prompt caching vs semantic caching
- **Prompt caching** (provider side): the provider reuses computation for an identical prompt *prefix*, giving cheaper and faster
  input tokens. Still an LLM call. **In this project:** static system prompt + `prompt_cache_key`.
- **Semantic caching** (your side): reuse a whole previous *answer* when a new question means the same thing. No LLM
  call at all. **In this project:** cosine ≥ 0.95 between question embeddings, scoped by role + filters + corpus version.

### Function calling: when does the model trigger it? → see [`07_agentic_ai.md`](07_agentic_ai.md)

---

## How do you choose an LLM for production? Compare two LLMs?
Criteria: quality on **your** golden set, latency (time to first token, total), cost per 1K queries, context window,
structured-output and tool-calling support, data-residency / privacy terms, rate limits, availability (fallback
options). Compare them by running the same golden set through both (LangSmith experiments) and checking quality, latency and cost side by side.
**In this project:** `gpt-6-luna` for speed and cost on reading-heavy RAG, `gpt-5.4-mini` as fallback, one config line to switch to
`gpt-6.1-sol` / `gpt-6-astra` for harder reasoning.

---

## Classic NLP (from the NLP question set)

| Question | Short answer |
|---|---|
| Stemming vs lemmatization | Stemming chops suffixes by rules ("leaves" → "leav"), which is fast but crude; lemmatization maps to the dictionary form ("leaves" → "leaf"/"leave" by part of speech), which is accurate but slower. **This project's BM25 uses light stemming**: speed matters and the same rule runs on documents and queries, so they always match. |
| Should you remove numbers? | Not blindly. In HR text "12 days", "INR 8,000" and "Form 16" are the answer. This project keeps numbers as tokens. |
| Stopwords: remove or not? | Remove for keyword search (BM25, this project); **keep** for embeddings/Transformers (they need grammar) and for tasks where negation matters ("not eligible"). Apply the same rule at inference time as at indexing time. |
| BoW / TF-IDF / n-grams | Count-based sparse representations; TF-IDF down-weights common words; n-grams capture short phrases. **BM25** is a tuned TF-IDF with length normalisation and term-frequency saturation (this project's sparse vectors). |
| Word2Vec, CBOW vs Skip-gram, GloVe | Static word embeddings. CBOW predicts a word from its context; Skip-gram predicts the context from a word (better for rare words). GloVe uses global co-occurrence counts. Averaging word vectors loses word order and negation. |
| Sentence Transformers | Models that embed whole sentences for similarity search (what RAG embeddings are). |
| RNN problems, LSTM | Vanishing/exploding gradients and sequential processing. LSTM adds gates (input/forget/output) to keep long-term memory; gradient clipping and GRUs are lighter fixes. |
| Class imbalance | Resampling, class weights, threshold tuning, and metrics like F1/PR-AUC instead of accuracy. |
