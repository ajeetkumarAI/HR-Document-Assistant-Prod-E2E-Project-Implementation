# 09: Coding Round

Every solution below is **tested**: run `python preparation/coding_solutions.py` and it prints `All checks passed`.
For each one, say **what the production version in this project does differently**. That's what impresses.

---

## 1. Retry with exponential backoff

*Wipro: "Write a Python function/decorator that retries an API call with exponential backoff." "Will it retry 400/401?" EPAM: "Design retry with backoff for agents."*

```python
class TransientError(Exception):
    """429 / 5xx / timeout -> worth retrying."""

class PermanentError(Exception):
    """400 / 401 / 404 -> retrying will fail the same way."""

def retry_with_backoff(
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError, TimeoutError, ConnectionError),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: exponential backoff + full jitter, retry ONLY transient errors."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on:
                    if attempt == max_attempts:
                        raise  # out of attempts -> surface the original error
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))  # 0.5, 1, 2, 4 ...
                    sleep(random.uniform(0, delay))  # full jitter: spreads out retry storms
            raise AssertionError("unreachable")

        return wrapper

    return decorator
```

**Talk track**
- Delays double (0.5 → 1 → 2 → 4 s, capped); **jitter** randomises them so thousands of clients don't retry in sync.
- **400/401/404 are not retried.** They'll fail the same way; retrying only delays the error (the self-check proves 1 call).
- `sleep` is injectable so the test runs instantly.
- **In this project:** `src/utils/helpers.py` uses `tenacity` with the same rules (`_is_retryable` checks OpenAI and httpx
  status codes), and the OpenAI SDK's own retries are disabled so attempts don't multiply.
- **At scale (thousands of users):** add a circuit breaker, respect `Retry-After`, put a global rate limiter in front of the
  provider, and switch to a fallback model (this project: `llm.fallback_model`).

---

## 2. In-memory cache for LLM queries

*Wipro: "Implement an in-memory cache to eliminate duplicate API calls." "Limitations of an in-memory dict?" "Would you cache every response?"*

```python
class LLMCache:
    """Thread-safe in-memory cache with TTL + LRU eviction for identical prompts."""

    def __init__(
        self, max_entries: int = 1000, ttl_seconds: float = 3600, clock: Callable[[], float] = time.time
    ) -> None:
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.max_entries, self.ttl, self.clock = max_entries, ttl_seconds, clock
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(model: str, prompt: str) -> str:
        normalised = re.sub(r"\s+", " ", prompt.strip().lower())
        return f"{model}::{normalised}"  # model in the key: different models -> different answers

    def get(self, model: str, prompt: str) -> str | None:
        k = self.key(model, prompt)
        with self._lock:
            item = self._data.get(k)
            if item is None or self.clock() - item[0] > self.ttl:
                self._data.pop(k, None)  # expired -> drop
                self.misses += 1
                return None
            self._data.move_to_end(k)  # mark as recently used
            self.hits += 1
            return item[1]

    def set(self, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._data[self.key(model, prompt)] = (self.clock(), response)
            self._data.move_to_end(self.key(model, prompt))
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # evict least recently used

    def get_or_call(self, model: str, prompt: str, call: Callable[[str], str]) -> str:
        cached = self.get(model, prompt)
        if cached is not None:
            return cached
        response = call(prompt)
        self.set(model, prompt, response)
        return response
```

**Talk track**
- Normalised key (case, whitespace) + **model in the key**.
- **TTL** (stale answers expire) + **LRU** (bounded memory) + **lock** (FastAPI serves requests on multiple threads).
- **Limits of a plain dict:** unbounded growth, not shared across workers, lost on restart, no invalidation → use Redis in
  production (this project: `cache.backend: redis`).
- **Don't cache everything:** not "I don't know" answers, not across users/roles with different permissions, invalidate when
  source documents change (this project: role + filters + corpus version in the key; semantic cache at cosine ≥ 0.95).

---

## 3. Top-k from 20 candidates, removing near-duplicates

*Wipro: "Vector DB returns top 20 by cosine; pass only the 5 most relevant. What if they're near-identical? Fix in production?"*

```python
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def top_k_unique(docs: list[dict[str, Any]], k: int = 5, dup_threshold: float = 0.95) -> list[dict[str, Any]]:
    """docs: [{"id", "score", "embedding"}]. Highest score first, skipping near-duplicates of already-chosen docs."""
    chosen: list[dict[str, Any]] = []
    for doc in sorted(docs, key=lambda d: d["score"], reverse=True):
        if all(cosine(doc["embedding"], c["embedding"]) < dup_threshold for c in chosen):
            chosen.append(doc)
        if len(chosen) == k:
            break
    return chosen

def mmr(query: list[float], docs: list[dict[str, Any]], k: int = 5, lam: float = 0.7) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance: balance relevance to the query against similarity to already-selected docs."""
    selected: list[dict[str, Any]] = []
    candidates = list(docs)
    while candidates and len(selected) < k:
        best = max(
            candidates,
            key=lambda d: (
                lam * cosine(query, d["embedding"])
                - (1 - lam) * max((cosine(d["embedding"], s["embedding"]) for s in selected), default=0.0)
            ),
        )
        selected.append(best)
        candidates.remove(best)
    return selected
```

**Talk track**
- `top_k_unique`: greedy, highest score first, skip anything ≥ 0.95 similar to an already-chosen chunk.
- `mmr`: **Maximal Marginal Relevance**. `lam` trades relevance against diversity (1.0 = pure relevance).
- Production: dedupe at ingestion too (this project: checksums + deterministic chunk ids), limit chunks per document,
  and rerank (this project reranks top 20 → best 5 with an LLM).

---

## 4. LLM returns malformed JSON

*Wipro: "Downstream service expects JSON; the model sometimes returns markdown, prefixes or broken JSON. What if it keeps failing?"*

```python
def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from messy LLM output: ```json fences, 'Sure! here it is:' prefixes, trailing text."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)) if start != -1 else []:
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError("no JSON object found")

def get_valid_json(
    call_llm: Callable[[str], str], prompt: str, required: set[str], max_attempts: int = 3
) -> dict[str, Any]:
    """Retry with the validation error fed back to the model; give up with a clear error."""
    feedback = ""
    for _ in range(max_attempts):
        raw = call_llm(prompt + feedback)
        try:
            data = extract_json(raw)
            missing = required - data.keys()
            if not missing:
                return data
            feedback = f"\n\nYour last reply was missing keys {sorted(missing)}. Return ONLY valid JSON."
        except (ValueError, json.JSONDecodeError) as exc:
            feedback = f"\n\nYour last reply was not valid JSON ({exc}). Return ONLY a JSON object."
    raise ValueError(f"LLM failed to return valid JSON after {max_attempts} attempts")
```

**Talk track**
1. **Prevent:** structured outputs / JSON-schema mode with `strict: true` (this project's LLM reranker does this).
2. **Repair:** strip code fences and prefixes, extract the first balanced `{...}`.
3. **Validate:** required keys / Pydantic model.
4. **Retry with feedback:** tell the model what was wrong.
5. **Give up safely:** after N attempts raise a clear error → fallback value, a different model, or human review. **Never pass
   invalid data downstream.** (This project's reranker fails open to search order.)

---

## 5. Chunking with overlap

*Wipro: "Split text into chunks; why is overlap necessary? Is fixed-character chunking optimal?"*

```python
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Word-based chunks with overlap; cut on sentence boundaries when possible."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    window: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if window and len(window) + len(words) > chunk_size:
            chunks.append(" ".join(window))
            window = window[-overlap:]  # carry the tail forward
        window.extend(words)
    if window:
        chunks.append(" ".join(window))
    return chunks
```

**Talk track:** overlap keeps a rule that spans a boundary complete in at least one chunk. Fixed-size cutting breaks
sentences and tables. **This project's** `src/chunking/chunker.py` goes further: it splits by **headings** first, measures in
**tokens**, recurses paragraph → line → sentence → word, merges tiny pieces, and prepends a context header before embedding.

---

## 6. Pandas: mark failing students

*PwC: "Given a DataFrame with 'Student' and 'Marks', mark students with marks < 30 as 'Fail' in a new 'Result' column."*

```python
def mark_failures(df: Any) -> Any:
    """Students with Marks < 30 -> Result = 'Fail' (others 'Pass')."""
    import numpy as np

    df = df.copy()
    df["Result"] = np.where(df["Marks"] < 30, "Fail", "Pass")
    return df
```

Alternative: `df.loc[df["Marks"] < 30, "Result"] = "Fail"` (the other rows stay NaN unless you set a default first). Prefer
vectorised operations (`np.where`, `.loc`) over `apply` or loops: they're much faster.

---

## 7. Python and DSA from the bank

```python
def group_by_tuple_index(data: dict[str, tuple[Any, ...]], index: int) -> dict[Any, list[str]]:
    """{'a': ('HR', 1), 'b': ('IT', 2), 'c': ('HR', 3)}, index 0 -> {'HR': ['a', 'c'], 'IT': ['b']}"""
    groups: dict[Any, list[str]] = defaultdict(list)
    for key, values in data.items():
        groups[values[index]].append(key)
    return dict(groups)

def second_largest_distinct(nums: list[int]) -> int | None:
    first = second = None
    for n in nums:
        if first is None or n > first:
            first, second = n, first
        elif n != first and (second is None or n > second):
            second = n
    return second

def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    seen: dict[int, int] = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return seen[target - n], i
        seen[n] = i
    return None

def char_frequency(s: str) -> dict[str, int]:
    return dict(Counter(s))

def binary_search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

| Problem | Complexity | Note |
|---|---|---|
| Group by tuple index (Deloitte) | O(n) | `defaultdict(list)` |
| Second-largest distinct (Google) | O(n), one pass | Handles duplicates; returns `None` if there isn't one |
| Two Sum | O(n) time, O(n) space | Hash map of value → index |
| Character frequency | O(n) | `collections.Counter` |
| Binary search | O(log n) | Needs a sorted list; searching an unsorted list is O(n) |

---

## 8. NLP live coding: clean a messy string

*NLP bank: "Clean this exact messy, real-world string end-to-end." "Could numbers contain useful information?"*

```python
def clean_text(text: str) -> str:
    """NLP live-coding: normalise messy real-world text but keep useful numbers and currency."""
    import html
    import unicodedata

    text = html.unescape(text)  # &amp; -> &
    text = unicodedata.normalize("NFKC", text)  # full-width / odd unicode -> standard
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " <EMAIL> ", text)  # mask emails
    text = text.lower()
    text = re.sub(r"[^a-z0-9₹$%.,\s<>]", " ", text)  # keep numbers, currency, basic punctuation
    return re.sub(r"\s+", " ", text).strip()
```

**Talk track:** decide per use case. **Keep numbers and currency** (in HR text "₹8,000" or "12 days" *is* the answer), mask PII
rather than delete it, and lower-case for keyword search but not for embeddings. This project's BM25 tokenizer keeps numbers and
applies the same normalisation to documents and queries.

---

## 9. SQL questions

```sql
-- Second-highest salary
SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees);

-- Nth highest with ties handled (window function)
SELECT salary FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk FROM employees
) t WHERE rnk = 2 LIMIT 1;

-- Find duplicate records
SELECT email, COUNT(*) FROM employees GROUP BY email HAVING COUNT(*) > 1;

-- INNER vs LEFT JOIN: INNER keeps only matching rows; LEFT keeps every left row (NULLs where there's no match)
SELECT e.name, d.name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id;

-- Running total per department
SELECT dept, month, cost, SUM(cost) OVER (PARTITION BY dept ORDER BY month) AS running_total FROM spend;
```

---

## 10. Python concepts (quick answers)

| Question | Answer |
|---|---|
| List vs tuple | List is mutable; tuple is immutable and hashable (can be a dict key) |
| Shallow vs deep copy | Shallow copies the outer object (nested objects shared); `copy.deepcopy` copies everything |
| Decorator | Wraps a function to add behaviour, e.g. `@retry_with_backoff` above, this project's `@traceable` |
| Generator / iterator | `yield` produces values lazily; used to **stream LLM tokens** (this project's `RAGPipeline.stream`) and to process big files without running out of memory |
| Context manager | `with` guarantees setup and cleanup, e.g. this project's `timer.stage("retrieve")` measures each stage |
| List comprehension vs loop | Comprehension is concise and usually faster for building lists; use a loop for side effects |
| Exception handling | Catch specific exceptions, re-raise with context (`raise X from exc`), never silently swallow errors. This project maps errors to HTTP codes in one place (`src/api/app.py`) |
| Hash table | Key → bucket via hash; O(1) average lookup (dict, set) |
| Stack vs queue | LIFO vs FIFO |
| BFS vs DFS | BFS goes level by level (shortest path in unweighted graphs); DFS goes deep first (cycle detection, exploring paths) |
| Sliding window | Move a window over a sequence keeping a running state, turning O(n·k) into O(n) |
| Recursion | A function calling itself with a base case; watch Python's recursion limit |
