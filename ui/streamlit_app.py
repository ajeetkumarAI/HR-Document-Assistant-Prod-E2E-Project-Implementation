"""Streamlit web UI for the HR Document Assistant.

This is a THIN CLIENT: it holds no RAG logic itself. Every action is an HTTP call to the
FastAPI backend, exactly like any other frontend (React app, Teams bot, ...) would make:

    Browser ──► Streamlit (this file, port 8501) ──HTTP──► FastAPI (port 8000) ──► Qdrant / OpenAI

Run (with the API already running in another terminal):
    streamlit run ui/streamlit_app.py

Pages
-----
* Chat       - ask questions, see streamed answers, citations, timings; give feedback
* Documents  - (hr_admin) upload policies with metadata, list / delete documents, re-index
* System     - health, readiness, cache controls
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import date
from typing import Any

import httpx
import streamlit as st

# =============================================================================
# Settings (env vars let docker-compose point the UI at the "api" container)
# =============================================================================
DEFAULT_API_URL = os.getenv("HR_API_URL", "http://localhost:8000")

# Demo keys matching .env.example -> API_KEYS. Each key maps to a role on the server.
# The SERVER decides what each role may see; the UI only chooses which key to send.
PRESET_KEYS = {
    "Employee (public docs)": os.getenv("HR_EMPLOYEE_KEY", "dev-employee-key"),
    "Manager (public + manager docs)": os.getenv("HR_MANAGER_KEY", "dev-manager-key"),
    "HR Admin (all docs + admin pages)": os.getenv("HR_ADMIN_KEY", "dev-admin-key"),
    "Custom key...": "",
}

CATEGORIES = ["", "leave", "workplace", "expenses", "conduct", "performance", "compensation", "holidays", "benefits"]

st.set_page_config(page_title="HR Document Assistant", page_icon="📘", layout="wide")


# =============================================================================
# Session state = memory that survives Streamlit re-runs (Streamlit re-runs the
# whole script on every click, so plain variables would be reset each time)
# =============================================================================
def _init_state() -> None:
    st.session_state.setdefault("messages", [])  # chat history shown on screen
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])  # server-side follow-up memory id


_init_state()


# =============================================================================
# HTTP helpers
# =============================================================================
def api_headers() -> dict[str, str]:
    return {"X-API-Key": st.session_state.get("api_key", "")}


def api_call(method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
    """Call the backend and return (status_code, json_or_text). Never raises - errors are shown in the UI."""
    try:
        with httpx.Client(base_url=st.session_state.api_url, timeout=120) as client:
            resp = client.request(method, path, headers=api_headers(), **kwargs)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, resp.text
    except httpx.HTTPError as exc:
        return 0, {"error": "connection_error", "message": f"Cannot reach API at {st.session_state.api_url}: {exc}"}


def show_error(status: int, body: Any) -> None:
    """Our API returns {"error", "message", "request_id"} for every error - show it nicely."""
    if isinstance(body, dict):
        msg = body.get("message") or body.get("detail") or body
        rid = body.get("request_id")
        st.error(f"**{status or 'No connection'}** - {msg}" + (f"  \n`request_id: {rid}`" if rid else ""))
    else:
        st.error(f"**{status}** - {body}")


def stream_answer(payload: dict[str, Any], sink: dict[str, Any]) -> Iterator[str]:
    """Read Server-Sent Events from /api/v1/query/stream.

    Yields answer text pieces (Streamlit's st.write_stream prints them as they arrive) and
    stores the non-text events in `sink`:
        sink["sources"] = citations sent before the answer
        sink["done"]    = the final full response (timings, model, run_id...)
        sink["error"]   = error event, if any
    SSE format:  "event: token\\ndata: {...}\\n\\n"
    """
    event_type, data_lines = None, []
    try:
        with (
            httpx.Client(base_url=st.session_state.api_url, timeout=120) as client,
            client.stream("POST", "/api/v1/query/stream", json=payload, headers=api_headers()) as resp,
        ):
            if resp.status_code != 200:
                resp.read()
                sink["error"] = (resp.status_code, resp.json() if resp.content else {})
                return
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif line == "" and event_type:  # blank line = end of one event
                    data = json.loads("\n".join(data_lines)) if data_lines else {}
                    if event_type == "token":
                        yield data.get("content", "")
                    elif event_type == "sources":
                        sink["sources"] = data.get("citations", [])
                    elif event_type == "done":
                        sink["done"] = data.get("response", {})
                    elif event_type == "error":
                        sink["error"] = (500, data)
                    event_type, data_lines = None, []
    except httpx.HTTPError as exc:
        sink["error"] = (0, {"message": f"Cannot reach API: {exc}"})


def render_details(resp: dict[str, Any], idx: int) -> None:
    """Citations, timing and model info under an assistant message."""
    citations = resp.get("citations") or []
    cache = resp.get("cache")
    meta = f"model `{resp.get('model') or '-'}`"
    if cache:
        meta += f" · cache **{cache}**"
    usage = resp.get("usage") or {}
    if usage.get("input_tokens"):
        meta += f" · tokens {usage.get('input_tokens')} in / {usage.get('output_tokens')} out"
    total = (resp.get("timings_ms") or {}).get("total")
    if total is not None:
        meta += f" · {total / 1000:.2f} s"
    st.caption(meta)

    if citations:
        with st.expander(f"📎 Sources ({len(citations)})", expanded=False):
            for c in citations:
                badge = "✅ cited" if c.get("cited") else "· provided"
                rerank = f" · rerank {c['rerank_score']:.0f}/10" if c.get("rerank_score") is not None else ""
                effective = c.get("effective_date") or "n/a"
                st.markdown(
                    f"**[{c['id']}] {c.get('title') or c.get('source')}** — {c.get('section') or ''}  \n"
                    f"<small>{c.get('source')} · effective {effective}{rerank} · {badge}</small>",
                    unsafe_allow_html=True,
                )
                st.text(c.get("snippet", ""))
    timings = resp.get("timings_ms")
    if timings:
        with st.expander("⏱ Timings (ms)", expanded=False):
            st.json(timings)

    # Feedback goes to LangSmith (only possible when tracing is on and the API returned a run_id)
    if resp.get("run_id"):
        score = st.feedback("thumbs", key=f"fb_{idx}")
        if score is not None and not st.session_state.get(f"fb_sent_{idx}"):
            code, body = api_call("POST", "/api/v1/feedback", json={"run_id": resp["run_id"], "score": float(score)})
            st.session_state[f"fb_sent_{idx}"] = True
            st.toast("Thanks for the feedback!" if code == 200 else "Feedback not sent")


# =============================================================================
# Sidebar: connection, identity, filters
# =============================================================================
with st.sidebar:
    st.title("📘 HR Assistant")

    st.session_state.api_url = st.text_input("API URL", value=st.session_state.get("api_url", DEFAULT_API_URL))

    who = st.selectbox("Sign in as", list(PRESET_KEYS), help="Each API key maps to a role on the server")
    if who == "Custom key...":
        st.session_state.api_key = st.text_input("API key", type="password")
    else:
        st.session_state.api_key = PRESET_KEYS[who]
    st.session_state.is_admin = who.startswith("HR Admin") or who == "Custom key..."

    # Live status of the backend
    code, ready = api_call("GET", "/ready")
    if code == 200:
        checks = ready.get("checks", {})
        st.success(f"API ready · {checks.get('indexed_chunks', '?')} chunks · reranker: {checks.get('reranker')}")
    else:
        st.error("API not reachable - start it with `python main.py`")

    st.divider()
    st.subheader("Search filters")
    f_category = st.selectbox("Category", CATEGORIES, format_func=lambda x: x or "(any)")
    f_department = st.text_input("Department", placeholder="e.g. hr, finance")
    use_date = st.checkbox("Only policies effective after…")
    f_after = st.date_input("Effective after", value=date(2026, 1, 1)) if use_date else None
    top_n = st.slider("Passages sent to the LLM", 1, 10, 5)
    use_cache = st.toggle("Use answer cache", value=True)
    streaming = st.toggle("Stream answer", value=True)

    st.divider()
    if st.button("🗑 New conversation", width="stretch"):
        # Forget on the server too, so follow-up context doesn't leak into the new chat
        api_call("DELETE", f"/api/v1/sessions/{st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()


def build_filters() -> dict[str, Any] | None:
    """Only send filters the user actually set (empty strings would filter out everything)."""
    f: dict[str, Any] = {}
    if f_category:
        f["category"] = f_category
    if f_department.strip():
        f["department"] = f_department.strip()
    if f_after:
        f["effective_after"] = f_after.isoformat()
    return f or None


# =============================================================================
# Pages
# =============================================================================
tab_chat, tab_docs, tab_sys = st.tabs(["💬 Chat", "📂 Documents (admin)", "⚙️ System"])

# ----------------------------------------------------------------------------- Chat
with tab_chat:
    st.header("Ask about HR policies")
    if not st.session_state.messages:
        st.info(
            "Try: *How many sick leaves do I get per year?* · *What is the hotel limit in Bengaluru?* · "
            "*Can I work from outside India?* · *What is the salary range for band B3?* (try as Employee vs HR Admin)"
        )

    # Re-draw the conversation so far
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("response"):
                render_details(msg["response"], i)

    question = st.chat_input("Type your question…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        payload = {
            "question": question,
            "filters": build_filters(),
            "session_id": st.session_state.session_id,
            "top_n": top_n,
            "use_cache": use_cache,
        }
        with st.chat_message("assistant"):
            if streaming:
                sink: dict[str, Any] = {}
                text = st.write_stream(stream_answer(payload, sink))  # prints tokens as they arrive
                if "error" in sink:
                    show_error(*sink["error"])
                    response = None
                else:
                    response = sink.get("done") or {}
                    text = response.get("answer", text)  # final text with invalid citations removed
            else:
                with st.spinner("Searching policies…"):
                    code, body = api_call("POST", "/api/v1/query", json=payload)
                if code == 200:
                    response, text = body, body["answer"]
                    st.markdown(text)
                else:
                    show_error(code, body)
                    response, text = None, ""
            if response:
                idx = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": text, "response": response})
                render_details(response, idx)

# ----------------------------------------------------------------------------- Documents
with tab_docs:
    st.header("Manage documents")
    if not st.session_state.is_admin:
        st.warning("Sign in as **HR Admin** in the sidebar to manage documents.")
    else:
        st.subheader("Upload")
        with st.form("upload", clear_on_submit=True):
            files = st.file_uploader(
                "Policy files", type=["pdf", "docx", "md", "txt", "html", "csv"], accept_multiple_files=True
            )
            c1, c2, c3 = st.columns(3)
            u_category = c1.selectbox("Category", CATEGORIES[1:])
            u_access = c2.selectbox("Access level", ["public", "manager", "confidential"])
            u_effective = c3.date_input("Effective date", value=date.today())
            c4, c5, c6 = st.columns(3)
            u_department = c4.text_input("Department", value="HR")
            u_doc_type = c5.selectbox("Document type", ["policy", "guideline", "calendar", "faq", "form"])
            u_tags = c6.text_input("Tags (comma separated)")
            u_force = st.checkbox("Re-index even if unchanged")
            submitted = st.form_submit_button("Upload & index", type="primary")
        if submitted:
            if not files:
                st.warning("Choose at least one file.")
            else:
                with st.spinner("Parsing, chunking and embedding…"):
                    code, body = api_call(
                        "POST",
                        "/api/v1/documents",
                        files=[("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files],
                        data={
                            "category": u_category,
                            "access_level": u_access,
                            "effective_date": u_effective.isoformat(),
                            "department": u_department,
                            "doc_type": u_doc_type,
                            "tags": u_tags,
                            "force": str(u_force).lower(),
                        },
                    )
                if code == 200:
                    st.success(f"Done: {body['summary']}")
                    st.dataframe(body["results"], width="stretch")
                else:
                    show_error(code, body)

        st.subheader("Indexed documents")
        col_a, col_b = st.columns([1, 1])
        if col_a.button("🔄 Re-index data/raw folder"):
            with st.spinner("Indexing…"):
                code, body = api_call("POST", "/api/v1/documents/reindex")
            st.success(body["summary"]) if code == 200 else show_error(code, body)

        code, docs = api_call("GET", "/api/v1/documents")
        if code == 200:
            if docs:
                columns = ("source", "title", "category", "access_level", "effective_date", "chunks")
                st.dataframe(
                    [{k: d.get(k) for k in columns} for d in docs],
                    width="stretch",
                    hide_index=True,
                )
                to_delete = col_b.selectbox(
                    "Delete document", [""] + [d["source"] for d in docs], format_func=lambda s: s or "(choose)"
                )
                if to_delete and col_b.button(f"Delete {to_delete}", type="secondary"):
                    doc_id = next(d["doc_id"] for d in docs if d["source"] == to_delete)
                    code, body = api_call("DELETE", f"/api/v1/documents/{doc_id}")
                    if code == 200:
                        st.success(f"Deleted {to_delete}")
                        st.rerun()
                    else:
                        show_error(code, body)
            else:
                st.info("No documents indexed yet. Upload some above or click Re-index.")
        else:
            show_error(code, docs)

# ----------------------------------------------------------------------------- System
with tab_sys:
    st.header("System")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("/health")
        st.json(api_call("GET", "/health")[1])
    with c2:
        st.subheader("/ready")
        st.json(api_call("GET", "/ready")[1])
    if st.session_state.is_admin and st.button("Clear answer cache"):
        code, body = api_call("DELETE", "/api/v1/cache")
        st.success(f"Cleared {body.get('cleared_entries', 0)} entries") if code == 200 else show_error(code, body)
    st.caption(f"Session id: `{st.session_state.session_id}` · API docs: {st.session_state.api_url}/docs")
