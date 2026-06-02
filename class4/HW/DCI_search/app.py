"""Streamlit shell for DCI_search.

Local PDF corpus chat with optional figure return. One question per turn
(no multi-turn memory — each turn is an independent dci-agent-lite run).

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = Path("/Users/ming/Desktop/Code/PDF_extract/papers_corpus")
# Images live in the upstream PDF_extract pipeline output, not in papers_corpus.
# Use PyMuPDF native images (cleaner than MinerU raw). Manifest's
# native_img_path is relative to PDF_extract root, so prefix with EXTRACT_ROOT.
EXTRACT_ROOT = Path("/Users/ming/Desktop/Code/PDF_extract")
BASE_PROMPT = REPO_ROOT / "prompts" / "pdf_corpus_prompt.txt"
IMAGE_ADDENDUM = REPO_ROOT / "prompts" / "image_addendum.txt"
EXTENSION_TS = REPO_ROOT / "extensions" / "ollama-provider.ts"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
CONTEXT_WINDOW = 131072  # must match extensions/ollama-provider.ts
CHARS_PER_TOKEN = 3      # rough heuristic for mixed zh/en

IMAGE_LINE_RE = re.compile(r"^\s*\[IMAGE\]\s*:\s*(.+?)\s*$", re.MULTILINE)
CANDIDATE_LINE_RE = re.compile(
    r"^\s*\[CANDIDATE\]\s*:\s*(.+?)\s*\|\|\s*(.+?)\s*$", re.MULTILINE
)


# ----------------------------------------------------------------------------- helpers

def list_ollama_models() -> list[str]:
    """Return Ollama tags currently registered in the pi extension file."""
    text = EXTENSION_TS.read_text(encoding="utf-8")
    return re.findall(r'"id"\s*:\s*"([^"]+)"', text)


def regen_extension() -> tuple[bool, str]:
    """Call the regen script. Returns (ok, message)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "regen_ollama_extension.py")],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def build_combined_prompt(include_image_rule: bool) -> Path:
    """Write a combined prompt file under .cache/ and return its path."""
    cache_dir = REPO_ROOT / ".cache"
    cache_dir.mkdir(exist_ok=True)
    out = cache_dir / ("prompt_with_image.txt" if include_image_rule else "prompt_base.txt")
    parts = [BASE_PROMPT.read_text(encoding="utf-8")]
    if include_image_rule and IMAGE_ADDENDUM.exists():
        parts.append("\n\n" + IMAGE_ADDENDUM.read_text(encoding="utf-8"))
    out.write_text("".join(parts), encoding="utf-8")
    return out


def run_agent(question: str, model: str, return_image: bool) -> dict:
    """Run dci-agent-lite once, return parsed result."""
    prompt_file = build_combined_prompt(return_image)
    cmd = [
        "uv", "run", "dci-agent-lite",
        "--provider", "ollama",
        "--model", model,
        "--cwd", str(CORPUS_DIR),
        "--append-system-prompt-file", str(prompt_file),
        "--extra-arg=-e",
        f"--extra-arg={EXTENSION_TS}",
        "--extra-arg=--thinking",
        "--extra-arg=medium",
        question,
    ]
    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - start

    latest_run = None
    runs = sorted(RUNS_DIR.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        latest_run = runs[0]

    final = ""
    context_chars = 0
    if latest_run:
        final_file = latest_run / "final.txt"
        if final_file.exists():
            final = final_file.read_text(encoding="utf-8")
        conv_file = latest_run / "conversation_full.json"
        if conv_file.exists():
            try:
                conv = json.loads(conv_file.read_text(encoding="utf-8"))
                context_chars = _estimate_context_chars(conv)
            except json.JSONDecodeError:
                pass

    candidates, no_cands = _extract_candidates(final)
    image_path, cleaned = _extract_image(no_cands)
    return {
        "answer": cleaned,
        "image_path": image_path,
        "candidates": candidates,
        "elapsed": elapsed,
        "context_tokens": context_chars // CHARS_PER_TOKEN,
        "ok": proc.returncode == 0,
        "stderr": proc.stderr[-2000:] if proc.returncode != 0 else "",
        "run_dir": str(latest_run.relative_to(REPO_ROOT)) if latest_run else None,
    }


def _estimate_context_chars(conv: object) -> int:
    """Sum text length across whatever message-like structures appear in conversation_full.json."""
    total = 0

    def walk(node):
        nonlocal total
        if isinstance(node, str):
            total += len(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(conv)
    return total


def _extract_candidates(answer: str) -> tuple[list[tuple[str, str]], str]:
    """Pull out all `[CANDIDATE]: label || prompt` lines from the answer.

    Returns (list of (label, prompt) tuples, answer with those lines stripped).
    """
    matches = CANDIDATE_LINE_RE.findall(answer)
    if not matches:
        return [], answer
    cleaned = CANDIDATE_LINE_RE.sub("", answer).rstrip() + "\n"
    return [(label.strip(), prompt.strip()) for label, prompt in matches], cleaned


def _extract_image(answer: str) -> tuple[Optional[Path], str]:
    """Pull out the `[IMAGE]: <native_img_path>` line.

    The path comes verbatim from manifest.json's `native_img_path` field,
    which is relative to PDF_extract root (e.g. `outputs/pymupdf_native_images/
    chang-2023/p002_i001_xref7.jpeg`). Returns (absolute path or None, cleaned).
    """
    match = IMAGE_LINE_RE.search(answer)
    if not match:
        return None, answer
    raw = match.group(1).strip()
    cleaned = IMAGE_LINE_RE.sub("", answer).rstrip() + "\n"
    if raw.upper() == "NONE" or not raw:
        return None, cleaned

    p = Path(raw)
    cand = p if p.is_absolute() else (EXTRACT_ROOT / p)
    return (cand if cand.exists() else None), cleaned


# ----------------------------------------------------------------------------- UI

st.set_page_config(page_title="DCI Search", page_icon="🔍", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {q, a, img, elapsed, tokens, model, candidates}
if "last_tokens" not in st.session_state:
    st.session_state.last_tokens = 0
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 DCI Search")
    st.caption("Local PDF corpus chat")

    st.divider()
    st.markdown("**Model**")
    models = list_ollama_models()
    if not models:
        st.error("No Ollama models found in extension.\nRun the refresh below.")
        models = [""]
    selected_model = st.selectbox(
        "Model", models, label_visibility="collapsed",
        index=models.index("qwen3.6:35b") if "qwen3.6:35b" in models else 0,
    )
    if st.button("↻ Refresh model list", use_container_width=True):
        ok, msg = regen_extension()
        if ok:
            st.success("Refreshed. Re-select model above.")
        else:
            st.error(msg)
        st.rerun()
    st.caption(
        "Pulled from `ollama list` via `scripts/regen_ollama_extension.py`. "
        "Run that script (or click refresh) after `ollama pull`."
    )

    st.divider()
    st.markdown("**Behaviour**")
    return_image = st.checkbox("📷 Return figure", value=True)
    st.caption("Agent appends a relevant figure path; UI renders it inline.")

    st.divider()
    st.markdown("**Context (last run)**")
    pct = min(100, round(100 * st.session_state.last_tokens / CONTEXT_WINDOW))
    bar_color = "🟢" if pct < 80 else ("🟡" if pct < 95 else "🔴")
    st.progress(pct / 100)
    st.caption(f"{bar_color} {st.session_state.last_tokens:,} / {CONTEXT_WINDOW:,} tokens ({pct}%)")
    if pct >= 95:
        st.warning("Context near limit — long answers may truncate.")
    elif pct >= 80:
        st.info("Context above 80% — keep an eye on it.")

    if st.button("🗑  Reset chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_tokens = 0
        st.rerun()


# Main chat area ------------------------------------------------------------
st.markdown("## Ask the corpus")
st.caption(
    f"Corpus: `{CORPUS_DIR}` · "
    "Each question is an independent agent run — no cross-question memory."
)

for turn_idx, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        if turn.get("img"):
            st.image(turn["img"], caption=Path(turn["img"]).name, use_container_width=True)
        for cand_idx, (label, ask) in enumerate(turn.get("candidates") or []):
            if st.button(
                f"📷 {label}",
                key=f"cand-{turn_idx}-{cand_idx}",
                help=ask,
                use_container_width=True,
            ):
                st.session_state.pending_question = ask
                st.rerun()
        meta_parts = [f"⏱ {turn['elapsed']:.0f}s", f"🧠 {turn['model']}"]
        if turn.get("tokens"):
            meta_parts.append(f"📊 {turn['tokens']:,} tokens")
        st.caption(" · ".join(meta_parts))

prompt = st.session_state.pending_question or st.chat_input("Ask about the corpus…")
if st.session_state.pending_question:
    st.session_state.pending_question = None
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status("Searching corpus…", expanded=False) as status:
            result = run_agent(prompt, selected_model, return_image)
            if result["ok"]:
                status.update(label=f"Done in {result['elapsed']:.0f}s", state="complete")
            else:
                status.update(label="Failed — see error below", state="error")

        if not result["ok"]:
            st.error("Agent run failed.")
            with st.expander("stderr"):
                st.code(result["stderr"])
        else:
            st.markdown(result["answer"])
            if result["image_path"]:
                st.image(
                    str(result["image_path"]),
                    caption=Path(result["image_path"]).name,
                    use_container_width=True,
                )
            for cand_idx, (label, ask) in enumerate(result.get("candidates") or []):
                if st.button(
                    f"📷 {label}",
                    key=f"new-cand-{len(st.session_state.history)}-{cand_idx}",
                    help=ask,
                    use_container_width=True,
                ):
                    st.session_state.pending_question = ask
                    st.rerun()
            if return_image and not result["image_path"] and not result.get("candidates"):
                st.info("No figure suggested for this question.")
            meta_parts = [f"⏱ {result['elapsed']:.0f}s", f"🧠 {selected_model}"]
            if result["context_tokens"]:
                meta_parts.append(f"📊 {result['context_tokens']:,} tokens")
            st.caption(" · ".join(meta_parts))

        st.session_state.history.append({
            "q": prompt,
            "a": result["answer"],
            "img": str(result["image_path"]) if result["image_path"] else None,
            "candidates": result.get("candidates", []),
            "elapsed": result["elapsed"],
            "tokens": result["context_tokens"],
            "model": selected_model,
        })
        st.session_state.last_tokens = result["context_tokens"]
        st.rerun()
