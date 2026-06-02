# AGENTS.md — DCI_search

This repo is a thin search interface over the corpus produced by the
**`PDF_extract`** pipeline (papers_corpus/{paper_id}/manifest.json + sidecars).
DCI_search owns nothing about how that corpus is built; for the upstream
contracts (manifest schema, figure_id conventions, generalization rules G1–G14),
read `PDF_extract/AGENTS.md`.

What lives here:

- `src/dci/benchmark/pi_rpc_runner.py` — the runtime (unchanged from upstream).
- `src/dci/benchmark/pi_system_prompt.py` — prints pi's default system prompt.
- `prompts/pdf_corpus_prompt.txt` — the corpus-specific system prompt appended
  at run time. Update this when the upstream `manifest.json` schema changes.
- `scripts/examples/pdf_search_ollama.sh` — the single working entry point.

If `prompts/pdf_corpus_prompt.txt` drifts from the real schema in
`PDF_extract/papers_corpus/*/manifest.json`, the agent will give plausible but
wrong answers (e.g. searching on a renamed field). Treat the prompt as an
artifact contract with the upstream pipeline.

---

## How to run

First-time setup (clones pi-mono, builds it, installs uv/rg/jq deps):

```bash
bash setup.sh
ollama pull qwen3.6:35b      # or any tag you want exposed
```

Sync the pi Ollama extension with what's installed locally — re-run
whenever you `ollama pull` or `ollama rm`:

```bash
uv run python scripts/regen_ollama_extension.py
```

Web UI (recommended):

```bash
uv run streamlit run app.py
# open http://localhost:8501
```

CLI one-shot:

```bash
bash scripts/examples/pdf_search_ollama.sh "your question in quotes"
```

Inspect the most recent run (final answer, full conversation, tool calls):

```bash
RUN=$(ls -1dt outputs/runs/*/ | head -1)
cat "$RUN/final.txt"
cat "$RUN/conversation_full.json" | jq
```

Paths assumed by `app.py` and `scripts/examples/pdf_search_ollama.sh`
(edit in-file if you move things):
- Corpus:  `/Users/ming/Desktop/Code/PDF_extract/papers_corpus`
- Images:  `/Users/ming/Desktop/Code/PDF_extract/outputs/pymupdf_native_images`

---

## Lessons from local-model tuning (2026-05-25)

Tuned against `qwen3.6:35b` on Ollama. Tested 16 questions across the 4-paper
corpus (stations counts, sampling rates, reference-station selection, regression
formulas, image lookups, cross-paper, numerical extraction). Final score after
fixes: 9/9 consecutive correct.

Failure modes observed in baseline (no prompt rules, no thinking):
- **Arithmetic hallucination** — agent retrieved correct evidence (G01-G05, A01-A07)
  but wrote "5 + 7 = 11". Local models silently fumble small sums.
- **Question reinterpretation** — short Chinese technical questions like
  "採樣率是多少？" were re-interpreted as "tell me about this paper",
  triggering 90+ tool calls dumping the whole manifest.

Three prompt rules in `prompts/pdf_corpus_prompt.txt` resolved both, *and*
made successful runs 30-70% faster by capping over-exploration:

1. Answer the LITERAL question (no overview dumps for factual questions)
2. Write arithmetic out (`G01-G05 = 5; A01-A07 = 7; 5 + 7 = 12`)
3. Stop exploring after ~20 tool calls and answer with what you have

Combined with `--thinking medium` (passed via `--extra-arg` in the example
script). Going higher (`--thinking high`) was not measured here — try if a
specific question class fails.

**Do not downgrade the model.** A lighter Qwen (≤14B) runs 3x faster per
token but its tool-call reliability drops sharply on multi-turn agentic
tasks. Wall-clock wins from smaller models are usually erased by retries,
off-topic dumps, and hallucinated tool outputs.

**Image handling.** The agent locates the correct `figure_id` and exact
`images/<hash>.jpg` path inside the corpus. It cannot view image contents
(text-only model). Open the returned path with `open <path>` (macOS) or
swap in a vision model (qwen2.5vl, llava, gemma3) when actual figure
comprehension is needed.

---

## Coding Guidelines

These apply to all agents (Claude, Codex, etc.) working in this repo.

**1. Think before coding.** State assumptions explicitly. If multiple interpretations
exist, present them — don't pick silently. If something is unclear, stop and ask
before implementing.

**2. Simplicity first.** Minimum code that solves the problem. No speculative features,
no abstractions for single-use code, no configurability that wasn't requested.
If you write 200 lines and it could be 50, rewrite it.

**3. Surgical changes.** Touch only what the task requires. Don't improve adjacent
code, refactor things that aren't broken, or delete pre-existing dead code unless
asked. Every changed line should trace directly to the request.

**4. Goal-driven execution.** For multi-step tasks, state a plan with verifiable
checks before starting:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```
Strong success criteria let you loop independently without constant clarification.
