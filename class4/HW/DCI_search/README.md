# DCI_search — local PDF corpus search

A trimmed fork of **[DCI-Agent-Lite](https://github.com/DCI-Agent/DCI-Agent-Lite)** wired to search a local corpus of papers produced by the `PDF_extract` pipeline, using a local Ollama model instead of a hosted API.

Direct corpus interaction: no embeddings, no vector DB, no offline index. The agent gets `read` + `bash` (with `rg`, `find`, `jq`) and works directly against the per-paper sidecar files (`manifest.json`, `paper_native_images.md`, `tables_structured.md`, `figure_alt_text.md`).

## What was changed from upstream

- Removed all benchmark code: `scripts/{bcplus_eval,bright,qa}/`, the HuggingFace dataset downloaders, and the `export_{bc_plus,bright}_docs` modules.
- Trimmed `pyproject.toml` to runtime deps only.
- Trimmed `setup.sh` to install + Pi-build only; no dataset download.
- Added `prompts/pdf_corpus_prompt.txt` documenting the papers_corpus layout and `manifest.json` schema for the agent.
- Added `scripts/examples/pdf_search_ollama.sh` as the single working entry point.

The runtime (`src/dci/benchmark/pi_rpc_runner.py`) is unchanged.

## Setup

```bash
bash setup.sh
```

This installs `uv` and `ripgrep`, syncs Python deps, then clones and builds `pi-mono` (the underlying coding-agent harness).

Then:

1. `ollama pull qwen3.6:35b` (or whatever local tag you have)
2. Provider registration is handled by `extensions/ollama-provider.ts` in this repo, loaded automatically by the example script via pi's `-e` flag. To add a new local model, append it to that file's `models` array (the `id` must match the Ollama tag exactly).

## Usage

### Web UI (recommended)

```bash
uv run python scripts/regen_ollama_extension.py   # one-off, after `ollama pull`
uv run streamlit run app.py
```

Sidebar lets you pick a model (auto-detected from `ollama list`), toggle "Return figure" (default on — agent appends the most relevant figure inline), and see how much of the context window the last run used. Reset clears the chat.

### CLI

```bash
bash scripts/examples/pdf_search_ollama.sh
# or with your own question:
bash scripts/examples/pdf_search_ollama.sh "Which figures in li-2024 have empty captions?"
```

The corpus path is hard-coded to `/Users/ming/Desktop/Code/PDF_extract/papers_corpus` in both `app.py` and the CLI example — edit `CORPUS_DIR` if you move it.

Run artifacts are written under `outputs/runs/<timestamp>/` (`final.txt`, `conversation_full.json`, `events.jsonl`).

## License

MIT — inherited from upstream. See `LICENSE`.
