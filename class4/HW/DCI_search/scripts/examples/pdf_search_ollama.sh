#!/usr/bin/env bash

set -euo pipefail

# Search the local PDF_extract papers_corpus with a local Ollama model.
#
# Prerequisites:
#   1. Ollama running locally: `ollama serve` (default at http://localhost:11434)
#   2. Model pulled:           `ollama pull qwen3.6:35b`
#
# Provider registration is handled by extensions/ollama-provider.ts, loaded
# via `pi -e <file>`. To add or tune models, edit that file.
#
# Override the Ollama endpoint with OLLAMA_BASE_URL if not localhost:11434.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
CORPUS_DIR="/Users/ming/Desktop/Code/PDF_extract/papers_corpus"

QUESTION="${1:-List every paper in this corpus and, for each, give the paper_id and the count of figures and tables recorded in manifest.json.}"

cd "$REPO_ROOT"
uv run dci-agent-lite \
  --provider ollama \
  --model "qwen3.6:35b" \
  --cwd "$CORPUS_DIR" \
  --append-system-prompt-file "$REPO_ROOT/prompts/pdf_corpus_prompt.txt" \
  --extra-arg="-e" \
  --extra-arg="$REPO_ROOT/extensions/ollama-provider.ts" \
  --extra-arg="--thinking" \
  --extra-arg="medium" \
  "$QUESTION"
