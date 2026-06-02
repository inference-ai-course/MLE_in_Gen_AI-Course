#!/usr/bin/env bash
set -euo pipefail

# DCI One-Click Setup (Unix/macOS)
# Usage: bash setup.sh

echo "==> Setting up DCI environment..."

# Load .env if present
if [ -f ".env" ]; then
    echo "==> Loading .env..."
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

# 1. Install uv if missing
if ! command -v uv &> /dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# 2. Install ripgrep if missing
if ! command -v rg &> /dev/null; then
    echo "==> Installing ripgrep..."
    if command -v brew &> /dev/null; then
        brew install ripgrep
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ripgrep
    elif command -v pacman &> /dev/null; then
        sudo pacman -S ripgrep
    else
        echo "WARN: Could not auto-install ripgrep. Please install manually: https://github.com/BurntSushi/ripgrep#installation"
    fi
fi

# 2b. Install jq if missing (used by the agent to query manifest.json)
if ! command -v jq &> /dev/null; then
    echo "==> Installing jq..."
    if command -v brew &> /dev/null; then
        brew install jq
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y jq
    elif command -v pacman &> /dev/null; then
        sudo pacman -S jq
    else
        echo "WARN: Could not auto-install jq. Please install manually: https://stedolan.github.io/jq/download/"
    fi
fi

# 3. Sync Python environment
echo "==> Syncing Python dependencies..."
uv sync

# 3b. Ensure Node >= 20 (pi-mono requires node >=20.0.0)
_node_major() { node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/'; }
if [ "$(_node_major)" -lt 20 ] 2>/dev/null; then
    echo "==> Node $(_node_major) < 20 detected. Installing Node 20 via nvm..."
    # Load nvm if available
    NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    # shellcheck disable=SC1091
    [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
    if command -v nvm &>/dev/null; then
        nvm install 20
        nvm use 20
    else
        echo "==> nvm not found. Installing nvm then Node 20..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
        # shellcheck disable=SC1091
        source "$NVM_DIR/nvm.sh"
        nvm install 20
        nvm use 20
    fi
    # Explicitly prepend Node 20 bin to PATH so all subsequent subprocesses use it
    _node20_bin="$(nvm which 20 2>/dev/null | xargs dirname)"
    if [ -n "$_node20_bin" ]; then
        export PATH="$_node20_bin:$PATH"
    fi
    echo "==> Now using Node $(node --version)"
fi

# 4. Clone and build Pi monorepo if CLI is not present
PI_CLI="pi-mono/packages/coding-agent/dist/cli.js"
if [ ! -f "$PI_CLI" ]; then
    if [ ! -d "pi-mono" ]; then
        echo "==> Cloning pi-mono..."
        git clone https://github.com/jdf-prog/pi-mono.git pi-mono
    fi
    cd pi-mono
    git checkout codex/context-management-ablation
    echo "==> Installing Pi dependencies (npm install)..."
    npm install
    echo "==> Building Pi (coding-agent and its deps only)..."
    (cd packages/tui && npm run build)
    (cd packages/ai && npm run build)
    (cd packages/agent && npm run build)
    (cd packages/coding-agent && npm run build)
    cd ..
else
    echo "==> Pi CLI already built, skipping."
fi

# 5. Check local Ollama (optional; only needed for the bundled example)
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "NOTE: 'ollama' CLI not found in PATH."
    echo "      Install from https://ollama.com if you plan to use the local-LLM example."
fi

echo ""
echo "==> Setup complete!"
echo "    Next steps:"
echo "    1. Pull the local model: ollama pull qwen3.6:35b"
echo "    2. Run a test:           bash scripts/examples/pdf_search_ollama.sh"
echo "    (Provider registration is handled by extensions/ollama-provider.ts;"
echo "     edit that file to add other local models.)"
