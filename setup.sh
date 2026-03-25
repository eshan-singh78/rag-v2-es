#!/bin/bash
set -e

# ─────────────────────────────────────────────
# RAG v2 — Ubuntu Server Setup Script
# Installs: Docker, Ollama, Python 3.11+, pgvector, pulls models, sets up venv
# Usage: bash setup.sh
# ─────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Must run as non-root (but with sudo access) ──────────────────────────────
if [ "$EUID" -eq 0 ]; then
  error "Do not run as root. Run as a regular user with sudo access."
fi

# ── Config ───────────────────────────────────────────────────────────────────
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
POSTGRES_DB="ragdb"
POSTGRES_PORT="5432"
OLLAMA_EMBED_MODEL="nomic-embed-text"
OLLAMA_LLM_MODEL="llama3.2:3b"
PYTHON_MIN="3.11"

# ── 1. System update ─────────────────────────────────────────────────────────
info "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq curl wget git ca-certificates gnupg lsb-release software-properties-common

# ── 2. Python ────────────────────────────────────────────────────────────────
info "Checking Python..."
PYTHON_OK=false
for cmd in python3.13 python3.12 python3.11; do
  if command -v $cmd &>/dev/null; then
    info "Found $cmd"
    PYTHON_CMD=$cmd
    PYTHON_OK=true
    break
  fi
done

if [ "$PYTHON_OK" = false ]; then
  info "Installing Python 3.12 via deadsnakes PPA..."
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
  PYTHON_CMD=python3.12
fi

$PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || true
info "Using $($PYTHON_CMD --version)"

# ── 3. Docker ────────────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
  info "Docker already installed: $(docker --version)"
else
  info "Installing Docker..."
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo usermod -aG docker "$USER"
  info "Docker installed. NOTE: log out and back in for group changes to take effect."
fi

# ── 4. pgvector via Docker ───────────────────────────────────────────────────
info "Starting pgvector container..."
if sudo docker ps -a --format '{{.Names}}' | grep -q "^pgvector$"; then
  warn "Container 'pgvector' already exists — starting it if stopped."
  sudo docker start pgvector 2>/dev/null || true
else
  sudo docker run -d \
    --name pgvector \
    --restart unless-stopped \
    -e POSTGRES_USER="$POSTGRES_USER" \
    -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
    -e POSTGRES_DB="$POSTGRES_DB" \
    -p "$POSTGRES_PORT":5432 \
    pgvector/pgvector:pg16
  info "pgvector container started."
fi

# Wait for Postgres to be ready
info "Waiting for Postgres to be ready..."
for i in $(seq 1 15); do
  if sudo docker exec pgvector pg_isready -U "$POSTGRES_USER" &>/dev/null; then
    info "Postgres is ready."
    break
  fi
  sleep 2
  if [ "$i" -eq 15 ]; then
    error "Postgres did not become ready in time."
  fi
done

# Enable pgvector extension
info "Enabling pgvector extension..."
sudo docker exec pgvector psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

# ── 5. Ollama ────────────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
  info "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
  info "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama service
if ! pgrep -x ollama &>/dev/null; then
  info "Starting Ollama service..."
  ollama serve &>/tmp/ollama.log &
  sleep 3
fi

# Pull models
info "Pulling embedding model: $OLLAMA_EMBED_MODEL..."
ollama pull "$OLLAMA_EMBED_MODEL"

info "Pulling LLM: $OLLAMA_LLM_MODEL..."
ollama pull "$OLLAMA_LLM_MODEL"

# ── 6. Python venv + dependencies ────────────────────────────────────────────
info "Setting up Python virtual environment..."
if [ ! -d "ragenv" ]; then
  $PYTHON_CMD -m venv ragenv
fi

source ragenv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
info "Python dependencies installed."

# ── 7. config.toml ───────────────────────────────────────────────────────────
if [ ! -f "config.toml" ]; then
  info "Creating config.toml from example..."
  cp config.toml.example config.toml
  # Patch in the configured values
  sed -i "s|url = .*|url = \"postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}\"|" config.toml
  sed -i "s|model = \"nomic-embed-text\"|model = \"${OLLAMA_EMBED_MODEL}\"|" config.toml
  sed -i "s|model = \"llama3.2:3b\"|model = \"${OLLAMA_LLM_MODEL}\"|" config.toml
  info "config.toml created."
else
  warn "config.toml already exists — skipping."
fi

# ── 8. Ingest documents ───────────────────────────────────────────────────────
if [ -d "data" ] && [ "$(ls -A data/*.pdf 2>/dev/null)" ]; then
  info "Ingesting documents from data/..."
  python populate_database.py
else
  warn "No PDFs found in data/ — skipping ingestion. Add PDFs and run: python populate_database.py"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✅ Setup complete.${NC}"
echo ""
echo "  Activate venv:     source ragenv/bin/activate"
echo "  Start bot:         python bot.py"
echo "  Single query:      python query_data.py \"your question\""
echo "  Re-ingest docs:    python populate_database.py --reset"
echo "  Run tests:         pytest test_rag.py -v"
echo ""
