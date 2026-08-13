#!/usr/bin/env bash
# Boots the AutoTuneLab backend (FastAPI :8000) and frontend (Vite :5173) together.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d backend/.venv ]; then
  echo "backend/.venv not found — run:  cd backend && uv venv --python 3.11 .venv && uv pip install -r requirements.txt --python .venv/bin/python"
  exit 1
fi

if [ ! -f backend/.env ]; then
  cp backend/.env.example backend/.env
  echo "Created backend/.env — add your TINKER_API_KEY there for cloud training + LLM-backed goal parsing/eval."
fi

if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi

cleanup() { kill 0; }
trap cleanup EXIT INT TERM

(cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload) &
(cd frontend && npm run dev -- --port 5173) &

wait
