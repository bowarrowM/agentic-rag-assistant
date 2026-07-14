#!/usr/bin/env bash
#
# One-command setup for the Kitty Pay Support Assistant.
# Builds the images, pulls the local LLM, ingests the docs, and starts everything.
#
set -euo pipefail

MODEL="llama3.2:3b"

echo "==> Building and starting containers..."
docker compose up -d --build

echo "==> Waiting for Ollama to be ready..."
until docker compose exec -T ollama ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "==> Pulling the model ($MODEL) — first run downloads ~2GB, please be patient..."
docker compose exec -T ollama ollama pull "$MODEL"

echo "==> Ingesting help-center documents into the vector store..."
docker compose exec -T backend python ingest.py

echo ""
echo "✅ Kitty Pay Support Assistant is running:"
echo "   Chat UI:   http://localhost:5173"
echo "   API docs:  http://localhost:8000/docs"
echo ""
echo "Stop:            docker compose down"
echo "Stop + wipe data (model + vectors): docker compose down -v"
