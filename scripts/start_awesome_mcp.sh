#!/usr/bin/env bash
set -euo pipefail

# Stop and remove existing containers
docker rm -f awesome-copilot || true
docker rm -f awesome-copilot-http || true

# Start container in HTTP mode detached
docker run -d --name awesome-copilot -p 8080:8080 ghcr.io/microsoft/mcp-dotnet-samples/awesome-copilot:latest --http

# wait and verify
sleep 1
curl -sS -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' http://localhost:8080/mcp || true

echo "awesome-copilot should be running at http://localhost:8080/mcp"
