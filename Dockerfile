# Champions Stack Dockerfile - Hybrid uv + Bun
# Multi-stage build combining the fastest tools in the industry

# ============================================================================
# Stage 1: Frontend Build with Bun 🥟
# ============================================================================
FROM oven/bun:1.3.8-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/bun.lockb* ./

# Install dependencies with Bun (faster than npm)
RUN bun install

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN bun run build

# ============================================================================
# Stage 2: Backend Setup with uv 🐍
# ============================================================================
FROM python:3.12-bookworm AS backend-builder

# Install uv
RUN pip install uv

WORKDIR /app

# Copy Python project files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv (faster than pip)
RUN uv sync

# Copy backend source
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/

# ============================================================================
# Stage 3: Production Runtime 🚀
# ============================================================================
FROM python:3.12-slim-bookworm AS production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv and virtual environment from backend builder
COPY --from=backend-builder /app/.venv /app/.venv
COPY --from=backend-builder /app/src /app/src
COPY --from=backend-builder /app/api /app/api
COPY --from=backend-builder /app/config /app/config
COPY --from=backend-builder /app/pyproject.toml /app/pyproject.toml

# Copy built frontend from frontend builder
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create supervisor configuration
RUN mkdir -p /etc/supervisor/conf.d
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root

[program:nginx]
command=nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/var/log/nginx/access.log
stderr_logfile=/var/log/nginx/error.log

[program:fastapi]
command=/app/.venv/bin/python -m api.main
directory=/app
autostart=true
autorestart=true
stdout_logfile=/var/log/fastapi.log
stderr_logfile=/var/log/fastapi.log
environment=PATH="/app/.venv/bin:%(ENV_PATH)s"
EOF

# Create log directories
RUN mkdir -p /var/log/nginx

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]