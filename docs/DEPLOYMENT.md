# Production Deployment & Containerization Guide

AIMLite applications are designed from the ground up for zero-friction containerization and cloud hosting. Because AIMLite utilizes convention over configuration, your entire project—model lifecycle, virtual environment, pre-trained checkpoints, and interactive web or custom frontend—packages into a lean, self-contained Docker image.

---

## 1. Production Dockerfile

Create a `Dockerfile` in the root of your AIMLite project:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies (curl for healthchecks & network tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install uv (blazing fast backend) and AIMLite framework
RUN pip install --no-cache-dir uv aimlite

# 2. Set project working directory
WORKDIR /app

# 3. Copy project manifest
COPY aimlite.json .

# 4. Install all project dependencies into managed .venv using aimlite CLI
RUN aimlite install

# 5. Copy the remaining application files, data, and frontend assets
COPY . .

# 6. Train and calibrate model weights for serving
RUN aimlite train

# 7. Expose serving port
EXPOSE 8000

# 8. Health check verifying inference server status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 9. Host multi-model inference server with custom frontend (dynamically binds to $PORT on cloud hosts)
CMD ["sh", "-c", "aimlite serve --host 0.0.0.0 --port ${PORT:-8000} --frontend frontend"]
```

---

## 2. Architecture Breakdown

| Stage | Directive | Rationale |
|---|---|---|
| **Base Image** | `FROM python:3.12-slim` | Minimal attack surface (~150MB base size), zero unnecessary build utilities, optimized for Python 3.12 wheel compatibility. |
| **Environment** | `ENV PYTHONUNBUFFERED=1` | Direct real-time terminal stdout/stderr streaming for Docker log drivers and cloud monitoring agents without buffering delay. |
| **Environment** | `ENV PATH="/app/.venv/bin:$PATH"` | Direct access to all binaries inside AIMLite's project virtual environment without needing manual activation. |
| **System Tooling**| `RUN apt-get install curl` | Installs lightweight networking tools required for native Docker container health checks. |
| **Fast Backend** | `RUN pip install uv aimlite` | Installs Astral `uv` for 10x-100x faster dependency resolution, plus the global `aimlite` CLI engine. |
| **Layer Caching**| `COPY aimlite.json .` + `RUN aimlite install` | **Critical caching step**: Dependencies are resolved into `.venv` before application code is copied. If you change code or datasets, Docker skips re-installing packages. |
| **Bake Weights** | `RUN aimlite train` | Trains model weights and saves checkpoints (`models/*.pkl`) directly into the image layer at build time. Containers start instantaneously with **zero cold-start latency**. |
| **Health Probe** | `HEALTHCHECK ... CMD curl` | Actively tests `GET /health` every 30 seconds. Unhealthy or unresponsive containers are automatically restarted by container orchestrators. |
| **Dynamic Port** | `CMD ["sh", "-c", "... ${PORT:-8000}"]` | Seamlessly accepts `$PORT` dynamically assigned by PaaS hosts (Render, Railway, Fly.io, Cloud Run) with fallback to `8000`. |
| **Unified SPA**  | `--frontend frontend` | Hosts your custom compiled single-page application (React, Vite, Vue) directly alongside inference endpoints on a single port with zero CORS configuration. |

---

## 3. Recommended `.dockerignore`

Add a `.dockerignore` file in your project root to exclude local environments, caches, and test artifacts:

```text
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
.pytest_cache
experiments/*.log
artifacts/cache/
.DS_Store
*.swp
```

---

## 4. Local Build & Run Walkthrough

```bash
# 1. Build the Docker image
docker build -t my-aimlite-app .

# 2. Run the container in detached mode mapping port 8000
docker run -d --name aimlite-service -p 8000:8000 my-aimlite-app

# 3. Check health endpoint status
curl http://localhost:8000/health
# Response: {"status":"healthy","uptime_seconds":5.2,"active_model":"ChurnClassifier","models":["ChurnClassifier"]}

# 4. Perform an inference prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [128.0, 1.0, 2.7, 1.0, 265.1, 110.0, 89.0, 9.8, 10.0]}'

# 5. Inspect container logs
docker logs -f aimlite-service

# 6. Stop and remove container
docker stop aimlite-service && docker rm aimlite-service
```

---

## 5. Docker Compose (`docker-compose.yml`)

For local development or multi-container stacks (e.g. AIMLite + PostgreSQL with `pgvector` for enterprise RAG):

```yaml
version: "3.8"

services:
  aimlite:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${PORT:-8000}:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - DATABASE_URL=postgresql://postgres:secret@postgres:5432/aimlite_db
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      - GEMINI_API_KEY=${GEMINI_API_KEY:-}
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: aimlite_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

---

## 6. Cloud Platform Deployment

### Google Cloud Run (Serverless Container)

```bash
# Set your GCP Project ID
export PROJECT_ID="my-gcp-project"

# 1. Build and push to Google Artifact Registry
gcloud builds submit --tag gcr.io/${PROJECT_ID}/aimlite-app

# 2. Deploy to Cloud Run with automatic scaling and dynamic $PORT
gcloud run deploy aimlite-service \
  --image gcr.io/${PROJECT_ID}/aimlite-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Render / Railway / Fly.io

1. Connect your GitHub repository containing the `Dockerfile` and `aimlite.json`.
2. Set service type to **Web Service / Docker**.
3. Cloud hosts dynamically populate the `$PORT` environment variable. The shell command in `CMD` binds to `${PORT:-8000}` automatically.
4. Set health check path to `/health`.

### Kubernetes (`deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aimlite-inference
  labels:
    app: aimlite
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aimlite
  template:
    metadata:
      labels:
        app: aimlite
    spec:
      containers:
      - name: aimlite
        image: my-dockerhub-user/aimlite-app:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: aimlite-service
spec:
  type: LoadBalancer
  selector:
    app: aimlite
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```
