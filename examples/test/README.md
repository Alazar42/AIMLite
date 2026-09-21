# Test (AIMLite Custom ML)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Project Structure
- `test/model.py`: Model architecture and forward prediction.
- `test/data.py`: Application dataset ingestion.
- `test/trainer.py`: Model training orchestration.
- `experiments/benchmark.py`: Forward throughput & latency benchmark.
- `client.py`: Ready-to-run test client.
- `.env.example`: Template for environment variables and secrets.

---

## Environment Variables & Secrets

AIMLite reads sensitive credentials from system environment variables or a local `.env` file in the project root.

### 1. Create your `.env` file
Copy the provided `.env.example` template:
```bash
cp .env.example .env
```

### 2. Configure Credentials

| Variable | Description | Example Value |
|---|---|---|
| `DATABASE_URL` | Feature store or SQL database connection string | `postgresql://user:pass@localhost:5432/ml_db` |
| `AIMLITE_DEVICE` | Hardware acceleration target (`auto`, `cuda`, `cpu`) | `auto` |
| `AIMLITE_PORT` | HTTP REST API server port | `8000` |

#### Example `.env` file:
```ini
# --- Database / Feature Store ---
DATABASE_URL=postgresql://postgres:secretpassword@localhost:5432/ml_db

# --- Serving ---
AIMLITE_DEVICE=auto
AIMLITE_PORT=8000
```

> **Security Note:** Never commit `.env` containing sensitive credentials to Git. `.env` is already configured in `.gitignore`.

---

## Quickstart

```bash
# 1. Install dependencies
aimlite install

# 2. Validate data
aimlite data validate

# 3. Train model
aimlite train

# 4. Evaluate performance
aimlite evaluate

# 5. Run local client test
python client.py

# 6. Run benchmark
python experiments/benchmark.py

# 7. Serve HTTP API
aimlite serve --port 8000
```
