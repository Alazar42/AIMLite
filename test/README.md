# Test (RAG Knowledge Engine)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Project Structure
- `test/chat_provider.py`: Chat Provider & Query Intelligence LLM configuration.
- `test/store.py`: Vector store, PostgreSQL ORM, and embedding setup.
- `test/model.py`: High-level Knowledge Model connecting retrieval and generation.
- `test/data.py`: SmartChunker document ingestion dataset.
- `experiments/benchmark.py`: Latency & retrieval quality benchmark script.
- `client.py`: Ready-to-run interactive/batch client query starter.
- `.env`: Active local environment configuration with model identifiers.
- `.env.example`: Template for sensitive credentials, database URLs, and API keys.

---

## Environment Variables & Secrets

AIMLite reads sensitive configuration (API keys, database credentials, model names, host URLs) from system environment variables or a local `.env` file in the project root.

### Configured Environment Variables

| Variable | Description | Default / Example Value |
|---|---|---|
| `OPENAI_MODEL` | OpenAI Model Identifier | `gpt-4o-mini` |
| `OLLAMA_MODEL` | Ollama Local Model Identifier | `llama3.2` |
| `GEMINI_MODEL` | Google Gemini Model Identifier | `gemini-1.5-flash` |
| `ANTHROPIC_MODEL` | Anthropic Claude Model Identifier | `claude-3-5-sonnet-20241022` |
| `EMBEDDING_MODEL` | Semantic Dense Vector Embedding Model | `all-MiniLM-L6-v2` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-proj-abc123xyz456...` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `sk-ant-api03-abc...` |
| `GEMINI_API_KEY` | Google Gemini API Key | `AIzaSyD...` |
| `OLLAMA_HOST` | Ollama daemon endpoint (local LLM) | `http://localhost:11434` |
| `DATABASE_URL` | PostgreSQL connection string for pgvector ORM | `postgresql://user:pass@localhost:5432/my_rag_db` |

#### Example `.env` file:
```ini
# --- Model Identifiers ---
OLLAMA_MODEL=qwen3:1.7b
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=nomic-embed-text

# --- LLM API Credentials (if using cloud providers) ---
OPENAI_API_KEY=
GEMINI_API_KEY=

# --- Local Ollama Endpoint ---
OLLAMA_HOST=http://localhost:11434

# --- Database & Vector Storage (PostgreSQL ORM) ---
DATABASE_URL=postgresql://postgres:secretpassword@localhost:5432/knowledge_db

# --- Hardware Acceleration & Serving ---
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

# 3. Build & persist vector index
aimlite train

# 4. Run local client test
python client.py

# 5. Run retrieval benchmark
python experiments/benchmark.py

# 6. Serve HTTP API
aimlite serve --port 8000
```
