"""Client test script for test RAG Knowledge Base."""

import os
import urllib.request
from test.data import KnowledgeDocsDataset
from test.model import SupportDocRAG


def check_provider_health(model: SupportDocRAG) -> None:
    """Pre-flight check verifying whether local servers (Ollama) or required API keys are configured."""
    provider = getattr(model, "chat_provider", None)
    if not provider:
        return

    provider_name = provider.__class__.__name__

    if "Ollama" in provider_name:
        host = getattr(provider, "base_url", "http://localhost:11434")
        model_name = getattr(provider, "model", "llama3.2")
        try:
            req = urllib.request.Request(f"{host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                pass
            print(f"[*] Ollama server connected at {host} (model: {model_name})")
        except Exception:
            print(f"\n[!] Note: Ollama daemon is not responding at {host}.")
            print(f"    To run Ollama locally:")
            print(f"      1. Start server:   ollama serve")
            print(f"      2. Pull model:     ollama pull {model_name}")
            print(f"      3. Set OLLAMA_HOST or OLLAMA_MODEL in .env if using custom settings\n")

    elif "OpenAI" in provider_name:
        if not getattr(provider, "api_key", None) and not os.environ.get("OPENAI_API_KEY"):
            print("\n[!] Note: OPENAI_API_KEY is not set in .env. LLM generation requires an active key.\n")
    elif "Anthropic" in provider_name:
        if not getattr(provider, "api_key", None) and not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n[!] Note: ANTHROPIC_API_KEY is not set in .env. LLM generation requires an active key.\n")
    elif "Gemini" in provider_name:
        if not getattr(provider, "api_key", None) and not os.environ.get("GEMINI_API_KEY"):
            print("\n[!] Note: GEMINI_API_KEY is not set in .env. LLM generation requires an active key.\n")


def main() -> None:
    print("[*] Initializing test Knowledge Model...")
    model = SupportDocRAG()
    dataset = KnowledgeDocsDataset()

    check_provider_health(model)

    # Ingest and chunk documents
    docs = dataset.load_documents()
    print(f"[*] Ingested and chunked {len(docs)} passage(s) from data/.")
    model.index_documents(docs)

    # Execute test queries
    test_queries = [
        "How do session tokens expire?",
        "How can I switch to PostgreSQL for vector storage?",
    ]

    for test_query in test_queries:
        print(f"\n[?] Query: {test_query}")
        result = model.predict(test_query, top_k=2)

        print(f"[!] Answer: {result['answer']}")
        print(f"[!] Sources Citations:")
        for idx, s in enumerate(result.get("sources", []), start=1):
            meta = s.get("metadata", {})
            source_label = meta.get("source", "knowledge_base.md")
            if meta.get("heading_path"):
                source_label = f"{source_label} ({meta.get('heading_path')})"
            score_str = f"{s.get('score', 0):.4f}" if isinstance(s.get("score"), (int, float)) else str(s.get("score"))
            print(f"    [{idx}] {source_label} (score: {score_str})")


if __name__ == "__main__":
    main()
