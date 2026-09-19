"""Chat Provider & Query Intelligence: chat_provider.py

Starter code for configuring LLM synthesis, system prompts, and query analysis (HyDE, intent decomposition, expansion).
"""

import os
from typing import Any, Dict, Optional
from aimlite.rag import (
    AnthropicChatProvider,
    BaseChatProvider,
    GeminiChatProvider,
    LocalChatProvider,
    MockChatProvider,
    OllamaChatProvider,
    OpenAIChatProvider,
    QueryAnalyzer,
    PROMPT_CONVERSATIONAL_RAG,
    PROMPT_QUERY_ANALYZER,
    PROMPT_RAG_QA,
    PROMPT_SMART_CHUNKER,
)


def get_chat_provider(
    provider_name: str = "ollama",
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatProvider:
    """Instantiates and returns the configured LLM Chat Provider reading models and endpoints from environment."""
    prompt = system_prompt or PROMPT_RAG_QA

    if provider_name == "openai":
        return OpenAIChatProvider(
            model=model_name or os.environ.get("OPENAI_MODEL", "qwen3:1.7b"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "gemini":
        return GeminiChatProvider(
            model=model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "anthropic":
        return AnthropicChatProvider(
            model=model_name or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "ollama":
        return OllamaChatProvider(
            model=model_name or os.environ.get("OLLAMA_MODEL", "qwen3:1.7b"),
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "local":
        return LocalChatProvider(
            model_name=model_name or os.environ.get("LOCAL_MODEL", "meta-llama/Llama-3.2-3B"),
            system_prompt=prompt,
            **kwargs,
        )
    else:
        return MockChatProvider(
            model=model_name or os.environ.get("MOCK_MODEL", "mock-gpt"),
            system_prompt=prompt,
            **kwargs,
        )


def get_query_analyzer(
    chat_provider: Optional[BaseChatProvider] = None,
    enable_hyde: bool = True,
) -> QueryAnalyzer:
    """Configures Query Intelligence for intent parsing, expansion, and hypothetical document generation."""
    provider = chat_provider or get_chat_provider()
    return QueryAnalyzer(chat_provider=provider, enable_hyde=enable_hyde)
