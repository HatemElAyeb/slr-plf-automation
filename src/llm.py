"""
LLM factory — returns a ChatOllama or ChatGroq based on settings.llm_provider.
Centralizes the choice so screener / extractor / synthesizer don't duplicate logic.
"""
from config.settings import settings


def get_llm(temperature: float = 0, json_mode: bool = True):
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {
            "model": settings.groq_screening_model,
            "api_key": settings.groq_api_key,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**kwargs)

    # Default: Ollama
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.screening_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        format="json" if json_mode else None,
    )
