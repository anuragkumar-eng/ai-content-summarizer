"""
summarizer.py
-------------
LangChain Map-Reduce summarisation supporting multiple FREE providers:

  Provider     | Free Tier              | Get Key At
  -------------|------------------------|-----------------------------
  Google Gemini| 15 RPM / 1500 RPD free | aistudio.google.com
  Groq         | 30 RPM / 14400 RPD free| console.groq.com
  OpenAI       | Paid only              | platform.openai.com

Robustness features:
  - Real-time live model detection from user API keys (no more 404/decommissioned errors).
  - Automatic fallback: If a selected model fails (404/decommissioned), seamlessly tries
    the next accessible model.
  - Automatic exponential retry on temporary 503 / 429 rate limit errors.
  - Single-pass optimization for standard-length text to minimize latency and quota.
  - Specialized summary personas (Executive Brief, Action Items, ELI5, Social Post).
  - Multilingual target translation & synthesis (10+ languages).
"""

import logging
import time
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported providers & baseline fallbacks
# ---------------------------------------------------------------------------
PROVIDERS = ["Groq (Free)", "Google Gemini (Free)", "OpenAI (Paid)"]

BASELINE_MODELS = {
    "Groq (Free)": [
        "llama-3.1-8b-instant",       # Universal Groq free model (always accessible)
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
    ],
    "Google Gemini (Free)": [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.5-pro",
    ],
    "OpenAI (Paid)": [
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-4o",
    ],
}

DEFAULT_MODELS = {
    "Groq (Free)":          "llama-3.1-8b-instant",
    "Google Gemini (Free)": "gemini-2.5-flash",
    "OpenAI (Paid)":        "gpt-4o-mini",
}

# ---------------------------------------------------------------------------
# Personas & Supported Target Languages
# ---------------------------------------------------------------------------
FORMAT_TYPES = [
    "Short Summary",
    "Detailed Bullet Points",
    "Executive Brief",
    "Action Items & Key Decisions",
    "Explain Like I'm 5 (ELI5)",
    "Social & LinkedIn Post",
]

LANGUAGES = [
    "English",
    "Spanish (Español)",
    "French (Français)",
    "German (Deutsch)",
    "Hindi (हिन्दी)",
    "Japanese (日本語)",
    "Chinese (中文)",
    "Portuguese (Português)",
    "Italian (Italiano)",
    "Arabic (العربية)",
]

# ---------------------------------------------------------------------------
# Dynamic Model Discovery (Queries provider API for exact accessible models)
# ---------------------------------------------------------------------------
def get_available_models(provider: str, api_key: str = "") -> list[str]:
    """Query the provider to get the real, active models for this specific API key."""
    if not api_key or not api_key.strip():
        return BASELINE_MODELS.get(provider, [])

    clean_key = api_key.strip()

    if provider == "Groq (Free)":
        try:
            import groq
            client = groq.Groq(api_key=clean_key)
            res = client.models.list()
            chat_models = [
                m.id for m in res.data
                if getattr(m, "active", True) and not any(
                    x in m.id.lower()
                    for x in ["whisper", "guard", "embed", "tts", "vision", "audio"]
                )
            ]
            if chat_models:
                if "llama-3.1-8b-instant" in chat_models:
                    chat_models.remove("llama-3.1-8b-instant")
                    return ["llama-3.1-8b-instant"] + sorted(chat_models)
                return sorted(chat_models)
        except Exception as exc:
            logger.debug("Groq dynamic model fetch fallback: %s", exc)

    elif provider == "Google Gemini (Free)":
        try:
            import google.generativeai as genai
            genai.configure(api_key=clean_key)
            models = [
                m.name.replace("models/", "")
                for m in genai.list_models()
                if "generateContent" in getattr(m, "supported_generation_methods", [])
            ]
            valid_models = [
                m for m in models
                if not any(x in m.lower() for x in ["embedding", "aqa", "vision", "imagen"])
            ]
            if valid_models:
                return valid_models
        except Exception as exc:
            logger.debug("Gemini dynamic model fetch fallback: %s", exc)

    elif provider == "OpenAI (Paid)":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=clean_key)
            res = client.models.list()
            chat_models = [
                m.id for m in res.data
                if "gpt" in m.id and "instruct" not in m.id and "audio" not in m.id
            ]
            if chat_models:
                return sorted(chat_models)
        except Exception as exc:
            logger.debug("OpenAI dynamic model fetch fallback: %s", exc)

    return BASELINE_MODELS.get(provider, [])


ALL_MODELS = BASELINE_MODELS

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1_200
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Specialized Persona Prompts (Direct Single-Pass)
# ---------------------------------------------------------------------------
DIRECT_PROMPT_TEMPLATE = {
    "Short Summary": """\
You are an expert technical editor. Summarise the following text into ONE concise,
clear, and fluent paragraph capturing all essential ideas without outside fluff.
Do NOT add outside information or speculate.
LANGUAGE DIRECTIVE: The final summary MUST be written in {language}.

Text:
---------
{text}
---------

Final paragraph summary ({language}):""",

    "Detailed Bullet Points": """\
You are an expert technical editor. Summarise the following text into a structured,
comprehensive bullet-point list covering every critical insight.
Each bullet must begin with "•". Do NOT add outside information or speculate.
LANGUAGE DIRECTIVE: The final output MUST be written in {language}.

Text:
---------
{text}
---------

Detailed bullet-point summary ({language}):""",

    "Executive Brief": """\
You are a senior executive strategist. Synthesize the text into an Executive Brief.
Structure your summary clearly with the following sections:
- 📌 Strategic Overview (2-3 concise sentences)
- 📊 Key Metrics, Findings & Evidence (bullet points)
- 💡 Strategic Implications & Takeaways (actionable points)

Do NOT speculate or add facts not present in the document.
LANGUAGE DIRECTIVE: The entire executive brief MUST be written in {language}.

Text:
---------
{text}
---------

Executive Brief ({language}):""",

    "Action Items & Key Decisions": """\
You are an expert technical project manager and analyst. Extract all actionable intelligence:
- 🎯 Key Decisions Made & Policies Established
- ✅ Action Items, Deliverables & Responsibilities (checklist format)
- ⏳ Timelines, Deadlines & Future Milestones

If any category has no items mentioned in the text, note "None specified in document".
LANGUAGE DIRECTIVE: The entire output MUST be written in {language}.

Text:
---------
{text}
---------

Action Items & Key Decisions ({language}):""",

    "Explain Like I'm 5 (ELI5)": """\
You are a brilliant educator renowned for making complex topics delightfully simple.
Explain the core ideas of the text so that a 5-year-old or curious non-expert can immediately grasp them:
- Use simple, everyday analogies
- Strip away corporate and academic jargon completely
- Keep sentences short, vivid, and friendly

LANGUAGE DIRECTIVE: The entire explanation MUST be written in {language}.

Text:
---------
{text}
---------

ELI5 Summary ({language}):""",

    "Social & LinkedIn Post": """\
You are an elite digital creator and thought leader. Transform the key ideas from the text into a viral, professional LinkedIn/Twitter post:
- 💥 Attention-grabbing opening hook line
- 💡 3-5 punchy takeaway points formatted with relevant emojis
- 🚀 One actionable conclusion or thought-provoking question for the audience
- 🏷️ 3-5 relevant trending hashtags at the end

LANGUAGE DIRECTIVE: The entire post MUST be written in {language}.

Text:
---------
{text}
---------

Social media post ({language}):""",
}

MAP_PROMPT_TEMPLATE = """\
You are an expert technical editor. Summarise ONLY the text excerpt below into a dense,
accurate factual summary. Preserve all critical statistics, names, and core arguments.
Do NOT add outside information or speculate.

Excerpt:
---------
{text}
---------

Concise factual summary:"""

# ---------------------------------------------------------------------------
# Specialized Persona Prompts (Map-Reduce Synthesis Step)
# ---------------------------------------------------------------------------
REDUCE_PROMPT_TEMPLATE = {
    "Short Summary": """\
You are an expert editor. Below are chunk-level summaries of a longer document.
Combine and synthesize them into ONE clear, fluent paragraph capturing the most vital ideas.
Do NOT add outside information or speculate.
LANGUAGE DIRECTIVE: The final synthesis MUST be written in {language}.

Chunk summaries:
{text}

Final paragraph summary ({language}):""",

    "Detailed Bullet Points": """\
You are an expert editor. Below are chunk-level summaries of a longer document.
Synthesize them into a clean, comprehensive bullet-point summary covering all key takeaways.
Each bullet must begin with "•". Group related points logically. Do NOT repeat points.
LANGUAGE DIRECTIVE: The final synthesis MUST be written in {language}.

Chunk summaries:
{text}

Detailed bullet-point summary ({language}):""",

    "Executive Brief": """\
You are a senior executive strategist. Below are chunk-level summaries of a longer document.
Synthesize them into an Executive Brief structured as follows:
- 📌 Strategic Overview (2-3 concise sentences)
- 📊 Key Metrics, Findings & Evidence (bullet points)
- 💡 Strategic Implications & Takeaways (actionable points)

Do NOT speculate or invent facts.
LANGUAGE DIRECTIVE: The entire executive brief MUST be written in {language}.

Chunk summaries:
{text}

Executive Brief ({language}):""",

    "Action Items & Key Decisions": """\
You are an expert technical project manager. Below are chunk-level summaries of a longer document.
Synthesize and extract all actionable intelligence:
- 🎯 Key Decisions Made & Policies Established
- ✅ Action Items, Deliverables & Responsibilities (checklist format)
- ⏳ Timelines, Deadlines & Future Milestones

If any category has no items mentioned in the summaries, note "None specified in document".
LANGUAGE DIRECTIVE: The entire output MUST be written in {language}.

Chunk summaries:
{text}

Action Items & Key Decisions ({language}):""",

    "Explain Like I'm 5 (ELI5)": """\
You are a brilliant educator renowned for making complex topics delightfully simple.
Below are chunk-level summaries of a longer document. Synthesize their core message so that a beginner can grasp it:
- Use simple, everyday analogies
- Strip away corporate and academic jargon completely
- Keep sentences short, vivid, and friendly

LANGUAGE DIRECTIVE: The entire explanation MUST be written in {language}.

Chunk summaries:
{text}

ELI5 Summary ({language}):""",

    "Social & LinkedIn Post": """\
You are an elite digital creator. Below are chunk-level summaries of a longer document.
Synthesize the most interesting takeaways into a viral, professional LinkedIn/Twitter post:
- 💥 Attention-grabbing opening hook line
- 💡 3-5 punchy takeaway points formatted with relevant emojis
- 🚀 One actionable conclusion or thought-provoking question for the audience
- 🏷️ 3-5 relevant trending hashtags at the end

LANGUAGE DIRECTIVE: The entire post MUST be written in {language}.

Chunk summaries:
{text}

Social media post ({language}):""",
}

# ---------------------------------------------------------------------------
# Helper: Call chain with automatic retry & exponential backoff
# ---------------------------------------------------------------------------
def _invoke_with_retry(chain, inputs: dict, max_retries: int = 3):
    """Executes a chain with exponential backoff on 503 (High Demand) or 429 errors."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            last_err = exc
            err_msg = str(exc).lower()
            is_transient = any(
                keyword in err_msg
                for keyword in ["503", "unavailable", "high demand", "429", "resource_exhausted", "rate limit"]
            )
            if is_transient and attempt < max_retries:
                wait_time = attempt * 3
                logger.warning(
                    "Temporary provider overload / rate limit. Retrying in %ds (attempt %d/%d)...",
                    wait_time,
                    attempt,
                    max_retries,
                )
                time.sleep(wait_time)
            else:
                raise last_err
    raise last_err


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def _build_llm(provider: str, api_key: str, model_name: str, temperature: float) -> BaseChatModel:
    """Instantiate the correct LangChain chat model for the chosen provider."""
    if provider == "Groq (Free)":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=model_name,
            groq_api_key=api_key,
            temperature=temperature,
        )

    elif provider == "Google Gemini (Free)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    elif provider == "OpenAI (Paid)":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Core Summarization Pipeline for a single model
# ---------------------------------------------------------------------------
def _run_pipeline(
    text: str,
    api_key: str,
    format_type: str,
    provider: str,
    model_name: str,
    temperature: float,
    language: str = "English",
) -> str:
    """Runs single-pass or Map-Reduce for the specified model and target language."""
    llm = _build_llm(provider, api_key, model_name, temperature)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = splitter.create_documents([text])
    if not chunks:
        raise ValueError("Text splitting produced zero chunks.")

    # Single-chunk optimisation: Direct Persona prompt with target language
    if len(chunks) == 1:
        template = DIRECT_PROMPT_TEMPLATE.get(format_type, DIRECT_PROMPT_TEMPLATE["Short Summary"])
        direct_chain = (
            ChatPromptTemplate.from_template(template)
            | llm
            | StrOutputParser()
        )
        result = _invoke_with_retry(
            direct_chain,
            {"text": chunks[0].page_content, "language": language},
        )
        return result.strip()

    # Multi-chunk Map-Reduce
    map_chain = ChatPromptTemplate.from_template(MAP_PROMPT_TEMPLATE) | llm | StrOutputParser()
    chunk_summaries: list[str] = []

    for i, doc in enumerate(chunks, start=1):
        chunk_summary = _invoke_with_retry(map_chain, {"text": doc.page_content})
        chunk_summaries.append(chunk_summary.strip())
        time.sleep(0.5)

    reduce_template = REDUCE_PROMPT_TEMPLATE.get(format_type, REDUCE_PROMPT_TEMPLATE["Short Summary"])
    reduce_chain = (
        ChatPromptTemplate.from_template(reduce_template)
        | llm
        | StrOutputParser()
    )
    combined = "\n\n".join(f"[Chunk {i}] {s}" for i, s in enumerate(chunk_summaries, start=1))
    final = _invoke_with_retry(
        reduce_chain,
        {"text": combined, "language": language},
    )
    return final.strip()


# ---------------------------------------------------------------------------
# Public API with Automatic Model Fallback
# ---------------------------------------------------------------------------
def summarize_text(
    text: str,
    api_key: str,
    format_type: str = "Short Summary",
    provider: str = "Groq (Free)",
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.3,
    language: str = "English",
) -> tuple[str, str]:
    """Run summarisation using the selected provider, persona style, and language.

    Returns
    -------
    tuple[str, str]: (final_summary, actual_model_used)
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty. Please provide content to summarise.")
    if not api_key or not api_key.strip():
        raise ValueError(f"{provider} API key is missing. Enter it in the sidebar.")
    if format_type not in REDUCE_PROMPT_TEMPLATE:
        raise ValueError(f"Unsupported format_type: {format_type}")

    available = get_available_models(provider, api_key)
    candidate_models = [model_name] + [m for m in available if m != model_name]

    last_error = None
    for idx, candidate in enumerate(candidate_models):
        try:
            logger.info("Attempting summarisation with %s / %s (Style: %s, Lang: %s)...", provider, candidate, format_type, language)
            result = _run_pipeline(
                text=text,
                api_key=api_key.strip(),
                format_type=format_type,
                provider=provider,
                model_name=candidate,
                temperature=temperature,
                language=language,
            )
            return result, candidate
        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            should_fallback = any(
                phrase in err_str
                for phrase in [
                    "not found",
                    "not_found",
                    "does not exist",
                    "decommissioned",
                    "404",
                    "model_not_found",
                    "access to it",
                ]
            )
            if should_fallback and idx < len(candidate_models) - 1:
                logger.warning(
                    "Model '%s' is unavailable (%s). Falling back to '%s'...",
                    candidate,
                    exc,
                    candidate_models[idx + 1],
                )
                continue
            else:
                raise last_error

    raise last_error or RuntimeError("Summarisation failed with all candidate models.")
