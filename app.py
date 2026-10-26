"""
app.py  —  AI-Powered Content Summarizer Studio
Modern, sleek SaaS-grade UI for LangChain-powered summarization & Q&A.
Supports Groq (Free), Google Gemini (Free), and OpenAI (Paid).
Features:
  - 4 Ingestion Formats: Raw Text, Web URLs, PDF Documents, YouTube Videos
  - 6 Specialized Summary Personas (Executive, Action Items, ELI5, Social, etc.)
  - Multilingual Output (10+ Languages)
  - Interactive "Chat with Document" Q&A Assistant (Phase 3)
  - Pre-processing metrics & Post-processing compression analytics
  - Auto-scrapes URLs, extracts PDFs, & handles YouTube transcripts
"""

import os
import io
import re
import math
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS

from document_processor import (
    extract_text_from_pdf,
    extract_text_from_url,
    extract_text_from_youtube,
    extract_youtube_video_id,
)
from summarizer import (
    summarize_text,
    answer_document_question,
    get_available_models,
    PROVIDERS,
    DEFAULT_MODELS,
    FORMAT_TYPES,
    LANGUAGES,
)

# Reload .env on every run so updates take immediate effect
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Summarizer Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern CSS Theme Injection
# ---------------------------------------------------------------------------
MODERN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Top Header / Hero ────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 2.2rem 1rem 1.8rem 1rem;
    background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin: 0.4rem 0 0.8rem 0;
}

.gradient-text {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Modern Sidebar Branding ──────────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.4rem 1.2rem 0.4rem;
}

.brand-icon-box {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

.brand-name {
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1.2;
}

.brand-tag {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Status Pills ────────────────────────────────────────────── */
.status-pill-ok {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.28);
    color: #34d399;
    padding: 0.5rem 0.8rem;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.6rem 0;
}

.status-pill-warn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.28);
    color: #fbbf24;
    padding: 0.5rem 0.8rem;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.6rem 0;
}

/* ── Interactive Stats Cards ─────────────────────────────────── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
}

.stat-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.stat-icon {
    font-size: 1.15rem;
    margin-bottom: 0.3rem;
}

.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1.2;
}

.stat-lbl {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-top: 0.2rem;
}

/* ── Summary Result Card ─────────────────────────────────────── */
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding-bottom: 0.8rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.result-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}

.meta-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.meta-pill {
    font-size: 0.75rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.meta-pill-lang {
    font-size: 0.75rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* ── Analytics Bar ───────────────────────────────────────────── */
.analytics-bar {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    background: rgba(15, 23, 42, 0.6);
    padding: 0.75rem 1.1rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 1.2rem;
}

.analytics-item {
    font-size: 0.85rem;
    color: #cbd5e1;
}

.analytics-item b {
    color: #ffffff;
    font-weight: 700;
}

.analytics-highlight {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
}

/* ── Q&A Assistant Section ───────────────────────────────────── */
.chat-container-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0 0.4rem 0;
}

.chat-badge {
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
    font-weight: 600;
}

/* ── Polished Streamlit Buttons ───────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    padding: 0.65rem 1.4rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.25s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.5) !important;
}

div.stButton > button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
}
</style>
"""
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TTS Audio Helpers (Phase 4)
# ---------------------------------------------------------------------------
TTS_LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Portuguese": "pt",
    "Italian": "it",
    "Arabic": "ar",
    "Russian": "ru",
    "Korean": "ko",
    "Dutch": "nl",
    "Turkish": "tr",
}

def clean_markdown_for_speech(text: str) -> str:
    """Strip markdown formatting syntax to produce clean, natural-sounding speech."""
    # Remove markdown links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove headers (#, ##, ###)
    cleaned = re.sub(r'#+\s*', '', cleaned)
    # Remove bold/italic/code markers (*, _, **, __, `)
    cleaned = re.sub(r'[*_~`]', '', cleaned)
    # Replace bullet point symbols
    cleaned = re.sub(r'^\s*[-•*+]\s+', '', cleaned, flags=re.MULTILINE)
    # Remove blockquotes
    cleaned = re.sub(r'^\s*>\s+', '', cleaned, flags=re.MULTILINE)
    # Remove excessive blank lines
    cleaned = re.sub(r'\n+', '\n', cleaned).strip()
    return cleaned

def generate_audio_summary(text: str, language_name: str) -> io.BytesIO:
    """Convert summarized text into an in-memory MP3 audio stream using gTTS."""
    lang_code = TTS_LANG_MAP.get(language_name, "en")
    clean_text = clean_markdown_for_speech(text)
    if not clean_text:
        clean_text = "No content available to read."
    tts = gTTS(text=clean_text, lang=lang_code, slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "active_summary" not in st.session_state:
    st.session_state["active_summary"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "active_doc_text" not in st.session_state:
    st.session_state["active_doc_text"] = ""
if "current_input_text" not in st.session_state:
    st.session_state["current_input_text"] = ""
if "cached_audio" not in st.session_state:
    st.session_state["cached_audio"] = None
if "cached_audio_id" not in st.session_state:
    st.session_state["cached_audio_id"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------------------------------------------------------------
# Provider Configurations & Environment Lookups
# ---------------------------------------------------------------------------
PROVIDER_ENV_VARS = {
    "Groq (Free)":          "GROQ_API_KEY",
    "Google Gemini (Free)": "GEMINI_API_KEY",
    "OpenAI (Paid)":        "OPENAI_API_KEY",
}

KEY_LINKS = {
    "Groq (Free)":          ("Groq Console",     "https://console.groq.com/keys"),
    "Google Gemini (Free)": ("Google AI Studio", "https://aistudio.google.com/app/apikey"),
    "OpenAI (Paid)":        ("OpenAI Platform",  "https://platform.openai.com/api-keys"),
}

FREE_BADGE = {
    "Groq (Free)":          "🟢 14,400 Free Requests / Day",
    "Google Gemini (Free)": "🟢 1,500 Free Requests / Day",
    "OpenAI (Paid)":        "🔴 Paid / Token Usage",
}

def get_env_api_key(provider_name: str) -> str:
    """Retrieve the API key for the selected provider from environment / .env."""
    env_var = PROVIDER_ENV_VARS.get(provider_name, "")
    val = os.getenv(env_var, "").strip()
    if not val and provider_name == "Google Gemini (Free)":
        val = os.getenv("GOOGLE_API_KEY", "").strip()
    return val

# ---------------------------------------------------------------------------
# Sidebar Studio Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon-box">⚡</div>
            <div class="brand-text">
                <div class="brand-name">Summarizer Studio</div>
                <div class="brand-tag">LangChain & LLM Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Engine Settings")

    # Provider Selection
    provider = st.selectbox(
        "AI Provider",
        options=PROVIDERS,
        index=0,
        help="Groq and Gemini are completely free and require no credit card.",
    )
    st.caption(f"**Tier:** {FREE_BADGE[provider]}")

    # API Key Status Display
    env_var_name = PROVIDER_ENV_VARS.get(provider, "API_KEY")
    api_key = get_env_api_key(provider)

    if api_key:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "••••••••"
        st.markdown(
            f"""
            <div class="status-pill-ok">
                <span>🟢</span>
                <div>
                    <div><b>Active in .env</b></div>
                    <div style="font-size:0.72rem; opacity:0.8;">{env_var_name} ({masked_key})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        link_label, link_url = KEY_LINKS[provider]
        st.markdown(
            f"""
            <div class="status-pill-warn">
                <span>⚠️</span>
                <div>
                    <div><b>Key Missing in .env</b></div>
                    <div style="font-size:0.72rem; opacity:0.85;">Set {env_var_name} to activate</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"👉 [Get free API key at {link_label}]({link_url})")

    st.markdown("---")

    # Dynamic Model Discovery
    available_models = get_available_models(provider, api_key)
    default_model = DEFAULT_MODELS.get(provider, available_models[0] if available_models else "")
    default_idx = available_models.index(default_model) if default_model in available_models else 0

    model_name = st.selectbox(
        "Model Selection",
        options=available_models,
        index=default_idx,
        help="Automatically discovered from your API credentials.",
    )

    st.markdown("---")
    st.markdown("### 🎯 Output Customization")

    # Persona / Style Selector
    format_type = st.selectbox(
        "Summary Persona",
        options=FORMAT_TYPES,
        index=0,
        help="Select the specialized lens through which to synthesize the content.",
    )

    # Multilingual Target Language Selector
    target_language = st.selectbox(
        "🌐 Target Output Language",
        options=LANGUAGES,
        index=0,
        help="The document will be synthesized and translated directly into this language.",
    )

    st.markdown("---")
    st.markdown(f"### 📚 Session History ({len(st.session_state['history'])})")

    if not st.session_state["history"]:
        st.caption("No summaries in this session yet. Summarize any content to save here.")
    else:
        history_labels = [
            f"{i+1}. [{item['time']}] {item['format']} ({item['source'][:16]}...)"
            for i, item in enumerate(st.session_state["history"])
        ]
        selected_idx = st.selectbox(
            "Select past summary",
            options=range(len(history_labels)),
            format_func=lambda i: history_labels[i],
            key="history_selector",
            label_visibility="collapsed",
        )
        col_hist_load, col_hist_clear = st.columns([1, 1])
        with col_hist_load:
            if st.button("📂 Load", use_container_width=True, help="Load this summary to main view"):
                selected_item = st.session_state["history"][selected_idx]
                st.session_state["active_summary"] = selected_item["summary"]
                st.session_state["active_model"] = selected_item["model"]
                st.session_state["active_provider"] = selected_item["provider"]
                st.session_state["active_format"] = selected_item["format"]
                st.session_state["active_language"] = selected_item["language"]
                st.session_state["active_doc_text"] = selected_item["doc_text"]
                st.session_state["active_source"] = selected_item["source"]
                st.session_state["orig_words"] = selected_item["orig_words"]
                st.session_state["sum_words"] = selected_item["sum_words"]
                st.session_state["compression_pct"] = selected_item["compression_pct"]
                st.session_state["time_saved"] = selected_item["time_saved"]
                st.session_state["chat_history"] = []
                st.session_state["cached_audio"] = None
                st.session_state["cached_audio_id"] = None
                st.rerun()
        with col_hist_clear:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear session history"):
                st.session_state["history"] = []
                st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            """
            - **Interactive Q&A**: Ask follow-up questions about the document anytime.
            - **YouTube Summarizer**: Extracts captions directly or supports instant fallback.
            - **Personas**: Custom engineered prompts extract executive takeaways or action items.
            - **Multilingual**: LLMs synthesize fluently into your target language.
            - **Auto-Fallback**: Retries with backoff on any provider rate limits automatically.
            """
        )

# ---------------------------------------------------------------------------
# Main Stage: Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-pill">✨ Next-Gen Content Intelligence</div>
        <h1 class="hero-title">AI Content <span class="gradient-text">Summarizer</span></h1>
        <p class="hero-subtitle">
            Instantly condense long articles, YouTube videos, research papers, and PDFs into crystal-clear executive summaries using high-speed LLMs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Quick Action / Sample Ingestion Helper
# ---------------------------------------------------------------------------
SAMPLE_TEXT = (
    "Artificial intelligence (AI) has emerged as one of the defining technological breakthroughs "
    "of the 21st century. At its core, AI encompasses systems and machines that emulate human cognitive "
    "functions, such as learning, reasoning, pattern recognition, and autonomous problem-solving. "
    "Machine learning, a foundational subfield of AI, allows algorithms to iteratively improve their performance "
    "by ingesting vast datasets without being explicitly programmed for every edge case. Deep learning further "
    "scales this paradigm through multi-layered artificial neural networks modeled loosely after human biology.\n\n"
    "In enterprise and scientific contexts, modern Large Language Models (LLMs) have revolutionized natural language "
    "understanding, code generation, and knowledge synthesis. However, deploying AI systems responsibly demands rigorous "
    "safeguards around data privacy, algorithmic bias, copyright integrity, and energy consumption. As organizations "
    "increasingly integrate autonomous agents into critical workflows, pair-programming assistants, and healthcare diagnosis, "
    "fostering transparent human-in-the-loop oversight remains the paramount challenge of contemporary technology governance."
)

col_tools1, col_tools2 = st.columns([1, 1])
with col_tools1:
    if st.button("💡 Load Sample Article", help="Click to populate sample text for instant testing"):
        st.session_state["current_input_text"] = SAMPLE_TEXT
        st.rerun()

with col_tools2:
    if st.session_state["current_input_text"]:
        if st.button("🧹 Clear Input", help="Clear the currently loaded text"):
            st.session_state["current_input_text"] = ""
            st.session_state["active_summary"] = None
            st.session_state["chat_history"] = []
            st.session_state["active_doc_text"] = ""
            st.rerun()

# ---------------------------------------------------------------------------
# Multi-Format Ingestion Tabs (Text, Web URL, PDF, YouTube)
# ---------------------------------------------------------------------------
tab_text, tab_url, tab_pdf, tab_yt = st.tabs([
    "📝 Raw Text / Notes",
    "🔗 Web Article URL",
    "📄 PDF Document",
    "🎥 YouTube Video",
])

raw_text = ""
input_label = ""

with tab_text:
    text_input = st.text_area(
        label="Document Text",
        value=st.session_state.get("current_input_text", ""),
        placeholder="Paste your long-form text, research paper, transcript, or blog post here...",
        height=260,
        label_visibility="collapsed",
    )
    if text_input.strip():
        raw_text = text_input.strip()
        input_label = "Raw Text Input"

with tab_url:
    url_input = st.text_input(
        label="Article URL",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
        label_visibility="collapsed",
    )
    st.caption("🌐 Auto-scrapes primary article content while stripping navigation, footers, and advertising noise.")

    if url_input.strip():
        with st.spinner("🔍 Fetching and cleaning web article..."):
            try:
                raw_text = extract_text_from_url(url_input.strip())
                input_label = f"URL: {url_input.strip()}"
                st.success(f"✅ Extracted **{len(raw_text):,}** characters from the page.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"❌ URL Extraction Error: {exc}")

with tab_pdf:
    uploaded_file = st.file_uploader(
        label="Upload PDF Document",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload text-based PDF files (up to 200MB).",
    )
    if uploaded_file:
        with st.spinner("📄 Extracting text layers from PDF..."):
            try:
                raw_text = extract_text_from_pdf(uploaded_file)
                input_label = f"PDF: {uploaded_file.name}"
                st.success(f"✅ Extracted **{len(raw_text):,}** characters from **{uploaded_file.name}**.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"❌ PDF Extraction Error: {exc}")

with tab_yt:
    yt_input = st.text_input(
        label="YouTube Video Link",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        label_visibility="collapsed",
    )
    st.caption("🎥 Fetches full captions/subtitles from YouTube videos (lectures, podcasts, news, tutorials).")

    if yt_input.strip():
        with st.spinner("🔍 Fetching YouTube transcript and video details..."):
            try:
                raw_text, vid_id = extract_text_from_youtube(yt_input.strip())
                input_label = f"YouTube: https://youtu.be/{vid_id}"
                st.success(f"✅ Extracted **{len(raw_text):,}** characters of spoken transcript.")
                st.video(f"https://www.youtube.com/watch?v={vid_id}")
            except RuntimeError as exc:
                if str(exc) == "YOUTUBE_IP_BLOCKED":
                    vid_id = extract_youtube_video_id(yt_input.strip())
                    st.warning(
                        "🛡️ **Cloud Datacenter IP Blocked by YouTube**: YouTube prevents automated cloud servers "
                        "(like Streamlit Cloud / AWS) from scraping captions directly."
                    )
                    st.info(
                        "💡 **2 Easy Ways to Summarize this Video:**\n\n"
                        "1. **Local App**: Run this app locally on your computer (`streamlit run app.py`) — your residential IP is never blocked by YouTube!\n"
                        "2. **Quick Copy**: On the YouTube video, click **'… More' ➔ 'Show transcript'**, then copy and paste it into the box below:"
                    )
                    if vid_id:
                        st.video(f"https://www.youtube.com/watch?v={vid_id}")
                    manual_transcript = st.text_area(
                        label="Paste YouTube Transcript Here",
                        placeholder="Paste the copied transcript from YouTube here...",
                        height=160,
                    )
                    if manual_transcript.strip():
                        raw_text = manual_transcript.strip()
                        input_label = f"YouTube: https://youtu.be/{vid_id}" if vid_id else "YouTube Transcript"
                        st.success(f"✅ Loaded **{len(raw_text):,}** characters of video transcript.")
                else:
                    st.error(f"❌ YouTube Error: {exc}")
            except ValueError as exc:
                st.error(f"❌ YouTube Error: {exc}")

# ---------------------------------------------------------------------------
# Live Pre-Processing Statistics Bar
# ---------------------------------------------------------------------------
if raw_text:
    char_count = len(raw_text)
    word_count = len(raw_text.split())
    approx_tokens = math.ceil(char_count / 4)
    est_reading_time = math.ceil(word_count / 220)

    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-val">{char_count:,}</div>
                <div class="stat-lbl">Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📚</div>
                <div class="stat-val">{word_count:,}</div>
                <div class="stat-lbl">Total Words</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔢</div>
                <div class="stat-val">~{approx_tokens:,}</div>
                <div class="stat-lbl">Estimated Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏱️</div>
                <div class="stat-val">~{est_reading_time} min</div>
                <div class="stat-lbl">Original Read Time</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Action CTA Button
# ---------------------------------------------------------------------------
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
summarize_btn = st.button(
    f"✨ Generate {format_type}",
    type="primary",
    use_container_width=True,
    disabled=not bool(raw_text),
)

# ---------------------------------------------------------------------------
# Execution Logic (Stores results into Session State)
# ---------------------------------------------------------------------------
if summarize_btn:
    if not api_key:
        link_label, link_url = KEY_LINKS[provider]
        st.error(
            f"❌ **API Key Missing**: `{env_var_name}` is not configured in your `.env` file.\n\n"
            f"Please open your `.env` file and set `{env_var_name}=your_key_here`\n\n"
            f"[👉 Get your free {provider} key at {link_label}]({link_url})"
        )
        st.stop()

    with st.spinner(f"⚡ Synthesizing {format_type} in {target_language} using {provider} ({model_name})..."):
        try:
            summary, actual_model = summarize_text(
                text=raw_text,
                api_key=api_key,
                format_type=format_type,
                provider=provider,
                model_name=model_name,
                language=target_language,
            )
        except ValueError as exc:
            st.error(f"❌ Validation Error: {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"❌ Summarization Error: {exc}")
            link_label, link_url = KEY_LINKS[provider]
            st.info(f"💡 Verify your API key at [{link_label}]({link_url})")
            st.stop()

    # Calculate Compression Analytics
    orig_words = len(raw_text.split())
    sum_words = len(summary.split())
    compression_pct = max(0, round((1 - (sum_words / max(orig_words, 1))) * 100))
    time_saved = max(0.5, round((orig_words - sum_words) / 220, 1))

    # Persist in Session State so Q&A doesn't wipe output on re-run
    st.session_state["active_summary"] = summary
    st.session_state["active_model"] = actual_model
    st.session_state["active_provider"] = provider
    st.session_state["active_format"] = format_type
    st.session_state["active_language"] = target_language
    st.session_state["active_doc_text"] = raw_text
    st.session_state["active_source"] = input_label
    st.session_state["orig_words"] = orig_words
    st.session_state["sum_words"] = sum_words
    st.session_state["compression_pct"] = compression_pct
    st.session_state["time_saved"] = time_saved
    st.session_state["chat_history"] = []  # fresh chat for new document/summary
    st.session_state["cached_audio"] = None
    st.session_state["cached_audio_id"] = None

    # Record into Session History
    history_entry = {
        "id": f"{int(datetime.now().timestamp())}",
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": input_label if input_label else "Raw Text",
        "format": format_type,
        "language": target_language,
        "provider": provider,
        "model": actual_model,
        "summary": summary,
        "doc_text": raw_text,
        "orig_words": orig_words,
        "sum_words": sum_words,
        "compression_pct": compression_pct,
        "time_saved": time_saved,
    }
    st.session_state["history"].insert(0, history_entry)
    st.rerun()

# ---------------------------------------------------------------------------
# Output Display & Interactive Document Q&A (Phase 3)
# ---------------------------------------------------------------------------
if st.session_state.get("active_summary"):
    act_summary = st.session_state["active_summary"]
    act_format = st.session_state.get("active_format", "Summary")
    act_prov = st.session_state.get("active_provider", provider)
    act_model = st.session_state.get("active_model", model_name)
    act_lang = st.session_state.get("active_language", "English")
    orig_words = st.session_state.get("orig_words", 0)
    sum_words = st.session_state.get("sum_words", 0)
    compression_pct = st.session_state.get("compression_pct", 0)
    time_saved = st.session_state.get("time_saved", 0)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.success(f"🎉 {act_format} Generated Successfully in {act_lang}!")

    # Styled Executive Summary Card
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="result-header">
                <div class="result-title">📄 {act_format}</div>
                <div class="meta-badges">
                    <span class="meta-pill">{act_prov}</span>
                    <span class="meta-pill">{act_model}</span>
                    <span class="meta-pill-lang">🌐 {act_lang}</span>
                </div>
            </div>
            <div class="analytics-bar">
                <div class="analytics-item">Original: <b>{orig_words:,} words</b></div>
                <div style="color: #64748b;">➔</div>
                <div class="analytics-item">Summary: <b>{sum_words:,} words</b></div>
                <div class="analytics-highlight">📉 {compression_pct}% Compressed</div>
                <div class="analytics-item">⏱️ Saved: <b>~{time_saved} min read</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(act_summary)

    # Action Toolbar (Download & Raw View)
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        clean_file_name = f"summary_{act_format.lower().replace(' ', '_')}_{act_lang.split()[0].lower()}.txt"
        st.download_button(
            label="⬇️ Download Summary (.txt)",
            data=act_summary,
            file_name=clean_file_name,
            mime="text/plain",
            use_container_width=True,
        )
    with col_dl2:
        with st.expander("📋 View Raw Markdown"):
            st.code(act_summary, language="markdown")

    # -----------------------------------------------------------------------
    # Audio Brief / Text-to-Speech (Phase 4)
    # -----------------------------------------------------------------------
    with st.expander("🎧 Audio Brief (Listen & Download MP3)", expanded=False):
        current_summary_id = f"{act_format}_{len(act_summary)}_{act_lang}"
        col_audio1, col_audio2 = st.columns([1, 2])
        with col_audio1:
            gen_audio_btn = st.button("▶️ Generate Audio Brief", use_container_width=True)
        with col_audio2:
            st.caption(f"Synthesizes natural spoken voice in **{act_lang}** via gTTS.")

        if gen_audio_btn:
            with st.spinner("🎙️ Synthesizing spoken audio brief..."):
                try:
                    audio_fp = generate_audio_summary(act_summary, act_lang)
                    st.session_state["cached_audio"] = audio_fp.getvalue()
                    st.session_state["cached_audio_id"] = current_summary_id
                except Exception as exc:
                    st.error(f"❌ Audio Generation Error: {exc}")

        if st.session_state.get("cached_audio") and st.session_state.get("cached_audio_id") == current_summary_id:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.audio(st.session_state["cached_audio"], format="audio/mp3")
            audio_file_name = f"audio_{act_format.lower().replace(' ', '_')}_{act_lang.split()[0].lower()}.mp3"
            st.download_button(
                label="⬇️ Download Audio Brief (.mp3)",
                data=st.session_state["cached_audio"],
                file_name=audio_file_name,
                mime="audio/mp3",
                use_container_width=True,
            )

    # -----------------------------------------------------------------------
    # Interactive "Chat with Document" Section (Phase 3)
    # -----------------------------------------------------------------------
    st.markdown("---")
    col_chat_title, col_chat_btn = st.columns([3, 1])
    with col_chat_title:
        st.markdown("### 💬 Chat with this Document")
        st.caption("Ask specific follow-up questions, extract details, or clarify any section.")
    with col_chat_btn:
        if st.session_state.get("chat_history"):
            if st.button("🗑️ Reset Chat", help="Clear conversation history for this document"):
                st.session_state["chat_history"] = []
                st.rerun()

    # Render previous conversation history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box for new user query
    user_query = st.chat_input("Ask any question about this document...")
    if user_query:
        if not api_key:
            st.error("Please ensure your API key is configured to use the Q&A Assistant.")
        else:
            # Add user message to state and render
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate assistant answer grounded in document
            with st.chat_message("assistant"):
                with st.spinner("🔍 Reviewing document and generating answer..."):
                    try:
                        answer, ans_model = answer_document_question(
                            document_text=st.session_state["active_doc_text"],
                            question=user_query,
                            chat_history=st.session_state["chat_history"][:-1],
                            api_key=api_key,
                            provider=provider,
                            model_name=model_name,
                        )
                        st.markdown(answer)
                        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    except Exception as exc:
                        st.error(f"❌ Q&A Error: {exc}")
