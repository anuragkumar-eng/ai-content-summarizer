"""
app.py  —  AI-Powered Content Summarizer
Reads API keys automatically from the .env file (no input field in UI).
Supports Groq (Free), Google Gemini (Free), and OpenAI (Paid).
"""

import os
import logging

import streamlit as st
from dotenv import load_dotenv

from document_processor import extract_text_from_pdf, extract_text_from_url
from summarizer import summarize_text, get_available_models, PROVIDERS, DEFAULT_MODELS

# Reload .env whenever the app runs so changes to .env take effect immediately
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Mapping of Providers to .env variable names
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
    "Groq (Free)":          "🟢 FREE — 14,400 requests/day",
    "Google Gemini (Free)": "🟢 FREE — 1,500 requests/day",
    "OpenAI (Paid)":        "🔴 PAID — requires billing",
}

def get_env_api_key(provider_name: str) -> str:
    """Retrieve the API key for the selected provider from environment / .env."""
    env_var = PROVIDER_ENV_VARS.get(provider_name, "")
    val = os.getenv(env_var, "").strip()
    # Support alternate GOOGLE_API_KEY for Gemini if set
    if not val and provider_name == "Google Gemini (Free)":
        val = os.getenv("GOOGLE_API_KEY", "").strip()
    return val

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # ── Provider picker ───────────────────────────────────────────────────────
    provider = st.selectbox(
        "🌐 AI Provider",
        options=PROVIDERS,
        index=0,
        help="Groq and Gemini are completely free with no credit card required.",
    )
    st.caption(FREE_BADGE[provider])

    # ── Automatic .env Key Status (No manual text input shown) ────────────────
    env_var_name = PROVIDER_ENV_VARS.get(provider, "API_KEY")
    api_key = get_env_api_key(provider)

    if api_key:
        st.success(f"🔐 Key loaded from `.env` (`{env_var_name}`)")
    else:
        st.warning(f"⚠️ `{env_var_name}` is empty in `.env`")
        link_label, link_url = KEY_LINKS[provider]
        st.caption(
            f"Paste your key into the `.env` file for `{env_var_name}`. "
            f"[Get free key at {link_label}]({link_url})"
        )

    st.markdown("---")

    # ── Dynamic Model Discovery ───────────────────────────────────────────────
    available_models = get_available_models(provider, api_key)
    default_model = DEFAULT_MODELS.get(provider, available_models[0] if available_models else "")
    default_idx = available_models.index(default_model) if default_model in available_models else 0

    model_name = st.selectbox(
        "🤖 Model",
        options=available_models,
        index=default_idx,
        help="Active models detected for this provider.",
    )

    # ── Output format ─────────────────────────────────────────────────────────
    format_type = st.radio(
        "📋 Output Format",
        options=["Short Summary", "Detailed Bullet Points"],
        index=0,
    )

    st.markdown("---")
    st.caption("Built with LangChain · Streamlit · Free AI APIs")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🧠 AI-Powered Content Summarizer")
st.markdown(
    "Upload a **PDF**, paste a **web URL**, or type **raw text** — "
    "get an AI summary instantly. **100% free** with Groq or Gemini!"
)
st.markdown("---")

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_text, tab_url, tab_pdf = st.tabs(["📝 Paste Text", "🔗 Web URL", "📄 Upload PDF"])

raw_text   = ""
input_label = ""

with tab_text:
    st.subheader("Paste or type your text below")
    text_input = st.text_area(
        label="Input Text",
        placeholder="Paste your article, blog post, research paper…",
        height=300,
        label_visibility="collapsed",
    )
    if text_input.strip():
        raw_text    = text_input.strip()
        input_label = "pasted text"

with tab_url:
    st.subheader("Enter a web article URL")
    url_input = st.text_input(
        label="URL",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
        label_visibility="collapsed",
    )
    if url_input.strip():
        with st.spinner("🌐 Fetching and parsing the page…"):
            try:
                raw_text    = extract_text_from_url(url_input.strip())
                input_label = f"URL: {url_input.strip()}"
                st.success(f"✅ Extracted **{len(raw_text):,}** characters.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"❌ URL Error: {exc}")

with tab_pdf:
    st.subheader("Upload a PDF document")
    uploaded_file = st.file_uploader(
        label="PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        with st.spinner("📄 Extracting text from PDF…"):
            try:
                raw_text    = extract_text_from_pdf(uploaded_file)
                input_label = f"PDF: {uploaded_file.name}"
                st.success(f"✅ Extracted **{len(raw_text):,}** characters from **{uploaded_file.name}**.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"❌ PDF Error: {exc}")

st.markdown("---")

# ── Stats ─────────────────────────────────────────────────────────────────────
if raw_text:
    c1, c2 = st.columns(2)
    c1.metric("📊 Characters",         f"{len(raw_text):,}")
    c2.metric("🔢 ~Tokens (estimate)", f"{len(raw_text)//4:,}")

# ── Summarize button ──────────────────────────────────────────────────────────
st.markdown("")
summarize_btn = st.button(
    "✨ Summarize",
    type="primary",
    use_container_width=True,
    disabled=not bool(raw_text),
)

if summarize_btn:
    if not api_key:
        env_var_name = PROVIDER_ENV_VARS.get(provider, "API_KEY")
        link_label, link_url = KEY_LINKS[provider]
        st.error(
            f"❌ **API Key Missing**: `{env_var_name}` is not set in your `.env` file.\n\n"
            f"Open your `.env` file and paste your key: `{env_var_name}=your_key_here`\n\n"
            f"[Get your free key at {link_label}]({link_url})"
        )
        st.stop()

    with st.spinner(f"🤖 Summarizing with **{provider} / {model_name}**…"):
        try:
            summary, actual_model = summarize_text(
                text=raw_text,
                api_key=api_key,
                format_type=format_type,
                provider=provider,
                model_name=model_name,
            )
        except ValueError as exc:
            st.error(f"❌ Input Error: {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"❌ API Error: {exc}")
            link_label, link_url = KEY_LINKS[provider]
            st.info(f"💡 Check your API key in `.env` is correct at [{link_label}]({link_url})")
            st.stop()

    st.success("🎉 Summary generated successfully!")

    with st.container(border=True):
        st.markdown(f"### 📄 Summary — *{format_type}*")
        if input_label:
            st.caption(f"Source: {input_label}  |  Provider: {provider}  |  Model: {actual_model}")
        st.markdown(summary)

    st.download_button(
        label="⬇️ Download Summary (.txt)",
        data=summary,
        file_name="summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
