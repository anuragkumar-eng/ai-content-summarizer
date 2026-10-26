# 🧠 AI-Powered Content Summarizer

A modern, full-stack web application built with **Streamlit**, **LangChain**, and **LLMs (Groq, Google Gemini, OpenAI)**. Ingests long-form text, PDF documents, or live web URLs and generates concise, accurate summaries using an optimized Map-Reduce architecture.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- **Multi-Format Ingestion**:
  - 📝 **Pasted Text**: Direct input for articles, transcripts, or notes.
  - 🎥 **YouTube Videos**: Directly extracts subtitles/captions from YouTube URLs with an embedded player.
  - 🔗 **Web URLs**: Scrapes clean article text using `BeautifulSoup4` (strips navigation, ads, and boilerplate).
  - 📄 **PDF Uploads**: Extracts text page-by-page using `PyPDF2`.
- **Multiple AI Providers**:
  - ⚡ **Groq (Free)**: Meta Llama 3.1 8B (~560 tokens/sec, 14,400 requests/day).
  - 🟢 **Google Gemini (Free)**: Gemini 2.5 Flash (1,500 requests/day).
  - 🔴 **OpenAI (Paid)**: GPT-4o-mini, GPT-4o, GPT-3.5-turbo.
- **Smart Summarization Pipeline**:
  - **Single-Pass Optimization**: Text fitting within ~1,200 characters is summarized in a single prompt to save quota and latency.
  - **Map-Reduce Architecture**: Long documents are split into overlapping chunks (1,200 chars, 150 overlap), summarized independently (Map), and synthesized into a final output (Reduce).
  - **Automatic Fallback & Retries**: Seamless retry on transient 503/429 spikes and automatic model fallback if a model is deprecated.
- **Output Customization & Personas**:
  - **6 Specialized Personas**: Executive Brief, Short Summary, Detailed Bullet Points, Action Items & Key Decisions, ELI5, Social & LinkedIn Post.
  - **10+ Output Languages**: Cross-lingual translation into English, Spanish, French, German, Hindi, Japanese, Chinese, etc.
  - One-click **Download (.txt)** and **Raw Markdown** preview.
- **💬 Interactive "Chat with Document"**:
  - Follow-up Q&A assistant embedded directly under the summary.
  - Ask specific questions, clarify facts, or extract data points strictly grounded in the document context.
- **Security**: API keys are securely loaded from a local `.env` file (never exposed in the UI or tracked in Git).

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-content-summarizer.git
cd ai-content-summarizer
```

### 2. Set up a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy the example environment file and add your key:
```bash
cp .env.example .env
```

Open `.env` and paste your key (e.g. Groq free key):
```env
GROQ_API_KEY=your_groq_api_key_here
```
> *Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys) (no credit card required).*

### 5. Launch the Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
ai-content-summarizer/
│
├── app.py                  # Streamlit frontend & UI controls
├── summarizer.py           # LangChain Map-Reduce pipeline & provider logic
├── document_processor.py   # PDF text extraction & web scraping engine
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Ignores .env, .venv, and build caches
└── README.md               # Documentation
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Orchestration**: LangChain (LCEL)
- **Providers**: Groq SDK, LangChain Google GenAI, LangChain OpenAI
- **Document Processing**: PyPDF2, BeautifulSoup4, Requests
- **Environment Management**: python-dotenv

---

## 📄 License
This project is licensed under the MIT License.
