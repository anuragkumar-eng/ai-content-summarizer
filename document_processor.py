"""
document_processor.py
---------------------
Handles text extraction from four input sources:
  1. Plain text   – returned as-is.
  2. PDF files    – extracted page-by-page with PyPDF2.
  3. Web URLs     – scraped and cleaned with requests + BeautifulSoup.
  4. YouTube URLs – transcripts fetched directly with youtube-transcript-api.
"""

import io
import re
import logging
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
import PyPDF2

# Configure module-level logger so callers can control verbosity
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_object) -> str:
    """Extract raw text from an uploaded PDF file object.

    Parameters
    ----------
    file_object : file-like object
        The binary stream of the uploaded PDF (e.g. ``st.file_uploader`` result).

    Returns
    -------
    str
        Concatenated text from every page, separated by newlines.

    Raises
    ------
    ValueError
        If the file object is empty or not a valid PDF.
    RuntimeError
        If PyPDF2 encounters a decryption or parsing error.
    """
    if file_object is None:
        raise ValueError("No file object provided.")

    try:
        # Wrap in BytesIO so PyPDF2 can seek; handles both raw bytes and streams.
        raw_bytes = file_object.read()
        if not raw_bytes:
            raise ValueError("The uploaded PDF file is empty.")

        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))

        if reader.is_encrypted:
            raise RuntimeError(
                "The PDF is password-protected. Please upload an unencrypted file."
            )

        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
            logger.debug("Extracted %d chars from page %d", len(page_text), page_num)

        full_text = "\n".join(pages_text).strip()

        if not full_text:
            raise ValueError(
                "Could not extract any text from the PDF. "
                "The file may contain only scanned images."
            )

        return full_text

    except (PyPDF2.errors.PdfReadError, PyPDF2.errors.PdfStreamError) as exc:
        raise RuntimeError(f"Failed to read PDF: {exc}") from exc


# ---------------------------------------------------------------------------
# URL / Web Scraping
# ---------------------------------------------------------------------------

# Tags that typically contain meaningful article content
_CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td"]

# Tags to strip before parsing (ads, navigation, scripts, styles)
_NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "noscript", "iframe", "figure",
]


def extract_text_from_url(url: str) -> str:
    """Scrape and return clean paragraph text from a web URL.

    Uses ``requests`` to fetch the HTML, then ``BeautifulSoup`` to:
      1. Remove boilerplate elements (navigation, ads, scripts, etc.).
      2. Extract text from semantic content tags only.

    Parameters
    ----------
    url : str
        A fully-qualified URL starting with ``http://`` or ``https://``.

    Returns
    -------
    str
        Clean, stripped text from the page.

    Raises
    ------
    ValueError
        If the URL is malformed or the page returned no usable text.
    RuntimeError
        If the HTTP request fails or returns a non-200 status code.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "Invalid URL. Please include the full URL with http:// or https://."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ContentSummarizerBot/1.0; "
            "+https://github.com/summarizer)"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to '{url}'. Please check the URL and your internet connection."
        ) from exc
    except requests.exceptions.Timeout:
        raise RuntimeError(f"The request to '{url}' timed out after 15 seconds.")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error {exc.response.status_code} when fetching '{url}'."
        ) from exc

    # Parse with lxml for speed; fall back to html.parser if unavailable
    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception:
        soup = BeautifulSoup(response.text, "html.parser")

    # ── Remove noise elements ────────────────────────────────────────────────
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # ── Collect content from semantic tags ───────────────────────────────────
    content_parts = []
    for element in soup.find_all(_CONTENT_TAGS):
        text = element.get_text(separator=" ", strip=True)
        if len(text) > 40:  # skip very short fragments (menu items, etc.)
            content_parts.append(text)

    full_text = "\n".join(content_parts).strip()

    if not full_text:
        raise ValueError(
            "No readable text found on the page. "
            "The site may require JavaScript or block automated access."
        )

    logger.debug("Extracted %d chars from URL '%s'", len(full_text), url)
    return full_text


# ---------------------------------------------------------------------------
# YouTube Transcript Extraction
# ---------------------------------------------------------------------------

def extract_youtube_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from various URL formats.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://m.youtube.com/watch?v=VIDEO_ID
      - Raw 11-character video IDs
    """
    if not url:
        return ""
    clean_url = url.strip()

    # Direct 11-character video ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", clean_url):
        return clean_url

    parsed = urlparse(clean_url)
    hostname = (parsed.hostname or "").lower()

    if "youtu.be" in hostname:
        path = parsed.path.strip("/")
        return path.split("/")[0] if path else ""

    if "youtube.com" in hostname:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/v/")):
            parts = parsed.path.strip("/").split("/")
            return parts[1] if len(parts) >= 2 else ""

    # Fallback regex search for any 11-character YouTube ID pattern
    match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", clean_url)
    return match.group(1) if match else ""


def extract_text_from_youtube(url: str) -> tuple[str, str]:
    """Fetch and concatenate transcript text from a YouTube video URL.

    Parameters
    ----------
    url : str
        YouTube video URL or video ID.

    Returns
    -------
    tuple[str, str]
        (full_transcript_text, video_id)

    Raises
    ------
    ValueError
        If URL does not contain a valid YouTube video ID or transcript is empty.
    RuntimeError
        If video subtitles are disabled, unavailable, or an API error occurs.
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(
            "Could not detect a valid YouTube video ID from the provided URL. "
            "Please check the link format (e.g. https://www.youtube.com/watch?v=...)."
        )

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Try to find transcripts in preferred languages or fallback to first available
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Check for English first
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            # Fallback to the first available transcript (original or auto-generated)
            transcript = next(iter(transcript_list))

        data = transcript.fetch()
        full_text = " ".join(item.get("text", "") for item in data if item.get("text")).strip()

        if not full_text:
            raise ValueError("The video's transcript was retrieved but contains no readable text.")

        logger.debug("Extracted %d chars of transcript from YouTube video %s", len(full_text), video_id)
        return full_text, video_id

    except Exception as exc:
        err_msg = str(exc)
        if "TranscriptsDisabled" in err_msg or "Subtitles are disabled" in err_msg:
            raise RuntimeError(
                f"Subtitles/captions are disabled for this YouTube video (ID: {video_id}). "
                "Only videos with enabled subtitles or auto-captions can be summarized."
            )
        elif "NoTranscriptFound" in err_msg:
            raise RuntimeError(
                f"No transcript or captions were found for YouTube video (ID: {video_id})."
            )
        elif "VideoUnavailable" in err_msg:
            raise RuntimeError(
                f"This YouTube video is private, age-restricted, or unavailable (ID: {video_id})."
            )
        else:
            raise RuntimeError(f"Failed to fetch YouTube transcript: {exc}")
