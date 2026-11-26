"""
Multimodal RAG Document Intelligence Assistant
==============================================
One Streamlit app (frontend + backend) that turns many kinds of files into ONE
searchable text index, then answers questions about them with Groq.

THE BIG IDEA
------------
Every modality is converted into plain text "content items" that carry metadata:

    PDF text ........ PyMuPDF text, page by page
    Scanned PDF ..... page rendered to an image -> Tesseract OCR text
    PDF images ...... extracted with PyMuPDF -> Gemini Vision description (text)
    Images .......... Gemini Vision description (text), OCR as a fallback
    Tables .......... pdfplumber / pandas -> "col | col | col" text rows
    Audio ........... Faster-Whisper transcript (text) with timestamps
    Video ........... sampled frames -> Gemini Vision descriptions (text)
                      + extracted audio -> Faster-Whisper transcript (text)

After that, ONE pipeline handles everything:

    content items -> chunks -> embeddings -> FAISS -> top-5 retrieval -> Groq

Each chunk keeps its metadata (filename, type, page, table id, timestamp,
image reference), so every answer can point back to its original source.

Run with:  streamlit run app.py
"""

import io
import os
import re
import subprocess
import tempfile

import cv2
import faiss
import numpy as np
import pandas as pd
import pdfplumber
import pymupdf  # PyMuPDF
import pytesseract
import streamlit as st
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from PIL import Image
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration (everything can be changed through the .env file)
# -----------------------------------------------------------------------------
load_dotenv()


def env_int(name, default):
    """Read an integer setting from the environment, falling back to a default."""
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODELS = os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash").split(",")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800                 # characters per chunk
CHUNK_OVERLAP = 150              # characters shared by neighbouring chunks
TOP_K = 5                        # chunks retrieved per question
MIN_TEXT_CHARS = 30              # PDF page with less text than this is treated as scanned
MIN_IMAGE_SIDE = 100             # ignore tiny PDF images (icons, bullets, lines)
MAX_PDF_IMAGES = env_int("MAX_PDF_IMAGES", 15)          # Gemini calls per PDF
MAX_TABLE_ROWS = env_int("MAX_TABLE_ROWS", 2000)        # rows indexed per table
MAX_FILE_MB = env_int("MAX_FILE_MB", 100)               # reject files bigger than this
VIDEO_FRAME_INTERVAL = env_int("VIDEO_FRAME_INTERVAL", 10)  # seconds between frames
MAX_VIDEO_FRAMES = env_int("MAX_VIDEO_FRAMES", 12)      # Gemini calls per video
AUDIO_WINDOW_SECONDS = 30        # transcript segments are grouped into ~30 s windows

DOC_MODE = "Multimodal Document Mode"
GENERAL_MODE = "General AI Mode"

SUPPORTED_TYPES = ["pdf", "txt", "png", "jpg", "jpeg", "csv", "xlsx",
                   "mp3", "wav", "m4a", "mp4", "mov"]
IMAGE_EXTS = {"png", "jpg", "jpeg"}
AUDIO_EXTS = {"mp3", "wav", "m4a"}
VIDEO_EXTS = {"mp4", "mov"}
SHEET_EXTS = {"csv", "xlsx"}

TYPE_LABELS = {
    "pdf_text": "PDF text", "pdf_ocr": "PDF OCR (scanned page)",
    "pdf_image": "PDF image", "pdf_table": "PDF table", "table": "Table",
    "image": "Image", "txt": "Text file", "audio": "Audio transcript",
    "video_frame": "Video frame", "video_audio": "Video audio transcript",
}


# -----------------------------------------------------------------------------
# Small helpers: API keys, content items, formatting
# -----------------------------------------------------------------------------
def key_is_set(name):
    """True if an API key is present (the key itself is never shown or logged)."""
    value = os.getenv(name, "").strip()
    return bool(value) and not value.startswith("your_")


def make_item(text, source, content_type, page=None, table_id=None,
              start=None, end=None, image_ref=None):
    """A 'content item' = some text plus metadata describing where it came from."""
    return {
        "text": text, "source": source, "content_type": content_type,
        "page": page, "table_id": table_id,
        "start": start, "end": end, "image_ref": image_ref,
    }


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def source_label(meta):
    """Human-readable description of where a chunk came from."""
    parts = [meta["source"], TYPE_LABELS.get(meta["content_type"], meta["content_type"])]
    if meta.get("page"):
        parts.append(f"page {meta['page']}")
    if meta.get("table_id"):
        parts.append(f"table {meta['table_id']}")
    if meta.get("start") is not None:
        if meta.get("end") is not None and int(meta["end"]) != int(meta["start"]):
            parts.append(f"{fmt_time(meta['start'])}-{fmt_time(meta['end'])}")
        else:
            parts.append(f"at {fmt_time(meta['start'])}")
    return " | ".join(parts)


def unique_columns(columns):
    """Make column names unique and non-empty (needed for display)."""
    seen, result = {}, []
    for i, col in enumerate(columns):
        name = str(col).strip() or f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        result.append(name)
    return result


def to_png_bytes(pil_image):
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Gemini Vision (images, charts, diagrams, video frames)
# -----------------------------------------------------------------------------
def get_gemini_client():
    if not key_is_set("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to the .env file.")
    from google import genai  # imported here so the app starts even without the SDK
    from google.genai import types
    return genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options=types.HttpOptions(
            timeout=30000,
            retry_options=types.HttpRetryOptions(attempts=2),
        ),
    )


def describe_image(image_bytes, mime_type, context=""):
    """Ask Gemini Vision for a detailed, searchable description of an image."""
    from google.genai import types

    prompt = (
        "You are helping build a search index. Describe this image in detail so "
        "someone could find it by asking questions. Cover: objects, people, and "
        "other visual elements; ALL visible text (transcribe it); and important "
        "relationships and information. If it is a chart or graph, give its type, "
        "axes, categories, approximate values, the highest and lowest items, and "
        "the overall trend. If it is a diagram or flowchart, list its components "
        "and explain how they connect, step by step. Only state what you can see; "
        "do not guess. " + context
    )
    client = get_gemini_client()
    models = list(dict.fromkeys([GEMINI_MODEL] + [
        model.strip() for model in GEMINI_FALLBACK_MODELS if model.strip()]))
    failures = []

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
            )
            text = (response.text or "").strip()
            if text:
                return text
            failures.append(f"{model}: empty response")
        except Exception as error:
            code = getattr(error, "code", None) or getattr(error, "status_code", None)
            failures.append(f"{model}: {error}")
            # Credentials and request errors will not improve by switching models.
            if code in (400, 401, 403):
                break

    raise RuntimeError("Gemini API error: " + "; ".join(failures))

# -----------------------------------------------------------------------------
# OCR (scanned pages and image fallback)
# -----------------------------------------------------------------------------
def perform_ocr(pil_image):
    """Run Tesseract OCR on a PIL image and return the text."""
    try:
        return pytesseract.image_to_string(pil_image).strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract is not installed or not on PATH (see README).")
    except Exception as error:
        raise RuntimeError(f"OCR failed: {error}")


def page_to_image(page):
    """Render a PDF page to a PIL image (used for OCR)."""
    pix = page.get_pixmap(dpi=200)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# -----------------------------------------------------------------------------
# A. PDF text  (+ E. scanned-page detection and OCR)
# -----------------------------------------------------------------------------
def extract_pdf_text(doc, filename):
    """Extract text page by page. Returns (items, pages_with_little_text, warnings)."""
    items, low_text_pages, warnings = [], [], []
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        try:
            text = page.get_text("text").strip()
        except Exception as error:
            warnings.append(f"{filename} page {page_number}: text extraction failed ({error})")
            low_text_pages.append(page_number)
            continue
        if len(text) < MIN_TEXT_CHARS:
            low_text_pages.append(page_number)  # probably scanned or image-only
        else:
            items.append(make_item(text, filename, "pdf_text", page=page_number))
    return items, low_text_pages, warnings


def ocr_pdf_pages(doc, filename, page_numbers):
    """OCR the given pages. Returns (items, pages_that_gave_text, warnings)."""
    items, ok_pages, warnings = [], set(), []
    for page_number in page_numbers:
        try:
            text = perform_ocr(page_to_image(doc[page_number - 1]))
        except RuntimeError as error:
            warnings.append(f"{filename} page {page_number}: {error}")
            if "not installed" in str(error):
                break  # no point retrying every page
            continue
        except Exception as error:
            warnings.append(f"{filename} page {page_number}: could not render page ({error})")
            continue
        if text.strip():  # short OCR results can still contain important values
            items.append(make_item(text, filename, "pdf_ocr", page=page_number))
            ok_pages.add(page_number)
    return items, ok_pages, warnings


# -----------------------------------------------------------------------------
# B. PDF images -> Gemini Vision
# -----------------------------------------------------------------------------
def extract_pdf_images(doc, filename, skip_pages, image_store):
    """Extract embedded images, describe each with Gemini, return (items, warnings)."""
    items, warnings = [], []
    if not key_is_set("GOOGLE_API_KEY"):
        has_images = any(page.get_images() for page in doc)
        if has_images:
            warnings.append(f"{filename}: has images but GOOGLE_API_KEY is missing, "
                            "so images were not analysed.")
        return items, warnings

    seen_xrefs, count = set(), 0
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        if page_number in skip_pages:  # already handled by OCR
            continue
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pil_image = Image.open(io.BytesIO(doc.extract_image(xref)["image"]))
                if min(pil_image.size) < MIN_IMAGE_SIDE:
                    continue  # skip icons and decorations
                png_bytes = to_png_bytes(pil_image)
            except Exception as error:
                warnings.append(f"{filename} page {page_number}: image could not be read ({error})")
                continue
            if count >= MAX_PDF_IMAGES:
                warnings.append(f"{filename}: only the first {MAX_PDF_IMAGES} images were analysed.")
                return items, warnings
            count += 1
            try:
                description = describe_image(
                    png_bytes, "image/png",
                    f"This image comes from page {page_number} of the PDF '{filename}'.")
            except RuntimeError as error:
                warnings.append(f"{filename} page {page_number}: {error}")
                continue
            image_ref = f"{filename}-p{page_number}-img{count}"
            image_store[image_ref] = png_bytes  # keep the picture for the frontend
            items.append(make_item(description, filename, "pdf_image",
                                   page=page_number, image_ref=image_ref))
    return items, warnings


# -----------------------------------------------------------------------------
# C. Uploaded images
# -----------------------------------------------------------------------------
def process_image(file_bytes, filename, extension, image_store):
    """Validate an uploaded PNG/JPG/JPEG and index a Gemini description."""
    items, warnings = [], []
    extension = extension.lower().lstrip(".")
    if extension not in IMAGE_EXTS:
        return items, [f"{filename}: unsupported image extension '.{extension}'."]
    if not file_bytes:
        return items, [f"{filename}: the image file is empty."]

    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image.load()
        actual_format = (pil_image.format or "").upper()
        if actual_format not in {"PNG", "JPEG"}:
            return items, [f"{filename}: expected a PNG or JPEG image, found {actual_format or 'unknown format'}."]
        # Normalize bytes and MIME type to PNG so they always agree.
        image_bytes = to_png_bytes(pil_image)
    except Exception as error:
        return items, [f"{filename}: not a readable PNG/JPG image ({error})"]

    try:
        description = describe_image(
            image_bytes, "image/png", f"The uploaded image file is '{filename}'.")
    except RuntimeError as error:
        warnings.append(f"{filename}: visual analysis failed: {error}")
        try:
            ocr_text = perform_ocr(pil_image)
            if ocr_text:
                description = (
                    "Visual description unavailable; OCR fallback detected this text: "
                    f"{ocr_text}")
            else:
                description = ""
        except RuntimeError as ocr_error:
            warnings.append(f"{filename}: OCR fallback failed: {ocr_error}")
            description = ""

    if description.strip():
        image_ref = f"{filename}-img"
        image_store[image_ref] = image_bytes
        items.append(make_item(description.strip(), filename, "image", image_ref=image_ref))
    else:
        warnings.append(
            f"{filename}: no searchable description or OCR text was produced. "
            "Check the Google API key, Gemini model access, and image contents.")
    return items, warnings

# -----------------------------------------------------------------------------
# D. Tables (pdfplumber for PDFs, pandas for CSV / XLSX)
# -----------------------------------------------------------------------------
def table_to_text(dataframe, table_id, source, page=None):
    """Turn a DataFrame into readable text that keeps the table structure.

    Line 1 describes the table, line 2 holds the column headers, then one line per row:
        Product | Price | Quantity
        Laptop | 50000 | 10
    """
    where = f" (page {page})" if page else ""
    description = (f"Table {table_id} from {source}{where}: {len(dataframe)} rows, "
                   f"columns: {', '.join(dataframe.columns)}")
    lines = [description, " | ".join(dataframe.columns)]
    for row in dataframe.itertuples(index=False):
        lines.append(" | ".join("" if pd.isna(v) else str(v) for v in row))
    return "\n".join(lines)


def dataframe_to_item(dataframe, table_id, source, content_type, table_store, page=None):
    dataframe = dataframe.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if dataframe.empty:
        return None
    dataframe.columns = unique_columns(dataframe.columns)
    table_store[table_id] = dataframe  # kept for the table preview in the frontend
    text = table_to_text(dataframe.head(MAX_TABLE_ROWS), table_id, source, page)
    return make_item(text, source, content_type, page=page, table_id=table_id)


def extract_tables(file_bytes, filename, extension, table_store):
    """Extract tables from a PDF, CSV or XLSX. Returns (items, warnings)."""
    items, warnings = [], []
    try:
        if extension == "pdf":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    for table_index, rows in enumerate(page.extract_tables(), start=1):
                        rows = [[("" if c is None else str(c).strip()) for c in r] for r in rows]
                        rows = [r for r in rows if any(r)]
                        if len(rows) < 2 or len(rows[0]) < 2:
                            continue
                        table_id = f"{filename}-p{page_index + 1}-t{table_index}"
                        dataframe = pd.DataFrame(rows[1:], columns=unique_columns(rows[0]))
                        item = dataframe_to_item(dataframe, table_id, filename,
                                                 "pdf_table", table_store, page_index + 1)
                        if item:
                            items.append(item)
        elif extension == "csv":
            try:
                dataframe = pd.read_csv(io.BytesIO(file_bytes))
            except UnicodeDecodeError:
                dataframe = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
            item = dataframe_to_item(dataframe, f"{filename}-t1", filename, "table", table_store)
            if item:
                items.append(item)
        elif extension == "xlsx":
            sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine="openpyxl")
            for number, (sheet_name, dataframe) in enumerate(sheets.items(), start=1):
                item = dataframe_to_item(dataframe, f"{filename}-{sheet_name}", filename,
                                         "table", table_store)
                if item:
                    items.append(item)
    except Exception as error:
        warnings.append(f"{filename}: table extraction failed ({error})")
    for table_id, table in table_store.items():
        if table_id.startswith(filename) and len(table) > MAX_TABLE_ROWS:
            warnings.append(f"{table_id}: only the first {MAX_TABLE_ROWS} rows were indexed.")
    return items, warnings


# -----------------------------------------------------------------------------
# F. Audio (Faster-Whisper)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading speech-to-text model...")
def load_whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def decode_audio_for_whisper(path, sampling_rate=16000):
    """Decode audio with PyAV while supporting PyAV versions without metadata_errors."""
    import av
    from faster_whisper.audio import (
        _group_frames, _ignore_invalid_frames, _resample_frames,
    )

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=sampling_rate)
    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(path, mode="r") as container:
        frames = _ignore_invalid_frames(container.decode(audio=0))
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    del resampler
    if dtype is None:
        return np.array([], dtype=np.float32)
    return np.frombuffer(raw_buffer.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0


def transcribe_path(path, filename, content_type):
    """Transcribe an audio file on disk into timestamped ~30 second windows."""
    try:
        model = load_whisper_model()
        try:
            segments, _info = model.transcribe(path)
        except TypeError as error:
            if "metadata_errors" not in str(error):
                raise
            # faster-whisper 1.2.1 passes metadata_errors to av.open(), but
            # PyAV 19 removed that keyword. Decode locally and pass a waveform.
            audio = decode_audio_for_whisper(path)
            segments, _info = model.transcribe(audio)
        segments = list(segments)
    except Exception as error:
        raise RuntimeError(f"Transcription failed: {error}")

    items, window_text, window_start, window_end = [], [], None, None
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if window_start is None:
            window_start = segment.start
        window_text.append(text)
        window_end = segment.end
        if window_end - window_start >= AUDIO_WINDOW_SECONDS:
            items.append(make_item(" ".join(window_text), filename, content_type,
                                   start=window_start, end=window_end))
            window_text, window_start = [], None
    if window_text:
        items.append(make_item(" ".join(window_text), filename, content_type,
                               start=window_start, end=window_end))
    return items


def transcribe_audio(file_bytes, filename, extension):
    """Transcribe an uploaded MP3 / WAV / M4A file. Returns (items, warnings)."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, f"audio.{extension}")
        with open(path, "wb") as f:
            f.write(file_bytes)
        try:
            items = transcribe_path(path, filename, "audio")
        except RuntimeError as error:
            return [], [f"{filename}: {error}"]
    if not items:
        return [], [f"{filename}: no speech was detected."]
    return items, []


# -----------------------------------------------------------------------------
# G. Video (OpenCV frames + FFmpeg audio)
# -----------------------------------------------------------------------------
def extract_video_frames(video_path):
    """Sample frames every VIDEO_FRAME_INTERVAL seconds (at most MAX_VIDEO_FRAMES).

    Near-identical frames are skipped. Returns a list of (seconds, jpeg_bytes).
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("the video could not be opened (unsupported or corrupted file)")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("the video has no readable frames")
    duration = total_frames / fps

    times = list(np.arange(0, duration, VIDEO_FRAME_INTERVAL))
    if len(times) > MAX_VIDEO_FRAMES:  # avoid extracting too many frames
        picks = np.linspace(0, len(times) - 1, MAX_VIDEO_FRAMES).astype(int)
        times = [times[i] for i in picks]

    frames, previous_small = [], None
    for seconds in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32))
        if previous_small is not None and np.mean(cv2.absdiff(small, previous_small)) < 3:
            continue  # same scene as the last kept frame
        previous_small = small
        ok, jpeg = cv2.imencode(".jpg", frame)
        if ok:
            frames.append((float(seconds), jpeg.tobytes()))
    capture.release()
    return frames


def extract_video_audio(video_path, wav_path):
    """Use FFmpeg to pull the audio track out of a video. Returns True if audio exists."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True, timeout=900)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not on PATH (see README)")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out while extracting audio")
    return result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 2000


def process_video(file_bytes, filename, extension, image_store):
    """Frames -> Gemini descriptions, audio -> Whisper transcript. Returns (items, warnings)."""
    items, warnings = [], []
    with tempfile.TemporaryDirectory() as folder:
        video_path = os.path.join(folder, f"video.{extension}")
        with open(video_path, "wb") as f:
            f.write(file_bytes)

        # Visual content: representative frames described by Gemini
        try:
            frames = extract_video_frames(video_path)
        except RuntimeError as error:
            return [], [f"{filename}: {error}"]
        if not key_is_set("GOOGLE_API_KEY"):
            warnings.append(f"{filename}: GOOGLE_API_KEY is missing, so frames were not analysed.")
        else:
            for seconds, jpeg_bytes in frames:
                try:
                    description = describe_image(
                        jpeg_bytes, "image/jpeg",
                        f"This is a frame from the video '{filename}' at {fmt_time(seconds)}.")
                except RuntimeError as error:
                    warnings.append(f"{filename} at {fmt_time(seconds)}: {error}")
                    continue
                image_ref = f"{filename}-frame-{int(seconds)}s"
                image_store[image_ref] = jpeg_bytes
                items.append(make_item(description, filename, "video_frame",
                                       start=seconds, end=seconds, image_ref=image_ref))

        # Spoken content: audio track -> Whisper
        wav_path = os.path.join(folder, "audio.wav")
        try:
            if extract_video_audio(video_path, wav_path):
                items.extend(transcribe_path(wav_path, filename, "video_audio"))
        except RuntimeError as error:
            warnings.append(f"{filename}: {error}")
    return items, warnings


# -----------------------------------------------------------------------------
# File router: decides which extraction functions to run for each upload
# -----------------------------------------------------------------------------
def process_pdf(file_bytes, filename, image_store, table_store):
    items, warnings = [], []
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as error:
        return [], [f"{filename}: corrupted or unreadable PDF ({error})"]
    with doc:
        if doc.needs_pass:
            return [], [f"{filename}: the PDF is password protected."]
        if doc.page_count == 0:
            return [], [f"{filename}: the PDF has no pages."]

        text_items, low_text_pages, warn = extract_pdf_text(doc, filename)
        items += text_items
        warnings += warn

        ocr_items, ocr_ok_pages, warn = ocr_pdf_pages(doc, filename, low_text_pages)
        items += ocr_items
        warnings += warn

        image_items, warn = extract_pdf_images(doc, filename, set(), image_store)
        items += image_items
        warnings += warn

    table_items, warn = extract_tables(file_bytes, filename, "pdf", table_store)
    items += table_items
    warnings += warn

    if not items:
        warnings.append(f"{filename}: no content could be extracted (empty document?).")
    return items, warnings


def process_file(uploaded_file, image_store, table_store):
    """Run the right extraction for one uploaded file. Returns (items, warnings)."""
    filename = uploaded_file.name
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    file_bytes = uploaded_file.getvalue()

    if extension not in SUPPORTED_TYPES:
        return [], [f"{filename}: unsupported file format '.{extension}'."]
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return [], [f"{filename}: larger than {MAX_FILE_MB} MB, skipped."]
    if not file_bytes:
        return [], [f"{filename}: the file is empty."]

    if extension == "pdf":
        return process_pdf(file_bytes, filename, image_store, table_store)
    if extension == "txt":
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        if not text:
            return [], [f"{filename}: the text file is empty."]
        return [make_item(text, filename, "txt")], []
    if extension in IMAGE_EXTS:
        return process_image(file_bytes, filename, extension, image_store)
    if extension in SHEET_EXTS:
        items, warnings = extract_tables(file_bytes, filename, extension, table_store)
        if not items and not warnings:
            warnings.append(f"{filename}: no data found.")
        return items, warnings
    if extension in AUDIO_EXTS:
        return transcribe_audio(file_bytes, filename, extension)
    return process_video(file_bytes, filename, extension, image_store)


# -----------------------------------------------------------------------------
# UNIFIED RAG PIPELINE
# -----------------------------------------------------------------------------
def convert_to_text(meta, body):
    """Common text representation: a metadata header line + the content itself.

    The header is embedded together with the content, so a question such as
    "what is on page 5" or "in the spreadsheet" also matches by source.
    """
    return f"[{source_label(meta)}]\n{body}"


def split_into_chunks(items):
    """Split content items into chunks. Every chunk keeps its item's metadata."""
    chunks = []
    for item in items:
        meta = {k: v for k, v in item.items() if k != "text"}
        body = item["text"]

        if item["content_type"] in ("table", "pdf_table"):
            # Split tables by rows and repeat the 2 header lines in every chunk
            lines = body.split("\n")
            header, rows = "\n".join(lines[:2]), lines[2:]
            current, size = [], 0
            for row in rows:
                if current and size + len(row) > CHUNK_SIZE:
                    pieces_text = header + "\n" + "\n".join(current)
                    chunks.append({**meta, "text": pieces_text})
                    current, size = [], 0
                current.append(row)
                size += len(row) + 1
            if current or not rows:
                chunks.append({**meta, "text": header + "\n" + "\n".join(current)})
            continue

        body = re.sub(r"[ \t]+", " ", body).strip()
        step = CHUNK_SIZE - CHUNK_OVERLAP
        start = 0
        while start < len(body):
            end = min(start + CHUNK_SIZE, len(body))
            if end < len(body):  # prefer to cut at a space, not inside a word
                space = body.rfind(" ", start + step // 2, end)
                end = space if space != -1 else end
            piece = body[start:end].strip()
            if piece:
                chunks.append({**meta, "text": piece})
            if end >= len(body):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    for chunk in chunks:
        chunk["search_text"] = convert_to_text(chunk, chunk["text"])
    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain adapter that preserves the existing embedding model/settings."""

    @staticmethod
    def _normalise(vectors):
        values = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    def embed_documents(self, texts):
        model = load_embedding_model()
        vectors = model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return self._normalise(vectors).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def create_vector_store(chunks):
    """Create a LangChain FAISS store using the original cosine-search behavior."""
    embedding = SentenceTransformerEmbeddings()
    documents = [
        Document(page_content=chunk["search_text"], metadata=chunk)
        for chunk in chunks
    ]
    vectors = embedding.embed_documents([document.page_content for document in documents])
    index = faiss.IndexFlatIP(len(vectors[0]))
    vector_store = LangChainFAISS(
        embedding_function=embedding,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_embeddings(
        [(document.page_content, vector)
         for document, vector in zip(documents, vectors)],
        metadatas=[document.metadata for document in documents],
    )
    return vector_store


def retrieve_relevant_chunks(question, index, chunks, top_k=TOP_K):
    """Retrieve the top_k chunks through LangChain's FAISS vector store."""
    results = []
    for document, score in index.similarity_search_with_score(
        question, k=min(top_k, index.index.ntotal)):
        results.append({**document.metadata, "score": float(score)})

    # Small helper for questions like "explain page 5" or "what happens at 30 seconds":
    # semantic search alone cannot understand numbers well, so pin matching chunks.
    pinned = []
    page_match = re.search(r"\bpage\s+(\d+)", question, re.IGNORECASE)
    if page_match:
        page = int(page_match.group(1))
        pinned += [c for c in chunks if c.get("page") == page][:3]
    time_match = re.search(r"\b(\d+)\s*[- ]?\s*(?:seconds?|secs?)\b", question, re.IGNORECASE)
    if time_match:
        target = int(time_match.group(1))
        timed = [c for c in chunks if c.get("start") is not None]
        timed.sort(key=lambda c: 0 if c["start"] <= target <= (c["end"] or c["start"])
                   else min(abs(target - c["start"]), abs(target - (c["end"] or c["start"]))))
        pinned += [c for c in timed[:2]
                   if abs(target - c["start"]) <= max(VIDEO_FRAME_INTERVAL, AUDIO_WINDOW_SECONDS)]
    if pinned:
        merged = []
        for chunk in pinned + results:
            if not any(chunk["search_text"] == m["search_text"] for m in merged):
                merged.append({"score": 0.0, **chunk})
        results = merged[:max(top_k, len(pinned))]
    return results


# -----------------------------------------------------------------------------
# Groq answer generation
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Groq model...")
def get_groq_model():
    if not key_is_set("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing. Add it to the .env file and restart.")
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


def generate_answer(question, retrieved_chunks=None, history=None):
    """Answer a question with Groq.

    Document mode: retrieved_chunks is a list and the answer must come from it.
    General mode:  retrieved_chunks is None and Groq answers directly.
    """
    model = get_groq_model()

    if retrieved_chunks is None:
        messages = [SystemMessage(content="You are a helpful AI assistant.")]
        for message in (history or [])[-6:]:
            message_type = HumanMessage if message["role"] == "user" else AIMessage
            messages.append(message_type(content=message["content"]))
        messages.append(HumanMessage(content=question))
    else:
        context = "\n\n".join(
            f"[{i}] {source_label(chunk)}\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks, start=1))
        system_prompt = (
            "You answer questions about the user's uploaded files (documents, images, "
            "tables, audio and video). Use ONLY the numbered context passages. "
            "Image and video-frame passages are AI-written descriptions. "
            "If the context does not contain enough information, say exactly: "
            "'The uploaded sources do not contain sufficient information to answer this question.' "
            "Never invent facts or numbers; when you calculate something from a table, "
            "use only the rows shown. Mention the passage numbers you used, like [1] or [2].")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

    try:
        response = model.invoke(messages)
        return response.content
    except Exception as error:
        raise RuntimeError(f"Groq API error: {error}")


# -----------------------------------------------------------------------------
# Processing all uploads
# -----------------------------------------------------------------------------
def process_files(uploaded_files, progress):
    """Extract -> chunk -> embed -> index. Returns a dict that is saved in session state."""
    image_store, table_store, items, warnings = {}, {}, [], []
    for number, uploaded_file in enumerate(uploaded_files, start=1):
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
        if extension in IMAGE_EXTS:
            progress_text = f"Validating and analyzing image with Gemini: {uploaded_file.name}"
        elif extension == "pdf":
            progress_text = f"Extracting PDF text, OCR, images, and tables: {uploaded_file.name}"
        else:
            progress_text = f"Processing {uploaded_file.name} ({number}/{len(uploaded_files)})"
        progress.progress((number - 1) / len(uploaded_files), text=progress_text)
        try:
            file_items, file_warnings = process_file(uploaded_file, image_store, table_store)
        except Exception as error:  # last-resort guard so one bad file never stops the rest
            file_items, file_warnings = [], [f"{uploaded_file.name}: unexpected error ({error})"]
        items += file_items
        warnings += file_warnings

    if not items:
        detail = " ".join(warnings[:3])
        error = "No searchable content could be extracted."
        if detail:
            error += f" Processing details: {detail}"
        return {"error": error, "warnings": warnings}

    progress.progress(0.95, text="Creating embeddings and FAISS index...")
    chunks = split_into_chunks(items)
    try:
        index = create_vector_store(chunks)
    except Exception as error:
        return {"error": f"Embedding / FAISS indexing failed: {error}", "warnings": warnings}
    progress.progress(1.0, text="Done")

    counts = {}
    for item in items:
        label = TYPE_LABELS.get(item["content_type"], item["content_type"])
        counts[label] = counts.get(label, 0) + 1
    return {"index": index, "chunks": chunks, "image_store": image_store,
            "table_store": table_store, "counts": counts, "warnings": warnings,
            "num_files": len(uploaded_files)}


# -----------------------------------------------------------------------------
# Streamlit frontend
# -----------------------------------------------------------------------------
def show_retrieved(chunks, image_store, table_store):
    """Retrieved context, source list, image previews and table previews."""
    with st.expander("Retrieved context"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{i}] {source_label(chunk)}** (similarity {chunk['score']:.2f})")
            st.text(chunk["text"])

    labels = []
    for chunk in chunks:
        label = source_label(chunk)
        if label not in labels:
            labels.append(label)
    st.caption("Sources: " + "  •  ".join(labels))

    shown_images = []
    for chunk in chunks:
        ref = chunk.get("image_ref")
        if ref and ref in image_store and ref not in shown_images:
            shown_images.append(ref)
            st.image(image_store[ref], caption=source_label(chunk), width=320)

    shown_tables = []
    for chunk in chunks:
        table_id = chunk.get("table_id")
        if table_id and table_id in table_store and table_id not in shown_tables:
            shown_tables.append(table_id)
            st.markdown(f"**Table preview: {table_id}**")
            st.dataframe(table_store[table_id].head(50), width="stretch")


def show_image_descriptions(chunks, image_store):
    """Show indexed descriptions and previews for uploaded/PDF images."""
    image_chunks = [chunk for chunk in chunks
                    if chunk.get("content_type") in {"image", "pdf_image"}]
    if not image_chunks:
        return
    with st.expander(f"Generated image descriptions ({len(image_chunks)})"):
        for chunk in image_chunks:
            st.markdown(f"**{source_label(chunk)}**")
            image_ref = chunk.get("image_ref")
            if image_ref and image_ref in image_store:
                st.image(image_store[image_ref], width=320)
            st.write(chunk["text"])

def main():
    st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠", layout="wide")

    defaults = {"index": None, "chunks": [], "image_store": {}, "table_store": {},
                "counts": {}, "messages": [], "num_files": 0, "status": "Waiting for files",
                "warnings": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # ---- Sidebar ----
    with st.sidebar:
        st.title("🧠 Multimodal RAG")
        mode = st.radio("Mode", [DOC_MODE, GENERAL_MODE])

        uploaded_files = st.file_uploader(
            "Upload files", type=SUPPORTED_TYPES, accept_multiple_files=True)
        st.caption("Supported: PDF, TXT, PNG, JPG, JPEG, CSV, XLSX, MP3, WAV, M4A, MP4, MOV")

        if st.button("Process Files", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
            else:
                progress = st.progress(0.0, text="Starting...")
                st.session_state.status = "Processing..."
                result = process_files(uploaded_files, progress)
                progress.empty()
                st.session_state.warnings = result.get("warnings", [])
                if "error" in result:
                    st.session_state.status = "Processing failed"
                    st.error(result["error"])
                else:
                    for key in ("index", "chunks", "image_store", "table_store",
                                "counts", "num_files"):
                        st.session_state[key] = result[key]
                    st.session_state.messages = []
                    st.session_state.status = "Files processed successfully"

        st.metric("Files processed", st.session_state.num_files)
        st.info(f"Status: {st.session_state.status}")
        if st.session_state.counts:
            st.markdown("**Extracted content**")
            for label, count in st.session_state.counts.items():
                st.write(f"- {label}: {count}")
            st.write(f"- Chunks indexed: {len(st.session_state.chunks)}")
        if st.session_state.warnings:
            with st.expander(f"Warnings ({len(st.session_state.warnings)})"):
                for warning in st.session_state.warnings:
                    st.warning(warning)

        st.caption(f"Groq key: {'set' if key_is_set('GROQ_API_KEY') else 'MISSING'} | "
                   f"Gemini key: {'set' if key_is_set('GOOGLE_API_KEY') else 'missing'}")

    # ---- Main page ----
    st.title("Multimodal RAG Document Intelligence Assistant")
    if mode == DOC_MODE:
        st.write("Upload PDFs, images, tables, audio or video, click **Process Files**, then "
                 "ask questions. Answers are built from the retrieved content of your files.")
    else:
        st.write("General AI Mode: your questions go straight to Groq, with no document retrieval.")
    show_image_descriptions(st.session_state.chunks, st.session_state.image_store)
    if not key_is_set("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it to the .env file and restart the app.")

    for message in st.session_state.messages:
        if message["mode"] != mode:
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                show_retrieved(message["sources"], st.session_state.image_store,
                               st.session_state.table_store)

    question = st.chat_input("Ask a question")
    if question is None:
        return
    if not question.strip():
        st.warning("Please enter a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question, "mode": mode})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            if mode == GENERAL_MODE:
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.messages[:-1] if m["mode"] == GENERAL_MODE]
                with st.spinner("Asking Groq..."):
                    answer = generate_answer(question, None, history)
                retrieved = []
            else:
                if st.session_state.index is None:
                    raise ValueError("Please upload files and click 'Process Files' first.")
                with st.spinner("Searching your files and generating an answer..."):
                    retrieved = retrieve_relevant_chunks(
                        question, st.session_state.index, st.session_state.chunks)
                    if not retrieved:
                        raise ValueError("No relevant content was found in the uploaded files.")
                    answer = generate_answer(question, retrieved)
        except Exception as error:
            st.error(str(error))
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {error}", "mode": mode})
            return

        st.write(answer)
        if retrieved:
            show_retrieved(retrieved, st.session_state.image_store, st.session_state.table_store)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": retrieved, "mode": mode})


if __name__ == "__main__":
    main()
