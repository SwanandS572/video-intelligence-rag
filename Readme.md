# 🤖 RAG-Based Video Intelligence System

> An intelligent, video-aware Q&A system that answers natural language questions over video content — returning **exact video numbers and timestamps** using Retrieval-Augmented Generation (RAG).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) -->
[![LLM: Gemini](https://img.shields.io/badge/LLM-Gemini%202.5-green.svg)](https://ai.google.dev)
[![Embeddings: BGE-M3](https://img.shields.io/badge/Embeddings-BGE--M3-orange.svg)](https://ollama.com)

---

## 📌 Problem It Solves

Searching through hours of video lectures to find *where* a specific topic is taught is slow and frustrating. This project solves that by:

- Transcribing all course videos into timestamped text chunks
- Encoding them as semantic vectors (embeddings)
- Accepting a natural language query and retrieving the **most relevant video + timestamp**
- Feeding context to an LLM to generate a precise, guided response

**Example Query:**
> *"Where is CSS overflow property taught?"*

**Response:**
> *"CSS Overflow is covered in Video 2 of the Sigma Web Development Course, starting at 340.0 seconds (approx. 5:40)..."*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Audio Extraction | FFmpeg |
| Speech-to-Text | OpenAI Whisper (base, 140 MB) |
| Embeddings Model | BGE-M3 via Ollama (1024-dim vectors) |
| Vector Storage | Pandas DataFrame + Joblib |
| Similarity Search | Scikit-learn (Cosine Similarity) |
| LLM Response | Gemini 2.5 Flash API / Llama3.2 (local) |
| Language | Python 3.10+ |

---

## 🏗️ Architecture

### Phase 1 — Ingestion Pipeline

| Step | Tool | Output |
|------|------|--------|
| 1. Video files | FFmpeg | `.mp3` audio files |
| 2. Audio files | OpenAI Whisper (`base`) | Timestamped JSON chunks |
| 3. Raw chunks (256) | `merge_chunks.py` | Optimized chunks (52) |

### Phase 2 — Embedding Pipeline

| Step | Tool | Output |
|------|------|--------|
| 4. JSON chunks | BGE-M3 via Ollama | 1024-dim vectors |
| 5. Vectors + metadata | Pandas + Joblib | `embeddings.joblib` |

### Phase 3 — Retrieval + Generation

| Step | Tool | Output |
|------|------|--------|
| 6. User query | BGE-M3 embed | Query vector |
| 7. Query vector | Cosine Similarity | Top-10 matching chunks |
| 8. Chunks + query | Prompt Engineering | Structured LLM prompt |
| 9. Prompt | Gemini 2.5 Flash / Llama3.2 | Video + Timestamp + Answer |

**Flow:**
```
Videos → MP3s → JSON Chunks → Merged Chunks → Embeddings → [joblib]
                                                                 ↑
User Query → Embed → Cosine Search → Top-10 Chunks → Prompt → LLM → Response
```

---

## ✨ Features

-  **Automated video ingestion** — batch converts any MP4 videos to MP3 using FFmpeg
-  **Multilingual transcription** — Whisper transcribes and translates Hindi audio to English with timestamps
-  **Smart chunking** — merges 256 raw segments into 52 optimized chunks (80% reduction), improving context quality 5x
-  **Semantic embeddings** — generates 1024-dimensional BGE-M3 vectors via local Ollama server
-  **Cosine similarity search** — retrieves top-10 semantically closest chunks per query
-  **Dual LLM support** — Gemini 2.5 Flash (cloud) or Llama3.2 (local via Ollama)
-  **Timestamp-aware responses** — every answer includes exact video number and start/end time in seconds
-  **Modular architecture** — 5 cleanly separated scripts for preprocessing and inference

---

## 📁 Folder Structure

```
RAG_Project/
│
├── videos/                  # Place your .mp4 course videos here
├── audios/                  # Auto-generated .mp3 files
├── jsons/                   # Raw transcript chunks (per video)
├── newjsons/                # Merged/optimized chunks (5-per-group)
│
├── video_to_mp3.py          # Step 1: Extract audio from videos
├── mp3_to_json.py           # Step 2: Transcribe audio → JSON chunks
├── merge_chunks.py          # Step 3: Merge small chunks → optimized chunks
├── preprocess_json.py       # Step 4: Generate embeddings → save embeddings.joblib
├── processing_incoming.py   # Step 5: Query → retrieve → LLM response
│
├── embeddings.joblib        # Persisted vector store (auto-generated)
├── prompt.txt               # Last generated prompt (debug reference)
├── response.txt             # Last LLM response (debug reference)
├── config.py                # API key configuration (excluded from git)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to PATH
- [Ollama](https://ollama.com) installed locally
- Google Gemini API key (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/SwanandS572/RAG_Assistant_Project.git
cd RAG_Assistant_Project
```

### 2. Install Python Dependencies

```bash
pip install openai-whisper pandas scikit-learn joblib requests google-genai
```

### 3. Pull the Embedding Model via Ollama

```bash
ollama pull bge-m3
```

### 4. Configure API Key

Create a `config.py` file in the root directory:

```python
# config.py
api_key = "your-gemini-api-key-here"
```

> Get your free Gemini API key at [https://aistudio.google.com](https://aistudio.google.com)

### 5. Add FFmpeg to PATH (Windows)

Download FFmpeg, extract to `C:\Program Files (x86)\ffmpeg\`, then add `C:\Program Files (x86)\ffmpeg\bin` to your System Environment Variables → Path.

---

## 🚀 Usage Guide

Run the pipeline **in order** when setting up for the first time:

### Step 1 — Add Your Videos

Place all `.mp4` course video files inside the `videos/` folder.

### Step 2 — Extract Audio

```bash
python video_to_mp3.py
```

Converts all videos → `.mp3` files saved in `audios/`

### Step 3 — Transcribe Audio to Text

```bash
python mp3_to_json.py
```

Whisper transcribes each audio file → timestamped JSON chunks saved in `jsons/`

### Step 4 — Optimize Chunks

```bash
python merge_chunks.py
```

Merges small transcript segments into larger context-rich chunks → saved in `newjsons/`

### Step 5 — Generate Embeddings

```bash
python preprocess_json.py
```

> ⚠️ Make sure Ollama is running: `ollama run bge-m3`

Generates 1024-dim BGE-M3 embeddings for all chunks → saved as `embeddings.joblib`

### Step 6 — Ask Questions

```bash
python processing_incoming.py
```

```
Ask a Question: Where is CSS flexbox taught?

→ CSS Flexbox is covered in Video 3 of the course at timestamp 245.0 to 310.5 seconds.
  Navigate to Video 3 and seek to approximately 4:05 minutes for this content.
```

> **After first-time setup**, only run `processing_incoming.py` for all future queries.

---

## 📊 Performance Metrics

| Metric | Value |
|---|---|
| Chunk reduction (merge strategy) | 256 → 52 chunks (80% reduction) |
| Embedding dimensions (BGE-M3) | 1,024-dimensional vectors |
| Embedding API calls saved | 256 → 52 (5x fewer calls) |
| Top-k retrieval per query | 10 chunks |
| Whisper model used | base (140 MB) |
| LLM backend | Gemini 2.5 Flash / Llama3.2 |

---

## 🔮 Future Improvements

- **Scale to 100+ videos** — process full YouTube playlists with channel-wide semantic search
- **Flask/FastAPI web interface** — browser-based Q&A instead of terminal
- **FAISS / ChromaDB integration** — replace Joblib with a proper vector database for faster retrieval at scale
- **Reranking layer** — add cross-encoder reranking after cosine retrieval for higher precision
- **Streaming LLM responses** — real-time token-by-token output for better UX
- **Direct YouTube URL support** — ingest videos directly via `yt-dlp` without manual download
- **Hyperparameter tuning** — optimize chunk merge size `n` per dataset for best retrieval accuracy

---

## 👤 Author

**Swanand Sinnarkar**
- GitHub: [@SwanandS572](/https://github.com/SwanandS572)
- LinkedIn: [linkedin/swanand](https://www.linkedin.com/in/swanand-sinnarkar-9167ab24b/)

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
