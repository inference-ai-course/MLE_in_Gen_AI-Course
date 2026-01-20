# NLP Data Pipeline: Scraping, OCR, Transcription & Cleaning

End-to-end pipeline for collecting and cleaning text data from arXiv papers, PDFs, and YouTube videos.

## Pipeline Overview

```
Task 1: arXiv Scraper → arxiv_clean.json, arxiv_pdf/*.pdf
         ↓
Task 2: PDF OCR → pdf_ocr/*.txt
         ↓
Task 3: YouTube Transcription → talks_transcripts.jsonl
         ↓
Task 4: Data Cleaner → clean_corpus.txt, stats.md
```

---

## Quick Start

### Installation
```bash
# Python packages
pip install trafilatura pytesseract pdf2image yt-dlp openai-whisper langdetect datasketch beautifulsoup4

# System dependencies (macOS)
brew install poppler tesseract ffmpeg

# Linux
sudo apt-get install poppler-utils tesseract-ocr ffmpeg
```

### Execution
```bash
# 1. Scrape arXiv papers (edit line 308 for PDF downloads)
python task1_scraper.py

# 2. OCR PDFs
python task2_batch_ocr.py

# 3. Transcribe YouTube videos (edit videos list first)
python task3_whisper_transcription.py

# 4. Clean and merge all data
python task4_data_cleaning.py
```

---

## Tasks

### Task 1: arXiv Paper Scraper
**Description:** Fetches academic papers from arXiv API, scrapes abstracts from web pages, and optionally downloads PDFs for downstream OCR processing.

**Script:** `task1_scraper.py`

**Input:**
- arXiv category (default: `cs.CL`)
- Max papers (default: 200)

**Output:**
- `arxiv_clean.json` - Paper URLs, titles, abstracts, authors, dates
- `arxiv_pdf/` - Downloaded PDF files (if `use_ocr=True`)

---

### Task 2: Batch PDF OCR
**Description:** Converts PDF documents to text using Tesseract OCR. Processes each page as high-resolution images and preserves layout with page separators for readability.

**Script:** `task2_batch_ocr.py`

**Input:**
- `arxiv_pdf/*.pdf` - PDFs from Task 1

**Output:**
- `pdf_ocr/*.txt` - OCR-extracted text with page headers
- `pdf_ocr/ocr_summary.json` - File sizes, line counts, processing stats

---

### Task 3: Whisper Transcription Bot
**Description:** Downloads audio from YouTube videos using yt-dlp, transcribes speech to text with OpenAI Whisper, and generates timestamped segments for each video.

**Script:** `task3_whisper_transcription.py`

**Input:**
- YouTube video URLs (configured in script)

**Output:**
- `talks_transcripts.jsonl` - Full text, segments with start/end timestamps, metadata
- `youtube_audio/*.mp3` - Downloaded audio files

---

### Task 4: End-to-End Data Cleaner
**Description:** Merges all data sources into a unified corpus, then applies comprehensive cleaning: language filtering, HTML removal, near-duplicate detection with MinHash LSH, PII masking, and repetitive pattern removal.

**Script:** `task4_data_cleaning.py`

**Input:**
- `arxiv_clean.json` - From Task 1
- `pdf_ocr/*.txt` - From Task 2
- `talks_transcripts.jsonl` - From Task 3

**Output:**
- `clean_corpus.txt` - Deduplicated, cleaned text corpus
- `stats.md` - Document counts, token statistics, removal percentages

**Pipeline:**
1. Merge all sources
2. Language detection (keep English only)
3. HTML stripping & entity decoding
4. MinHash deduplication (similarity ≥ 0.7)
5. PII masking (emails → `[EMAIL]`, phones → `[PHONE]`, etc.)
6. Repetitive n-gram removal (3-grams appearing >5 times)

---

## Output Files

| File | Description |
|------|-------------|
| `arxiv_clean.json` | arXiv abstracts and metadata |
| `arxiv_pdf/*.pdf` | Downloaded papers |
| `pdf_ocr/*.txt` | OCR-extracted text |
| `talks_transcripts.jsonl` | Video transcripts with timestamps |
| `clean_corpus.txt` | Final cleaned corpus |
| `stats.md` | Token counts, removal rates |

---

## Configuration

**Task 1:**
- Line 293: `max_papers` - Number of papers to fetch
- Line 308: `use_ocr=True` - Enable PDF downloads

**Task 3:**
- Lines 235-276: Edit `videos` list with YouTube URLs

**Task 4:**
- `similarity_threshold` - MinHash deduplication threshold (default: 0.7)
- `n` - N-gram size (default: 3)
- `threshold` - Max n-gram repetitions (default: 5)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| pdf2image not found | `pip install pdf2image && brew install poppler` |
| Tesseract not found | `brew install tesseract` (macOS) |
| ffmpeg not found | `brew install ffmpeg` (macOS) |
| Large file issues | Reduce `max_papers` in Task 1 |

---

## Project Structure

```
class2/
├── task1_scraper.py
├── task2_batch_ocr.py
├── task3_whisper_transcription.py
├── task4_data_cleaning.py
├── arxiv_clean.json
├── arxiv_pdf/*.pdf
├── pdf_ocr/*.txt
├── talks_transcripts.jsonl
├── clean_corpus.txt
└── stats.md
```
