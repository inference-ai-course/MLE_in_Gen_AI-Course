
# Week 3: Pretraining Data Collection & Extraction

Welcome to **Week 3** of the Machine Learning Engineer in the Generative AI Era course!

This week, we focus on **how the world’s best LLMs are actually trained: building and processing massive, high-quality pretraining datasets**. You’ll learn the core skills for collecting, cleaning, and preparing raw web and multimodal data—essential for research agents and anyone building foundation models.

---

## 📝 Overview

**Lecture Themes:**
- Why data quality and diversity matter for LLM pretraining
- Real-world pretraining pipelines: Common Crawl, OCR (image-to-text), ASR (audio-to-text)
- Data cleaning and filtering techniques used in LLaMA 4, GPT-4, and other SOTA models
- Hands-on project: from web scraping to deduplication

**Key Takeaways:**
- “Garbage in, garbage out”—the value of your pretraining data defines the value of your model
- Multi-source data pipelines (web, PDF, audio) are the norm for modern LLMs
- Filtering, deduplication, and language/PII checks are essential for data quality
- You’ll practice with open-source tools like Trafilatura, Tesseract, Surya, Whisper, and more

---

## 🖥️ Hands-On Project 3: Build Your Own Pretraining Dataset (Challenging)

### Objective
You’ll design and implement a *mini* LLM data pipeline, just like big labs do—scraping real scientific papers, running OCR on PDFs, and cleaning/filtering your own dataset.

### Tasks

1. **Scrape Papers from arXiv**
   - Select a topic (e.g., NLP, AI safety, robotics)
   - Use Python (requests, BeautifulSoup, scrapy, etc.) to download a sample set of arXiv papers

2. **Extract Text via OCR**
   - Convert downloaded PDFs to raw text using OCR tools:
     - [Tesseract](https://github.com/tesseract-ocr/tesseract) (easy, open-source)
     - [Surya](https://github.com/VikParuchuri/surya) (layout-aware OCR for scientific docs)
     - OpenAI GPT-4o Vision API (if available; handles tables and images)
   - Tips: Pre-process images (grayscale, denoise) for better OCR accuracy

3. **Apply Data Cleaning & Filtering**
   - Remove duplicate content (e.g., MinHash deduplication)
   - Strip PII, non-English or low-quality text
   - Optional: Filter HTML, extract only main article content

4. **Report & Analyze**
   - Document your pipeline, the data you collected, and *defects/issues* you observed (e.g., OCR failures, duplicates, wrong language)

---

## 🗂️ Example Tools & Libraries

- **Web Scraping:** requests, BeautifulSoup, scrapy, arxiv.py, trafilatura
- **OCR:** pytesseract, Surya, OpenAI GPT-4o (Vision)
- **Data Cleaning:** pandas, regex, MinHash, langdetect, spaCy
- **Deduplication:** datasketch (MinHash)
- **ASR (optional):** Whisper, faster-whisper, Google Speech API

---

## 🔬 Real-World Pipeline (Case Study: LLaMA 4)

- **Data sources:** Web pages, PDFs, images, proprietary datasets
- **Filtering:** MinHash for deduplication, language classification, PII removal
- **Volume:** Up to 40 trillion tokens for frontier models
- **Multimodal:** Modern LLMs train on both text and images (Scout, Maverick models)
- **Ablation Studies:** Analyze which data types help most; report what improves performance

---

## 🧑‍💻 How to Run This Week’s Code

1. **Clone the repository:**  
   ```bash
   git clone <your-class-repo-url>
   cd <repo>
   ```

2. **Open the notebook:**  
   - [`class_3_lecture.ipynb`](./class_3_lecture.ipynb) — Guided code and explanations
   - Use any GPU-enabled Colab/Kaggle or your local Python environment

3. **Install dependencies:**
   - Most are installable via pip (`pip install tesseract pandas trafilatura datasketch ...`)
   - For Tesseract, follow [install instructions here](https://github.com/tesseract-ocr/tesseract)

---

## 💡 Tips & Best Practices

- **Start small:** Process a handful of papers end-to-end before scaling up
- **Check quality:** Inspect random samples of your extracted text
- **Keep it reproducible:** Save all code/scripts and document each pipeline step
- **Report what didn’t work:** List all data defects and pipeline failures for discussion

---

## 📣 Homework Submission

- Complete all code and analysis in `class_3_lecture.ipynb`
- Document defects/issues (duplicate text, OCR errors, noise) and suggest ways to fix them
- Share your best/funniest/most interesting extraction result on Discord

---

## 🏁 What’s Next?

**Next week:** You’ll move from raw data to retrieval-augmented generation (RAG)—combining LLMs with external knowledge, vector DBs, and LangChain.

---

**References:**  
- [Meta AI Blog: The LLaMA 4 Herd](https://ai.meta.com/blog/meta-llama-4/)  
- [Trafilatura](https://trafilatura.readthedocs.io/en/latest/)  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)  
- [Surya OCR](https://github.com/VikParuchuri/surya)

---

*Questions or issues?*  
Ask on Discord or post in the project repo!

---
