
# Machine Learning Engineer in the Generative AI Era

Welcome to the comprehensive 10-week course on building production-ready AI agents and mastering the entire ML engineering lifecycle in the era of Large Language Models (LLMs).

## 🎯 Course Overview

This course integrates every key topic in modern AI engineering: Data Engineering, foundational LLM concepts, Retrieval-Augmented Generation (RAG), LLM fine-tuning, and model alignment. Everything builds toward creating your own **end-to-end research agent** that can search papers, extract content via OCR, generate summaries, and even create podcast-style content.

## 📚 Course Structure

* **Duration** : 10 weeks
* **Format** : 2-hour weekly lectures + hands-on projects
* **Focus** : Data Engineering for LLMs
* **Final Deliverable** : Deployable AI research agent

## 🗓️ Weekly Schedule

### **Weeks 1-3: Data Engineering Foundations**

| Week             | Topic                              | Key Concepts                                                      | Project                                                           |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Week 1** | Intro to LLMs & Prompt Engineering | Generative AI, prompting techniques (CO-STAR), JSON/XML output    | Design prompts for research agent using CO-STAR framework         |
| **Week 2** | LLM Architecture & Training        | Transformers, hallucination, SFT/DPO/PPO, scaling laws            | Run local LLM inference (LLaMA 3/4), evaluate with custom prompts |
| **Week 3** | Data Collection & Extraction       | Web scraping, OCR (Tesseract/Surya), ASR (Whisper), data cleaning | Scrape arXiv, OCR PDFs, filter & clean data for pretraining       |

### **Weeks 4-7: AI Model Training & Enhancement**

| Week             | Topic                                | Key Concepts                                            | Project                                                   |
| ---------------- | ------------------------------------ | ------------------------------------------------------- | --------------------------------------------------------- |
| **Week 4** | Retrieval-Augmented Generation (RAG) | Embeddings, chunking, vector DBs, LangChain             | Build RAG pipeline to augment LLM with external knowledge |
| **Week 5** | Supervised Fine-Tuning (SFT) I       | Full vs. LoRA fine-tuning, ChatML format, TRL/Deepspeed | Apply LoRA and full fine-tuning, explore overfitting      |
| **Week 6** | Supervised Fine-Tuning (SFT) II      | Synthetic data, quality checks, LLM-as-judge            | Generate synthetic SFT data, perform ablation studies     |
| **Week 7** | Model Alignment                      | RLHF, DPO/PPO, reward modeling, data labeling           | Build Gradio labeling tool, run DPO alignment experiment  |

### **Weeks 8-10: Advanced Topics & Project Completion**

| Week              | Topic                 | Key Concepts                                                 | Project                                                       |
| ----------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------- |
| **Week 8**  | Safety & Ethics       | Hallucination prevention, jailbreak methods, bias mitigation | Test model safety, explore jailbreaking, safety datasets      |
| **Week 9**  | Voice & Multimodal AI | GPT-4o real-time, ASR/TTS pipelines, chained agents          | Build voice agent (GPT-4o style), explore NotebookLM pipeline |
| **Week 10** | Final Capstone        | Agents, MCP protocol, function calling, task chaining        | Complete end-to-end research agent with voice output          |

## 🚀 Core Projects

### 1. **End-to-End AI Research Agent** (Main Capstone)

Build an intelligent agent that enables natural language queries about research papers:

* **Data Engineering** : Collect and preprocess academic papers
* **RAG Integration** : Ground responses in real research documents
* **Fine-Tuning** : Personalize with supervised fine-tuning
* **Alignment** : Ensure safe, accurate, and relevant answers
* **Deployment** : Create a working, demonstrable agent

### 2. **Voice Research Assistant** (Homework Project)

Develop a voice-driven research assistant:

* Integrate ASR, LLMs, and TTS
* Build chained audio-AI pipelines
* Create demo video for portfolio

### 3. **Custom AI Agent** (Optional)

Design an agent aligned with your career interests:

* Choose any domain (music, biotech, legal, etc.)
* Present at public showcase event
* Compete for top project recognition

## 🛠️ Technical Stack

### **Core Technologies**

* **LLMs** : LLaMA 3/4, ChatGPT, Claude
* **Frameworks** : LangChain, TRL, Deepspeed
* **Data** : Web scraping, OCR (Tesseract/Surya), ASR (Whisper)
* **Vector DBs** : For RAG implementation
* **Fine-tuning** : LoRA, full fine-tuning methods

### **Tools & Protocols**

* **MCP (Model Context Protocol)** : For agent integration
* **Gradio** : For building labeling interfaces
* **Git** : Version control and collaboration
* **Discord** : Community Q&A and code sharing

## 🎓 Learning Outcomes

By the end of this course, you will:

✅  **Master Modern AI Engineering** : From data collection to model deployment

✅  **Build Production-Ready Agents** : Complete end-to-end AI systems

✅  **Understand LLM Lifecycle** : Pretraining, fine-tuning, and alignment

✅  **Implement RAG Systems** : Advanced retrieval-augmented generation

✅  **Deploy Real Applications** : Career-ready portfolio projects

✅  **Navigate AI Safety** : Ethical considerations and safety alignment

## 🚦 Getting Started

### Prerequisites

* Python 3.8+
* Basic understanding of machine learning
* Familiarity with Git and command line
* 8GB+ RAM (16GB recommended for local LLM inference)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/inference-ai-course/MLE_in_Gen_AI-Course.git
cd MLE_in_Gen_AI-Course

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# or just follow the instruction from the jupyter Notebooks

```

### Environment Configuration

1. **API Keys** : Set up OpenAI, Anthropic, or other LLM API keys
2. **MCP Setup** : Configure Model Context Protocol for agent integration
3. **Discord** : Join the course Discord for Q&A and collaboration

## 📋 Project Milestones

| Week | Milestone          | Deliverable                                    |
| ---- | ------------------ | ---------------------------------------------- |
| 1    | Project Kickoff    | Define research agent goals, initial prompts   |
| 4    | Project Insight I  | Share progress, receive peer feedback          |
| 7    | Project Insight II | Lock in project direction & components         |
| 10   | Final Presentation | Working agent demo, learnings, technical depth |

## 🤝 Community & Support

* **Discord Server** : Real-time Q&A and code sharing
* **Office Hours** : Weekly TA sessions for project guidance
* **Peer Review** : Collaborative feedback sessions
* **Showcase Event** : Public presentation of final projects

## 📈 Assessment & Portfolio Impact

### Course Projects as Career Assets

* **Deployable Research Agent** : Live demo for interviews
* **GitHub Portfolio** : Complete, documented projects
* **Technical Blog Posts** : Document your learning journey
* **Demo Videos** : Showcase multimodal agent capabilities

### Recognition Opportunities

* **Top Project Awards** : Judged showcase competition
* **Industry Connections** : Guest speakers and networking
* **Open Source Contributions** : Contribute to course materials

## 🔧 Technical Requirements

### Hardware

* **Minimum** : 8GB RAM, modern CPU
* **Recommended** : 16GB+ RAM, GPU for local training
* **Cloud Alternative** : Google Colab Pro, AWS, or similar

### Software

* Python 3.8+, Node.js (for MCP)
* Git, Docker (optional)
* Code editor (VS Code recommended)

---

## 🎉 Ready to Start?

1. **Week 1** : Head to `week01/` and follow the setup instructions
2. **Join Discord** : Connect with classmates and instructors
3. **Define Your Goal** : Write your one-sentence agent mission
4. **Start Building** : Begin with prompt engineering fundamentals

**Your AI engineering journey starts now!** 🚀

---

*This course is designed to be highly practical, career-focused, and immediately applicable to real-world AI engineering roles. Every project builds toward creating tangible, demonstrable skills that will set you apart in the rapidly evolving AI landscape.*
