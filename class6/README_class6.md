
# Week 6: Supervised Fine-Tuning (SFT) II – README

## 📚 Course: Machine Learning Engineer in the Generative AI Era  
**Series 1: Data Engineering**  
**Week 6 – Supervised Fine-Tuning (SFT) Part II**

---

## 🚩 Objectives

- Understand the importance of diversity in SFT data (both human and model-generated)
- Learn techniques for synthetic data generation
- Apply quality assurance and filtering methods for SFT datasets
- Use LLMs as judges and implement rejection sampling
- Explore the LLMsys Arena & Elo rating system for model evaluation
- Complete hands-on projects to generate and evaluate SFT data

---

## 📝 Lecture Summary

### 1. **Ensuring Diversity in SFT Data**

- **Why it matters:**  
  - Improves model generalization and reduces bias
  - Covers a range of linguistic patterns, domains, and demographics
  - Prevents overfitting to narrow data subsets
- **Strategies:**  
  - Include diverse annotators, domains, and formats (factual Q&A, dialogues, code, summarization)
  - Use different base models (e.g., LLaMA 3, GPT-4-turbo, Mistral-Instruct) for candidate generation
  - Employ varied sampling techniques (top-k, top-p, multiple seeds)

### 2. **Synthetic Data Generation**

- **Motivation:**  
  - Fills gaps where real data is scarce
  - Reduces annotation cost, enhances privacy, and accelerates iteration
- **Common Approaches:**  
  - **Self-Instruct / Model-in-the-Loop:** Prompt LLMs to generate Q&A/dialog pairs; iteratively refine via rejection sampling
  - **Distillation:** Teacher model (e.g., GPT-4) labels raw input; student (smaller model) trains on this data
  - **GANs / VAE:** Generate realistic text or sample latent representations
  - **Evolutionary Methods:** Generate, evaluate (with LLM-as-judge), and iteratively refine synthetic sets
- **Key Tools:**  
  - [Hugging Face Synthetic Data Generator](https://huggingface.co/spaces/trl-lib/synthetic-data-generator)
  - Scale AI, Gretel Navigator

### 3. **Best Practices for High-Quality Synthetic Data**

- **Realism & Alignment:** Use in-domain prompts and system instructions for better data
- **Edge Cases:** Generate adversarial/uncommon examples and simulate noisy inputs
- **Privacy:** Apply differential privacy, automated PII detection
- **Validation & Filtering:**  
  - Statistical checks on distribution overlap  
  - Expert review of sample data  
  - Automated metrics (e.g., perplexity thresholds)
- **Iterative Refinement:** Generate → Evaluate (LLM-as-judge) → Filter → Retrain

### 4. **Quality Assurance: LLM-as-a-Judge & Rejection Sampling**

- **Concept:**  
  - LLMs (using chain-of-thought reasoning) compare and score generated examples for plausibility, diversity, and factuality
- **Rejection Sampling Pipeline:**  
  1. Generate N candidates per prompt  
  2. Use LLM judge to score and select top K examples  
  3. Iterate based on feedback and metrics

### 5. **LLMsys Arena & Elo Rating System**

- **What:**  
  - A head-to-head benchmarking platform where models “battle” on prompts, and human/LLM judges vote on responses
- **How:**  
  - Elo rating: Track progress, measure improvements across SFT cycles
  - Pairwise model comparison using a standard set of prompts
- **Benefits:**  
  - Captures relative model improvement  
  - Provides a single score for tracking SFT progress

---

## 🚀 Project 6: SFT Data Generation & Evaluation

**Goal:** Synthesize SFT data in ChatML format and optimize for diversity and quality.

### Tasks

1. **Synthesize SFT Data**
    - Use your model’s weaknesses and data scraped in Project 3 to prompt an LLM for new Q&A pairs
    - Ensure diversity: Vary prompts, annotators, and generation settings
    - Format all data in ChatML style (multi-turn conversations)
2. **Ablation Study**
    - Analyze the impact of different data quantities and diversity levels on SFT performance
    - Mix your synthetic data with HF datasets, try different ratios (e.g., 30/70, 50/50)
    - Fine-tune your SFT model, track improvements over the original model
    - Report on metrics (accuracy, loss, Elo, qualitative comparison)
3. **(Optional/Bonus):**  
    - Build a mini LLMsys Arena or use existing open platforms (e.g., [OpenLM.ai Chatbot Arena](https://openlm.ai/arena)) to benchmark your fine-tuned model

---

## 🛠️ Code Starter – SFT Data Synthesis & Evaluation

```python
# Example: Self-Instruct for Synthetic Data
from transformers import pipeline

generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")

prompt = (
    "Generate 5 question-answer pairs about Python decorators. "
    "Format as JSON: [{"question": "...", "answer": "..."}, ...]"
)
output = generator(prompt, max_new_tokens=256)[0]["generated_text"]
print(output)

# For rejection sampling, you can iterate this process and use another LLM to score each Q&A pair.
```

```python
# Example: LLM-as-a-Judge (pseudo-code)
def judge_pairs(candidate_pairs):
    judge_prompt = (
        "For each question-answer pair below, rate factuality and diversity (1–5):\n"
        f"{candidate_pairs}"
    )
    judge = pipeline("text-generation", model="gpt-4")  # or GPT-4o API
    scores = judge(judge_prompt, max_new_tokens=128)
    return scores
```

---

## 📑 References & Further Reading

- [LabelYourData: Synthetic Data for LLM Fine-Tuning](https://labelyourdata.com/research/synthetic-data-llm-fine-tuning-2025)
- [Scale AI: Synthetic Data Strategies for LLMs](https://scale.com/blog/synthetic-data-llms)
- [Hugging Face Synthetic Data Generator](https://huggingface.co/spaces/trl-lib/synthetic-data-generator)
- [Gretel Navigator for Synthetic Data](https://gretel.ai/)
- [LLMsys Arena / OpenLM.ai](https://openlm.ai/arena)
- [ArXiv: “Unlocking Comprehensive Evaluations for LLM-as-a-Judge”](https://arxiv.org/abs/2502.12501)

---

## ✅ Deliverables

- A report with:
    - Synthetic SFT data in ChatML format
    - Ablation study results and analysis of data mixture ratios
    - Elo rating or qualitative evaluation of your fine-tuned model
- (Optional): Demo or screenshots from LLMsys Arena or a similar benchmarking tool

---

*For more details, check the class repo, lecture slides, and the starter code in the week_6_lecture notebook.*

---

**Good luck and happy fine-tuning!**
