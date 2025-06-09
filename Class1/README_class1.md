
# Week 1: Introduction to Large Language Models & Prompt Engineering

Welcome to the **Machine Learning Engineer in the Generative AI Era** course!  
This week, you'll dive into the fundamentals of LLMs (Large Language Models), prompt engineering, and lay the foundation for building your own research agent.

---

## 📝 Course Overview

**Series Goal:**  
Build a functional research agent, learning practical data engineering and AI model skills along the way.

**This Week's Focus:**  
- What is Generative AI? What is agentic AI?
- LLM fundamentals: what they are, how they're trained, what they can do
- Prompt engineering: how to “talk” to LLMs to get the results you want
- Best practices: CO-STAR framework, structured output, and more

**By the end of this week, you should be able to:**  
- Describe LLMs and their capabilities in your own words
- Write and improve prompts for research/agentic tasks
- Format prompts for structured (JSON, XML) outputs
- Identify and analyze failure cases in LLM outputs

---

## 📚 Lecture Summary

- **What are LLMs?**  
  LLMs are models trained on massive internet-scale datasets, capable of reasoning, coding, summarizing, translating, and more.
- **How are LLMs trained?**  
  High-level overview: large datasets → next-token prediction → scaling law
- **How do we interact with LLMs?**  
  - Prompt engineering (zero-shot, few-shot, chain-of-thought)
  - System prompts vs. user prompts
  - Best practices: clear intent, context, explicit structure
- **CO-STAR Prompting Framework**  
  Provides structure and clarity to prompts, improving consistency in LLM outputs.
- **Research Agent Project**  
  All assignments connect to building your own research assistant.

---

## 🚀 Class 1 Hands-on Project

This project is **medium difficulty** (should take 2–4 hours). No GPU needed.

### Tasks

1. **Interact with ChatGPT**  
   Build your first simple prompt for a personalized research agent (e.g., “Summarize this paper in 5 bullet points for a high school student”).

2. **CO-STAR Rewrite**  
   Rephrase your prompt using the [CO-STAR](https://promptingguide.ai/techniques/costar) framework:
   - Context
   - Objective
   - Style
   - Tone
   - Audience
   - Response format

3. **Structured Output**  
   Learn to request outputs in JSON format. Example:
   ```json
   {
     "summary": "...",
     "key_points": ["...", "..."]
   }
   ```

4. **XML Output & Nesting**  
   Practice rewriting your prompt for XML output. Try using nested XML for more complex outputs or logic.

5. **Defects & Observations**  
   List the issues you find (e.g., LLM ignores format, makes factual errors, struggles with complex logic, lacks domain knowledge, etc.)

---

## 🖥️ Example: Prompt Engineering Code

This week's notebook walks you through:
- Sending a basic prompt to ChatGPT/OpenAI API
- Modifying the prompt using the CO-STAR framework
- Asking for JSON/XML output
- Documenting and analyzing observed LLM behaviors

**See [`class_1_lecture.ipynb`](./class_1_lecture.ipynb) and [`class_1_homework.ipynb`](./class_1_homework.ipynb) for step-by-step code and detailed explanations.**

---

## 🧑‍💻 How to Use This Repo

1. **Clone the repository:**  
   ```bash
   git clone <your-class-repo-url>
   cd <repo>
   ```

2. **Install dependencies:**  
   (see requirements in the notebook or provided `requirements.txt`)

3. **Run the notebooks:**  
   - `class_1_lecture.ipynb` for guided instruction and code explanations
   - `class_1_homework.ipynb` for hands-on practice and submission

---

## 💡 Best Practices & Tips

- **Prompt Engineering is an iterative process:**  
  Test, refine, and experiment.
- **Be explicit about format and constraints:**  
  LLMs follow clear instructions better.
- **Always check outputs:**  
  Look for mistakes, ignored instructions, or hallucinations.
- **Share questions and results:**  
  Join the Discord group (see syllabus) to discuss with peers and instructors.

---

## 📣 Homework Submission

- Complete all tasks in `class_1_homework.ipynb`
- Share your insights and interesting prompts in the Discord group
- Post at least one defect or surprising LLM behavior you found

---

## 📆 What’s Next?

**Next week:**  
You’ll get hands-on with LLM architectures (Transformers!), run your own LLM inference using Hugging Face and vLLM, and compare with ChatGPT.

**Prepare:**  
- Setup MCP (Model Context Protocol) server locally (instructions in lecture and Discord)
- Stay curious and experiment!

---

**Reference:**  
Course designed based on the latest open-source practices and real-world research agent projects.  
Inspired by Meta LLaMA, OpenAI, Hugging Face, and more.

---

*Questions?*  
Post on Discord or contact your instructor!

---
