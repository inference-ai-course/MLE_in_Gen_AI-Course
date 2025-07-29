# Week 2 Support Guide – Understanding LLMs for Beginners

## **Lecture 2 was meant to give you a high-level overview of the course, a sneak peek into all the powerful concepts we’ll be learning. Don’t worry if it feels like a lot right now. We’ll break everything down step by step in the coming weeks, and you’ll build confidence as we go. One concept at a time. you’ve got this!**

---

## 👩‍🏫 Key Concepts Explained Simply

### 1. What is a Transformer?

A Transformer is like a smart librarian who reads an entire book at once and can tell you what the next word is, based on the story so far.

📖 Learn more:

- Video: [Jay Alammar’s Illustrated Transformer](https://www.youtube.com/watch?v=4Bdc55j80l8)
- 3Blue1Brown Video: [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M&t=1162s)
- Article: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

### 2. What is Attention?

"Attention" helps the model figure out which words matter the most — like highlighting key sentences in an article.

📖 Learn more:

- Beginner video: [Attention Mechanism ](https://www.youtube.com/watch?v=fjJOgb-E41w)(Google)
- Step by Step: [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=6s)(3Blue1Brown)

---

### 3. What is Next Token Prediction?

It’s like texting with autocomplete — the model guesses the next word you might want to write.

📖 Learn more:

- Akash Kesrwani's Mediumblog: [Understanding Next Token Prediction](./Addition-Reading/Understanding-Next-Token-Prediction.pdf)

---

### 4. What is Hallucination?

When the model makes up information that sounds real but isn’t. Like confidently saying “Bananas were invented in France.”

📖 Learn more:

- IBM: [Why Large Language Models Hallucinate](https://www.youtube.com/watch?v=cfqtFvWOfg0)

---

### 5. What is LLM Pretraining?

Before the model can help you, it reads A LOT — like millions of books and websites. This is called “pretraining.”

📖 Learn more:

- New Machina Video: [What is LLM Pre-Training?](https://www.youtube.com/watch?v=P7emqEtkiSk)

---

### 6. What is Supervised Fine-Tuning (SFT)?

After pretraining, we teach the model to behave better by showing it examples of good responses. Think of it like tutoring after school.

📖 Learn more:

- Guide: [Supervised Fine-Tuning (SFT)](https://klu.ai/glossary/supervised-fine-tuning)
- Video: [Supervised Fine Tuning (SFT)](https://www.youtube.com/watch?v=ofhHKs1kRBE)

---

### 7. What is Alignment (DPO, PPO)?

These are methods to make the model act more ethically and follow human rules. Like teaching it to behave politely and not say bad things.

📖 Learn more:

- Explainer: [DPO for LLMs](https://unfoldai.com/dpo-llms/)
- Overview: [RLHF and PPO vs DPO](./Additional-Reading/RLHF(PPO)_vs_DPO.pdf)

---

### 8. What is Test-Time Scaling (O1, O3)?

Techniques to make the model faster and cheaper when we use it. Like turbo-charging your car for better gas mileage.

📖 Learn more:

- Simple overview: [Test-Time Optimization](https://huggingface.co/blog/Kseniase/testtimecompute)

---

## 🧰 Tools We Use (in Lecture Code)

- [Hugging Face](https://huggingface.co/models) for downloading models like LLaMA or Zephyr.
- [vLLM](https://docs.vllm.ai/en/latest/) for fast local model serving.
- [LangChain](https://docs.langchain.com/docs/) for building smart QA systems.

---

## 🎓 Hands-On: What You Should Try

- Try loading a model like `llama3` or `zephyr` and ask it: “What is a Transformer?”
- Compare what your local model vs ChatGPT says — which one is smarter?

---

## 👩‍🔬 Bonus Reading: LLaMA 4 Case Study

Learn how Meta trained one of the most powerful models using trillions of tokens:

- [LLaMA 4 Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/)

---

Many students felt overwhelmed after Lecture 2 — and that’s totally okay. This lecture was designed to give you a big-picture view of the course. It introduces all the key ideas we’ll dive deeper into throughout the coming weeks. Don’t worry, we’ll break everything down step by step and build your understanding gradually. You’ve got this!
