# Class 2: LLM Overview – Transformers, Pretraining & Fine-Tuning

Welcome to **Week 2** of the Machine Learning Engineer in the Generative AI Era series!
This week, you’ll get hands-on with LLMs (large language models), understand their architecture, learn how they’re trained, and perform model inference using open-source tools. By the end, you’ll be able to serve and experiment with your own models on local GPUs.

---

## 📅 Agenda

- What is a Transformer?
- Attention Mechanism: Self-Attention & Multi-Head Attention
- LLM Training Phases:
  - Pretraining
  - Supervised Fine-Tuning (SFT)
  - Alignment (DPO, PPO)
- Data Requirements, Costs, and Challenges
- Test-Time Scaling (O1, O3)
- Hands-On Project: Hugging Face & vLLM Inference

---

## 1. Transformer Architecture & Attention

- **Transformer**: The foundation of modern LLMs. Unlike RNNs, they process all tokens in parallel using *attention* mechanisms.
- **Multi-Head Self-Attention**: Allows the model to focus on different parts of the input at once, understanding relationships in context.
- **Positional Encoding**: Adds info about the order of tokens, since transformers lack inherent sequence-awareness.
- **Key References**:
  - [Vaswani et al., “Attention is All You Need”](https://arxiv.org/abs/1706.03762)
  - [Dive into Deep Learning: Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
- **Hands-On**: See class notebook for code demos and visualizations.

---

## 2. Next Token Prediction & Hallucination

- **Next Token Prediction**: LLMs predict the next word given a sequence._Example: “The Eiffel Tower is located in ___” → “Paris”_
- **Hallucination**: When the model generates plausible-sounding but incorrect output.
  - _Reduce by_: better prompts, RAG, instruction tuning, post-processing validation.

---

## 3. LLM Pretraining

- **Goal**: Train on massive, diverse datasets (Common Crawl, Wikipedia, code, books).
- **Tokenization**: Text is broken into tokens using tools like BPE or SentencePiece.
- **Challenges**: Data quality, PII removal, cost (compute, GPUs), balancing diversity.
- **Modern Example**:
  - **LLaMA 4** (2025):
    - ~22T–40T tokens, 200+ languages, multimodal inputs, FP8 training, Mixture-of-Experts (MoE), deduplication & filtering
    - [Llama 4 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/)
- **Case Study**: See notebook for data pipeline code and best practices.

---

## 4. Supervised Fine-Tuning (SFT)

- **Purpose**: Adapt a pretrained model to specific tasks using labeled data.
  - Methods: *Full finetuning* (all weights), *LoRA* (efficient, trains small adapters)
  - Example applications: question answering, summarization, text classification
- **LoRA, QLoRA, LowRA**:
  - Techniques for parameter-efficient fine-tuning using low-rank matrix adapters and quantized weights.
  - **Why it matters**: Fine-tune big models on consumer GPUs!
- **Recommended tools**: [Hugging Face PEFT](https://github.com/huggingface/peft), [Deepspeed](https://github.com/microsoft/DeepSpeed), [TRL](https://github.com/huggingface/trl)

---

## 5. Alignment: DPO & PPO

- **Direct Preference Optimization (DPO)**: Aligns model outputs with human preferences using ranking data.
- **Proximal Policy Optimization (PPO)**: Reinforcement learning technique for safer, more reliable models.
- **When is alignment needed?** After SFT, to ensure models follow user intent and are robust.

---

## 6. Data Needs, Costs, and Test-Time Scaling

- **Data Requirements**: At least trillions of tokens; diverse and high-quality data is key.
- **Training Costs**: Pretraining can cost millions (e.g., 3000+ A100s for months), but SFT/LoRA is accessible to small teams.
- **Test-Time Scaling**:
  - **O1**: Mixed-precision, quantization, memory optimization.
  - **O3**: Advanced attention mechanisms, reflection, adaptive computation.
  - **Goal**: Reduce inference cost, improve speed, enable large context windows.

---

## 🛠️ Hands-On Project

### Project 2: LLM Inference & Serving

**Goal:**

- Get familiar with Hugging Face and vLLM
- Serve your own Llama 3/4 or Mistral model locally or on cloud GPU
- Compare performance against OpenAI/ChatGPT

**Tasks:**

1. **Download and Run Open-Source Models**
   - Use [Hugging Face Transformers](https://huggingface.co/docs/transformers/) to download Llama3/4 or Mistral
   - Example:
     ```python
     from transformers import AutoTokenizer, AutoModelForCausalLM
     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
     model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
     prompt = "The Eiffel Tower is located in"
     inputs = tokenizer(prompt, return_tensors="pt")
     outputs = model.generate(**inputs, max_new_tokens=10)
     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
     ```
2. **Try vLLM Serving**
   - [vLLM](https://github.com/vllm-project/vllm) enables lightning-fast inference
   - Start server:`python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct`
   - Call the model via OpenAI-compatible API
3. **Compare Results**
   - Use your prompt from Project 1
   - Evaluate how local models stack up vs ChatGPT in accuracy, speed, and cost

---

## 📚 Resources

- [Transformers: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA 4 Model Card &amp; Data Strategy](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [LoRA and QLoRA Paper](https://arxiv.org/abs/2106.09685)
- [Meta AI Blog: Llama 4 Data Pipeline](https://ai.facebook.com/blog/llama4-herd/)
- [Course Discord Community](https://discord.gg/your-invite-link)

---

## 💡 Tips

- If you don’t have a GPU, use free cloud platforms like [Google Colab](https://colab.research.google.com/) or request access to university GPU clusters.
- Troubleshoot model loading or CUDA issues in Discord.
- Bonus: Try running SFT or LoRA finetuning using your own dataset (see next week for full workflow).

---

## ✅ Homework Checklist

- [ ] Run and test open-source LLMs (Llama3/4, Mistral) using Hugging Face
- [ ] Serve a model with vLLM, try OpenAI API interface
- [ ] Compare your prompt results with ChatGPT, note differences
- [ ] Document issues, solutions, and insights in your project repo

---

Happy learning! 🚀
_Questions? Reach out on Discord!_
