
# Week 5: Supervised Fine-Tuning (SFT) I

**Machine Learning Engineer in the Generative AI Era — Series 1: Data Engineering**

## 📚 Overview

This week covers **Supervised Fine-Tuning (SFT)** — adapting a pre-trained LLM (like Llama 3/4, Mistral, or Zephyr) to perform specific tasks using labeled data. You’ll learn SFT concepts, dataset formats (especially ChatML), and get hands-on with LoRA, QLoRA, and full fine-tuning using DeepSpeed and TRL. All code and experiments will use cloud GPU infrastructure.

---

## 🏆 Learning Objectives

- Understand what SFT is and why it matters.
- Learn the ChatML data format for multi-turn LLM training.
- Compare **Full Fine-Tuning** vs. **LoRA/QLoRA/LowRA** approaches.
- Get practical experience with HuggingFace TRL, DeepSpeed, and PEFT.
- Fine-tune LLMs using open datasets and evaluate results.
- Grasp the differences in efficiency, resource requirements, and use cases for each SFT method.

---

## 🛠️ Environment & Prerequisites

1. **Inference AI Account**:  
   Sign up at [Inference AI Console](https://console.inference.ai/).  
   Set up your GPU server (recommended: A100 80GB/H100, SSH enabled).

2. **SSH Setup**:  
   Generate or find your SSH key (`cat ~/.ssh/id_rsa.pub`) and add it to your server via the console.

3. **Remote Access Example**:
   ```sh
   ssh -L 8000:localhost:8000 jovyan@YOUR_SERVER_IP -p PORT
   ```

4. **Environment**:
   - Python 3.10+
   - `pip install vllm peft transformers trl deepspeed langchain sentence-transformers`
   - [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli): `huggingface-cli login`
   - Download model (e.g., `HuggingFaceH4/zephyr-7b-alpha`) as shown in the lecture.

---

## 💡 What is Supervised Fine-Tuning (SFT)?

- **SFT** bridges general LLM pretraining and task-specific use by training on input-output pairs.
- Part of the model alignment pipeline:  
  *Pretraining → SFT → RLHF (RL with Human Feedback)*

**Why SFT?**
- Teaches the LLM your domain, style, or task with ground-truth examples.
- Great for customization, QA, chatbots, coding agents, etc.

---

## 🗃️ Data Format: ChatML

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is LoRA?"},
  {"role": "assistant", "content": "LoRA stands for Low-Rank Adaptation..."}
]
```
- Required for LLaMA 3/4, GPT-4, and similar models.
- Encourages multi-turn dialog.

---

## 🏗️ SFT Approaches

- **Full Fine-Tuning**: All model weights updated. Highest resource, max performance, risk of overfitting.
- **LoRA**: Trainable adapters update ~0.1% of parameters. Efficient, fast, great for customization.
- **QLoRA**: LoRA with 4-bit quantization — even less memory, fits on consumer GPUs.
- **LowRA**: <2-bit quantization (cutting-edge, ultra-compact).

**Tools:**  
- `peft` (LoRA, QLoRA, LowRA for Hugging Face)
- `trl` (Hugging Face’s SFT pipeline)
- `deepspeed` (efficient multi-GPU training, ZeRO optimization)
- `wandb`, `tensorboard` for monitoring

---

## 🚀 Hands-On: Example Fine-Tuning Workflow

### 1. Choose a Dataset (HuggingFace Datasets)
- e.g., `openassistant-guanaco`, `alpaca-cleaned`
- Format data to ChatML

### 2. Run LoRA Fine-Tuning (30min-1hr)
- Use `peft`, `trl`, or [DeepSpeed](https://www.deepspeed.ai/).
- Monitor training loss and metrics.

### 3. Run Full Fine-Tuning (2+ hrs)
- Update all model weights.
- Monitor for overfitting.

### 4. Evaluate & Compare
- Use new prompts and compare answers.
- Metrics: loss, accuracy, helpfulness, style match, etc.

### 5. (Bonus) Try QLoRA or LowRA

---

## 🖥️ Code Example: SFT with LoRA

Check the provided notebook (`class_5_llama3.py`) for full code, including:

- Data preprocessing and formatting to ChatML
- Setting up model and adapters
- Running SFT with LoRA/QLoRA
- Inference and comparison with original model

---

## 📝 Project 5: Tasks

1. **Select**: Find a related Hugging Face dataset, convert to ChatML.
2. **LoRA Fine-Tune**: Fine-tune your LLM using LoRA, report run time and metrics.
3. **Full Fine-Tune**: Run full fine-tuning, compare results to LoRA and baseline.
4. **Evaluate**: Prepare a short report—show outputs, metrics, and discuss overfitting or style transfer.
5. **(Bonus)**: Try QLoRA or LowRA.

---

## 🏁 Evaluation Criteria

- Proper data formatting and preprocessing
- GPU setup and usage (logs required)
- Hyperparameter tuning (learning rate, epochs, batch size)
- Clear report: LoRA vs. Full Fine-Tune, evaluation on new prompts
- Bonus: QLoRA or LowRA exploration

---

## 🔗 Resources

- [Hugging Face LoRA Guide](https://huggingface.co/docs/peft/index)
- [DeepSpeed Docs](https://www.deepspeed.ai/)
- [TRL Package](https://github.com/huggingface/trl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Class Notebook & Code](./class_5_llama3.py)
- [Inference AI Docs](https://console.inference.ai/)

---

## 📝 Submission

- Push your code, logs, and a brief report to the class repo.
- Submit a short comparison table for LoRA vs. Full Fine-Tune.
- Join the discussion on Discord for Q&A!

---

## 👋 Next Week

SFT II — Data diversity, quality, synthetic data, LLMs as judges, evaluation with Elo/rejection sampling.

---

# Happy Fine-Tuning! 🚀
