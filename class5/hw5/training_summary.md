# Training Summary Report
## LLM Fine-Tuning Methods Comparison

**Report Date:** 2025-12-03
**Model:** microsoft/Phi-3.5-mini-instruct
**Dataset:** OpenAssistant Guanaco (2000 samples)
**Device:** CUDA (GPU)

---

## Executive Summary

This report summarizes the training results from three different fine-tuning methods applied to the Phi-3.5-mini-instruct model. All methods were trained on the same dataset (2000 samples from OpenAssistant Guanaco) to enable fair comparison.

## 1. Training Metrics Comparison

### Overview Table

| Method | Training Time | Final Loss | Best Loss | Total Steps | Samples/Sec | Epochs | Trainable % |
|--------|--------------|------------|-------------|-------------|--------|-------------|
| **LoRA** | 85m 58s | 4.3016 | ~0.30| 4,000 | 0.39 | 8 | ~1% |
| **QLoRA** | 63m 40s | 0.6791 | ~0.40 | 2,000 | 0.52 | 8 | ~1% |
| **LowLoRA** | 58m 28s | 1.5765 | ~0.40 | 2,500 | 0.57 | 5 | ~0.1% |

### Key Observations

1. **Best Convergence:** QLoRA achieved the lowest loss (0.6791), indicating the best model convergence
2. **Fastest Training:** LowLoRA completed in 58m 28s, making it ideal for rapid iterations
3. **Most Efficient:** QLoRA had the best throughput at 0.52 samples/second despite using 4-bit quantization
4. **Concerning Result:** LoRA's high loss (4.3016) suggests potential training issues or need for hyperparameter tuning

---

### 2. Questions
1. meta-llama3 didn't run due to authorization issue even after approved. Need to take a closer look at it.
Error:
❌ Failed google/gemma-2-2b-it: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/google/gemma-2-2b-it.

2. Don't know when and how to stop after seeing the loss went up. The number fluctuates all the time.

3. For full training, how to solve out of memory issue.
❌ Failed microsoft/Phi-3.5-mini-instruct: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 15.84 GiB of which 43.75 MiB is free. Including non-PyTorch memory, this process has 15.84 GiB memory in use. Of the allocated memory 15.47 GiB is allocated by PyTorch, and 50.07 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting

## 3. Configuration Comparison

### Hyperparameter Differences

| Hyperparameter | LoRA | QLoRA | LowLoRA |
|---------------|------|-------|---------|
| Batch Size | 2 | 4 | 1 |
| Gradient Accum | 2 | 2 | 4 |
| **Effective Batch** | **4** | **8** | **4** |
| Learning Rate | 3e-4 | 2e-4 | 5e-4 |
| Epochs | 8 | 8 | 5 |
| LoRA Rank | 16-32 | 8-16 | 2-4 |
| Quantization | None | 4-bit | None |

**Key Insights:**
- QLoRA's larger effective batch size (8) contributed to its success
- LowLoRA compensates for low rank with higher learning rate
- LoRA's configuration may need optimization

---

## 4. Appendix: Files Generated

### Model Checkpoints
- `./balanced_hw5_sft/` - LoRA model (⚠️ needs retraining)
- `./qlora_hw5_sft/` - QLoRA model (✅ recommended)
- `./lowlora_hw5_sft/` - LowLoRA model (✅ good for prototyping)

### Metrics Files
- `hw5_training_metrics_report.json` - LoRA detailed metrics
- `hw5_qlora_metrics_report.json` - QLoRA detailed metrics
- `hw5_lowlora_metrics_report.json` - LowLoRA detailed metrics

### Reports
- `lora_report.txt` - LoRA human-readable report
- `lowlora_report.txt` - LowLoRA human-readable report
- `hw5_comprehensive_report.md` - Original comparison report
- `training_summary.md` - This document

