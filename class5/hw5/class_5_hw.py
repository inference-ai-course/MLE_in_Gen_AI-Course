# Balanced Resume Trainer - Practical Middle Ground
# Effective learning with reasonable training time and computational requirements

import traceback
import os
import json
import sys

# Suppress transformers warnings before importing
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class TeeOutput:
    """Class to write output to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

import torch
import warnings
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import time
from datetime import datetime
import logging

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

load_dotenv()

class Class5HWTrainer:
    """Balanced trainer - effective learning with practical constraints"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"⚖️ Class5 homework Trainer")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Goal: Effective learning with reasonable training time")
        print(f"⏱️ Strategy: Quality over quantity - focused, efficient training")
        
    def load_dataset(self, num_samples=1000, split='train'):
        """
        Load the OpenAssistant Guanaco dataset from Hugging Face.

        Args:
            num_samples: Number of samples to load (default: 1000 for efficiency)
            split: Dataset split to load ('train' or 'test')

        Returns:
            List of Q&A pairs in the format [{"question": ..., "answer": ...}, ...]
        """
        print(f"📥 Loading OpenAssistant Guanaco dataset from Hugging Face...")
        print(f"   Split: {split}, Samples: {num_samples}")

        try:
            # Load the dataset
            dataset = load_dataset("timdettmers/openassistant-guanaco", split=split)

            print(f"✅ Dataset loaded: {len(dataset)} total examples")

            # Limit to num_samples for efficiency
            if num_samples and num_samples < len(dataset):
                dataset = dataset.select(range(num_samples))
                print(f"📊 Selected {num_samples} samples for training")

            # Convert to Q&A pairs format
            qa_pairs = []

            for item in dataset:
                # The OpenAssistant Guanaco dataset has 'text' field with conversation format
                text = item['text']

                # Parse the conversation format
                # Format is typically: "### Human: ... ### Assistant: ..."
                if "### Human:" in text and "### Assistant:" in text:
                    parts = text.split("### Assistant:")

                    if len(parts) >= 2:
                        # Extract question (remove "### Human:" prefix)
                        question = parts[0].replace("### Human:", "").strip()

                        # Extract answer (take first assistant response)
                        answer = parts[1].strip()

                        # Clean up any remaining markers
                        if "### Human:" in answer:
                            answer = answer.split("### Human:")[0].strip()

                        qa_pairs.append({
                            "question": question,
                            "answer": answer
                        })

            print(f"✅ Converted {len(qa_pairs)} Q&A pairs")

            # Show sample
            if qa_pairs:
                print("\n📝 Sample Q&A pair:")
                print(f"   Q: {qa_pairs[0]['question'][:100]}...")
                print(f"   A: {qa_pairs[0]['answer'][:100]}...")

            return qa_pairs

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("💡 Tip: Make sure you have internet connection and 'datasets' library installed")
            print("   pip install datasets")

            # Return empty list on error
            return []

    def convert_to_chatml_format(self, qa_pairs, system_message=None):
        """
        Convert Q&A pairs to ChatML format for training.

        ChatML format uses special tokens:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        {answer}<|im_end|>

        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
            system_message: Optional system message (default: helpful assistant)

        Returns:
            List of dicts with 'text' key containing ChatML formatted conversations
        """
        print(f"🔄 Converting {len(qa_pairs)} Q&A pairs to ChatML format...")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

        chatml_data = []

        for qa in qa_pairs:
            # Build ChatML conversation
            conversation = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{qa['question']}<|im_end|>
<|im_start|>assistant
{qa['answer']}<|im_end|>"""

            chatml_data.append({"text": conversation})

        print(f"✅ Converted to ChatML format: {len(chatml_data)} examples")

        # Show sample
        if chatml_data:
            print("\n📝 Sample ChatML format:")
            print(chatml_data[0]['text'][:200] + "...")

        return chatml_data

    def save_chatml_dataset(self, chatml_data, output_file="chatml_dataset_hw5.jsonl"):
        """
        Save ChatML formatted data to JSONL file.

        Args:
            chatml_data: List of dicts with 'text' key
            output_file: Output filename (default: chatml_dataset.jsonl)

        Returns:
            Path to saved file
        """
        print(f"💾 Saving ChatML dataset to {output_file}...")

        with open(output_file, 'w') as f:
            for item in chatml_data:
                f.write(json.dumps(item) + '\n')

        print(f"✅ Saved {len(chatml_data)} examples to {output_file}")
        return output_file

    def setup_balanced_model(self):
        """Setup model with balanced LoRA settings"""
        print("🔧 Setting up model with balanced LoRA settings...")
        
        model_options = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3.5-mini-instruct", 
            "google/gemma-2-2b-it",
            "microsoft/DialoGPT-small"
        ]
        
        for model_name in model_options:
            try:
                print(f"🔄 Trying {model_name}...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
                
                # BALANCED LoRA configuration
                if "llama" in model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Focus on attention
                    r, alpha = 32, 64  # Moderate values
                elif "phi" in model_name.lower():
                    target_modules = ["qkv_proj", "o_proj"]  # Phi-3.5 uses combined qkv_proj
                    r, alpha = 16, 32
                elif "gemma" in model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj"]
                    r, alpha = 16, 32
                else:
                    target_modules = ["c_attn", "c_proj"]
                    r, alpha = 32, 64
                
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=r,  # Balanced rank
                    lora_alpha=alpha,  # Balanced alpha
                    lora_dropout=0.05,  # Small dropout for regularization
                    target_modules=target_modules,
                    bias="none"
                )
                
                model = get_peft_model(model, lora_config)

                # Enable gradient checkpointing to save memory
                model.enable_input_require_grads()
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()

                model.print_trainable_parameters()

                print(f"✅ Loaded {model_name} with balanced LoRA")
                return model, tokenizer, model_name
                
            except Exception as e:
                print(f"❌ Failed {model_name}: {e}")
                continue
        
        raise Exception("No models could be loaded")
    

    def setup_full_finetuning_model(self):
        """Setup model for full fine-tuning (all parameters trainable)"""
        print("🔧 Setting up model for FULL fine-tuning...")
        print("⚠️  Warning: This will train ALL parameters (slower, more memory)")

        model_options = [
            "microsoft/DialoGPT-small",  # Start with smallest model
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "google/gemma-2-2b-it",
        ]

        for model_name in model_options:
            try:
                print(f"🔄 Trying {model_name}...")

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)

                # Make ALL parameters trainable
                for param in model.parameters():
                    param.requires_grad = True

                # Count trainable parameters
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                all_params = sum(p.numel() for p in model.parameters())

                print(f"✅ Loaded {model_name} for full fine-tuning")
                print(f"📊 Trainable parameters: {trainable_params:,} / {all_params:,} (100%)")

                return model, tokenizer, model_name

            except Exception as e:
                print(f"❌ Failed {model_name}: {e}")
                continue

        raise Exception("No models could be loaded")

    def setup_qlora_model(self):
        """Setup model with QLoRA (Quantized LoRA) - 4-bit quantization"""
        print("🔧 Setting up model with QLoRA (4-bit quantization)...")
        print("💡 QLoRA uses 4-bit quantization for even lower memory usage")

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            print("❌ Error: bitsandbytes not installed")
            print("💡 Install with: pip install bitsandbytes")
            raise

        model_options = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "google/gemma-2-2b-it",
            "microsoft/DialoGPT-small"
        ]

        for model_name in model_options:
            try:
                print(f"🔄 Trying {model_name}...")

                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # Normal Float 4
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,  # Nested quantization
                )

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                # Load model with 4-bit quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )

                # Prepare model for k-bit training
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)

                # QLoRA configuration
                if "llama" in model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                    r, alpha = 16, 32  # Smaller than regular LoRA
                elif "phi" in model_name.lower():
                    target_modules = ["qkv_proj", "o_proj"]  # Phi-3.5 uses combined qkv_proj
                    r, alpha = 8, 16
                elif "gemma" in model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj"]
                    r, alpha = 8, 16
                else:
                    target_modules = ["c_attn", "c_proj"]
                    r, alpha = 16, 32

                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=0.05,
                    target_modules=target_modules,
                    bias="none"
                )

                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

                print(f"✅ Loaded {model_name} with QLoRA (4-bit)")
                return model, tokenizer, model_name

            except Exception as e:
                print(f"❌ Failed {model_name}: {e}")
                continue

        raise Exception("No models could be loaded with QLoRA")

    def setup_lowlora_model(self):
        """Setup model with LowLoRA - ultra-low rank for maximum efficiency"""
        print("🔧 Setting up model with LowLoRA (ultra-low rank)...")
        print("💡 LowLoRA uses very small rank (r=2-4) for maximum speed and efficiency")

        model_options = [
            "microsoft/DialoGPT-small",  # Start with smallest for speed
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "google/gemma-2-2b-it",
        ]

        for model_name in model_options:
            try:
                print(f"🔄 Trying {model_name}...")

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)

                # ULTRA-LOW LoRA configuration
                if "llama" in model_name.lower():
                    target_modules = ["q_proj", "v_proj"]  # Minimal modules
                    r, alpha = 4, 8  # Very low rank
                elif "phi" in model_name.lower():
                    target_modules = ["qkv_proj"]  # Phi-3.5 uses combined qkv_proj
                    r, alpha = 2, 4  # Ultra minimal
                elif "gemma" in model_name.lower():
                    target_modules = ["q_proj", "v_proj"]
                    r, alpha = 2, 4
                else:
                    target_modules = ["c_attn"]  # Single module only
                    r, alpha = 4, 8

                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=r,  # Ultra-low rank
                    lora_alpha=alpha,
                    lora_dropout=0.01,  # Minimal dropout
                    target_modules=target_modules,
                    bias="none"
                )

                model = get_peft_model(model, lora_config)

                # Enable gradient checkpointing to save memory
                model.enable_input_require_grads()
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()

                model.print_trainable_parameters()

                print(f"✅ Loaded {model_name} with LowLoRA")
                print(f"💡 Ultra-efficient: r={r}, targeting {len(target_modules)} module(s)")
                return model, tokenizer, model_name

            except Exception as e:
                print(f"❌ Failed {model_name}: {e}")
                continue

        raise Exception("No models could be loaded")

    def train_lowlora(self, model, tokenizer, dataset, model_name="unknown"):
        """Train with LowLoRA - ultra-efficient, fast training"""
        print("⚡ Starting LowLoRA training...")
        print("💡 Ultra-efficient settings for maximum speed")

        output_dir = "./lowlora_hw5_sft"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=1,  # Reduced from 4 to 1 for memory efficiency
            gradient_accumulation_steps=4,  # Increased from 1 to 4 to maintain effective batch size
            warmup_steps=5,
            learning_rate=5e-4,
            fp16=False,
            logging_steps=5,
            save_steps=50,
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,
            dataloader_num_workers=0,
            weight_decay=0.001,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            logging_dir="./logs_lowlora",
        )

        # Use helper to execute training
        trainer, metrics = self._execute_training(
            model, tokenizer, dataset, training_args,
            output_dir, model_name, "lowlora"
        )

        # Print summary
        self._print_training_summary(metrics, output_dir, "LowLoRA Training")

        # Save metrics
        self.save_lowlora_metrics(metrics)

        return trainer, metrics

    def save_lowlora_metrics(self, metrics):
        """Save LowLoRA training metrics to JSON file"""
        metrics_file = "hw5_lowlora_metrics_report.json"

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Detailed metrics saved to: {metrics_file}")

        # Also create a human-readable report
        self.generate_lowlora_report(metrics)

    def generate_lowlora_report(self, metrics):
        """Generate a human-readable LowLoRA training report"""
        report_file = "lowlora_report.txt"

        report = f"""
{'='*80}
LOWLORA TRAINING REPORT (ULTRA-LOW RANK)
{'='*80}

Model Configuration:
-------------------
Model Name: {metrics['model_name']}
Device: {metrics['device']}
Training Type: LowLoRA (Ultra-Low Rank LoRA)
Rank: r=2-4 (minimal)

Training Configuration:
----------------------
Dataset Size: {metrics['dataset_size']} examples
Number of Epochs: {metrics['num_epochs']}
Batch Size: {metrics['batch_size']}
Gradient Accumulation Steps: {metrics['gradient_accumulation_steps']}
Effective Batch Size: {metrics['effective_batch_size']}
Learning Rate: {metrics['learning_rate']}

Training Timeline:
-----------------
Start Time: {metrics['start_time']}
End Time: {metrics['end_time']}
Total Duration: {metrics['training_duration_formatted']} ({metrics['training_duration_seconds']}s)

Training Metrics:
----------------
Final Training Loss: {metrics['final_loss']}
Total Training Steps: {metrics['total_steps']}
Throughput: {metrics['samples_per_second']} samples/second

Performance Analysis:
--------------------
"""

        # Add performance analysis
        if metrics['training_duration_minutes'] < 5:
            report += "✅ LowLoRA training completed very quickly (< 5 minutes)\n"
        elif metrics['training_duration_minutes'] < 15:
            report += "✅ LowLoRA training completed quickly (< 15 minutes)\n"
        else:
            report += "⚠️  LowLoRA training took longer than expected (> 15 minutes)\n"

        if metrics['final_loss'] and metrics['final_loss'] < 1.0:
            report += "✅ Model converged well (loss < 1.0)\n"
        elif metrics['final_loss'] and metrics['final_loss'] < 2.0:
            report += "⚠️  Model converged moderately (loss < 2.0)\n"
        else:
            report += "⚠️  Model may need more training (loss >= 2.0)\n"

        report += f"\nModel Output Location:\n"
        report += f"----------------------\n"
        report += f"Model Directory: ./lowlora_hw5_sft\n"
        report += f"Metrics File: hw5_lowlora_metrics_report.json\n"
        report += f"Report File: {report_file}\n"

        report += f"\nLowLoRA Advantages:\n"
        report += f"------------------\n"
        report += f"- Ultra-low rank (r=2-4) for minimal parameters\n"
        report += f"- Fastest training among all LoRA variants\n"
        report += f"- Lowest memory footprint\n"
        report += f"- Good for rapid prototyping and experimentation\n"
        report += f"- Ideal for limited hardware\n"

        report += f"\nTrade-offs:\n"
        report += f"-----------\n"
        report += f"- May have slightly lower quality than higher-rank LoRA\n"
        report += f"- Best for simple tasks or quick iterations\n"
        report += f"- Consider regular LoRA if quality is critical\n"

        report += f"\nComparison Table:\n"
        report += f"-----------------\n"
        report += f"Method          | Rank | Trainable % | Speed    | Memory  | Quality\n"
        report += f"----------------|------|-------------|----------|---------|--------\n"
        report += f"Full Finetune   | N/A  | 100%        | Slowest  | Highest | Best\n"
        report += f"LoRA            | 16-32| ~1%         | Fast     | Medium  | Good\n"
        report += f"QLoRA           | 8-16 | ~1%         | Fast     | Low     | Good\n"
        report += f"LowLoRA (This)  | 2-4  | ~0.1%       | Fastest  | Lowest  | Fair\n"

        report += f"\n{'='*80}\n"
        report += f"Report Generated: {metrics['end_time']}\n"
        report += f"{'='*80}\n"

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📄 LowLoRA report saved to: {report_file}")

        # Print the report to console as well
        print(report)

    def train_qlora(self, model, tokenizer, dataset, model_name="unknown"):
        """Train with QLoRA - 4-bit quantized LoRA"""
        print("🔥 Starting QLoRA training...")
        print("💡 Using 4-bit quantization for minimal memory usage")

        output_dir = "./qlora_hw5_sft"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=8,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=2,
            save_steps=20,
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,
            dataloader_num_workers=0,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_dir="./logs_qlora",
        )

        # Use helper to execute training
        trainer, metrics = self._execute_training(
            model, tokenizer, dataset, training_args,
            output_dir, model_name, "qlora"
        )

        # Add QLoRA-specific metadata
        metrics["quantization"] = "4-bit"

        # Print summary
        self._print_training_summary(metrics, output_dir, "QLoRA Training")

        # Save metrics
        self.save_qlora_metrics(metrics)

        return trainer, metrics

    def save_qlora_metrics(self, metrics):
        """Save QLoRA training metrics to JSON file"""
        metrics_file = "hw5_qlora_metrics_report.json"

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Detailed metrics saved to: {metrics_file}")

        # Also create a human-readable report
        self.generate_qlora_report(metrics)

    def generate_qlora_report(self, metrics):
        """Generate a human-readable QLoRA training report"""
        report_file = "qlora_report.txt"

        report = f"""
{'='*80}
QLORA TRAINING REPORT (4-BIT QUANTIZATION)
{'='*80}

Model Configuration:
-------------------
Model Name: {metrics['model_name']}
Device: {metrics['device']}
Training Type: QLoRA (Quantized LoRA)
Quantization: {metrics['quantization']}

Training Configuration:
----------------------
Dataset Size: {metrics['dataset_size']} examples
Number of Epochs: {metrics['num_epochs']}
Batch Size: {metrics['batch_size']}
Gradient Accumulation Steps: {metrics['gradient_accumulation_steps']}
Effective Batch Size: {metrics['effective_batch_size']}
Learning Rate: {metrics['learning_rate']}

Training Timeline:
-----------------
Start Time: {metrics['start_time']}
End Time: {metrics['end_time']}
Total Duration: {metrics['training_duration_formatted']} ({metrics['training_duration_seconds']}s)

Training Metrics:
----------------
Final Training Loss: {metrics['final_loss']}
Total Training Steps: {metrics['total_steps']}
Throughput: {metrics['samples_per_second']} samples/second

Performance Analysis:
--------------------
"""

        # Add performance analysis
        if metrics['training_duration_minutes'] < 10:
            report += "✅ QLoRA training completed quickly (< 10 minutes)\n"
        elif metrics['training_duration_minutes'] < 30:
            report += "✅ QLoRA training completed in reasonable time (< 30 minutes)\n"
        else:
            report += "⚠️  QLoRA training took longer than expected (> 30 minutes)\n"

        if metrics['final_loss'] and metrics['final_loss'] < 1.0:
            report += "✅ Model converged well (loss < 1.0)\n"
        elif metrics['final_loss'] and metrics['final_loss'] < 2.0:
            report += "⚠️  Model converged moderately (loss < 2.0)\n"
        else:
            report += "⚠️  Model may need more training (loss >= 2.0)\n"

        report += f"\nModel Output Location:\n"
        report += f"----------------------\n"
        report += f"Model Directory: ./qlora_hw5_sft\n"
        report += f"Metrics File: {metrics_file}\n"
        report += f"Report File: {report_file}\n"

        report += f"\nQLoRA Advantages:\n"
        report += f"-----------------\n"
        report += f"- Uses 4-bit quantization (75% less memory than LoRA)\n"
        report += f"- Can use larger batch sizes on same hardware\n"
        report += f"- Enables fine-tuning larger models on limited hardware\n"
        report += f"- Minimal performance loss compared to regular LoRA\n"
        report += f"- Same adapter size as LoRA (~1% parameters)\n"

        report += f"\nComparison:\n"
        report += f"-----------\n"
        report += f"- Full Fine-Tuning: 100% parameters, highest memory\n"
        report += f"- LoRA: ~1% parameters, moderate memory\n"
        report += f"- QLoRA: ~1% parameters, lowest memory (4-bit quantized)\n"

        report += f"\n{'='*80}\n"
        report += f"Report Generated: {metrics['end_time']}\n"
        report += f"{'='*80}\n"

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📄 QLoRA report saved to: {report_file}")

        # Print the report to console as well
        print(report)

    def train_full_finetuning(self, model, tokenizer, dataset, model_name="unknown"):
        """Train with full fine-tuning - all parameters are updated"""
        print("🔥 Starting FULL fine-tuning...")
        print("⚠️  This will take longer and use more memory than LoRA")

        output_dir = "./full_finetuning_hw5_sft"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=1e-5,
            fp16=False,
            logging_steps=2,
            save_steps=20,
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,
            dataloader_num_workers=0,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_dir="./logs_full",
        )

        # Use helper to execute training
        trainer, metrics = self._execute_training(
            model, tokenizer, dataset, training_args,
            output_dir, model_name, "full_finetuning"
        )

        # Print summary
        self._print_training_summary(metrics, output_dir, "Full Fine-Tuning")

        # Save metrics
        self.save_full_finetuning_metrics(metrics)

        return trainer, metrics

    def save_full_finetuning_metrics(self, metrics):
        """Save full fine-tuning metrics to JSON file"""
        metrics_file = "hw5_full_finetuning_metrics_report.json"

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Detailed metrics saved to: {metrics_file}")

        # Also create a human-readable report
        self.generate_full_finetuning_report(metrics)

    def generate_full_finetuning_report(self, metrics):
        """Generate a human-readable full fine-tuning report"""
        report_file = "full_finetuning_report.txt"

        report = f"""
{'='*80}
FULL FINE-TUNING TRAINING REPORT
{'='*80}

Model Configuration:
-------------------
Model Name: {metrics['model_name']}
Device: {metrics['device']}
Training Type: Full Fine-Tuning (ALL parameters trained)

Training Configuration:
----------------------
Dataset Size: {metrics['dataset_size']} examples
Number of Epochs: {metrics['num_epochs']}
Batch Size: {metrics['batch_size']}
Gradient Accumulation Steps: {metrics['gradient_accumulation_steps']}
Effective Batch Size: {metrics['effective_batch_size']}
Learning Rate: {metrics['learning_rate']}

Training Timeline:
-----------------
Start Time: {metrics['start_time']}
End Time: {metrics['end_time']}
Total Duration: {metrics['training_duration_formatted']} ({metrics['training_duration_seconds']}s)

Training Metrics:
----------------
Final Training Loss: {metrics['final_loss']}
Total Training Steps: {metrics['total_steps']}
Throughput: {metrics['samples_per_second']} samples/second

Performance Analysis:
--------------------
"""

        # Add performance analysis
        if metrics['training_duration_minutes'] < 15:
            report += "✅ Full fine-tuning completed quickly (< 15 minutes)\n"
        elif metrics['training_duration_minutes'] < 45:
            report += "✅ Full fine-tuning completed in reasonable time (< 45 minutes)\n"
        else:
            report += "⚠️  Full fine-tuning took longer than expected (> 45 minutes)\n"

        if metrics['final_loss'] and metrics['final_loss'] < 1.0:
            report += "✅ Model converged well (loss < 1.0)\n"
        elif metrics['final_loss'] and metrics['final_loss'] < 2.0:
            report += "⚠️  Model converged moderately (loss < 2.0)\n"
        else:
            report += "⚠️  Model may need more training (loss >= 2.0)\n"

        report += f"\nModel Output Location:\n"
        report += f"----------------------\n"
        report += f"Model Directory: ./full_finetuning_hw5_sft\n"
        report += f"Metrics File: {metrics_file}\n"
        report += f"Report File: {report_file}\n"

        report += f"\nComparison Notes:\n"
        report += f"-----------------\n"
        report += f"- Full fine-tuning trains ALL model parameters (100%)\n"
        report += f"- LoRA fine-tuning trains only ~1% of parameters\n"
        report += f"- Full fine-tuning typically achieves better performance but:\n"
        report += f"  * Takes longer to train\n"
        report += f"  * Requires more memory\n"
        report += f"  * Produces larger saved models\n"
        report += f"  * Higher risk of overfitting on small datasets\n"

        report += f"\n{'='*80}\n"
        report += f"Report Generated: {metrics['end_time']}\n"
        report += f"{'='*80}\n"

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📄 Full fine-tuning report saved to: {report_file}")

        # Print the report to console as well
        print(report)

    def _execute_training(self, model, tokenizer, dataset, training_args, output_dir, model_name, training_type):
        """
        Common training execution logic to reduce code duplication.

        Args:
            model: The model to train
            tokenizer: Tokenizer for the model
            dataset: Training dataset
            training_args: TrainingArguments configuration
            output_dir: Directory to save the model
            model_name: Name of the model being trained
            training_type: Type of training (lora, full_finetuning, qlora, lowlora)

        Returns:
            Tuple of (trainer, metrics dict)
        """
        # Track start time
        start_time = time.time()
        start_datetime = datetime.now()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Estimate training time
        estimated_time = (len(dataset) * training_args.num_train_epochs) // (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        )
        print(f"⏱️ Estimated training time: ~{estimated_time//60 + 1} minutes")

        # Train the model
        print(f"\n🚀 Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        train_result = trainer.train()

        # Calculate training time
        end_time = time.time()
        training_duration = end_time - start_time
        end_datetime = datetime.now()

        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Collect metrics
        metrics = {
            "model_name": model_name,
            "training_type": training_type,
            "device": str(self.device),
            "dataset_size": len(dataset),
            "num_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "training_duration_seconds": round(training_duration, 2),
            "training_duration_minutes": round(training_duration / 60, 2),
            "training_duration_formatted": f"{int(training_duration // 60)}m {int(training_duration % 60)}s",
            "final_loss": round(train_result.training_loss, 4) if hasattr(train_result, 'training_loss') else None,
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else None,
            "samples_per_second": round(len(dataset) / training_duration, 2) if training_duration > 0 else None,
        }

        return trainer, metrics

    def _print_training_summary(self, metrics, output_dir, training_name):
        """Print training summary to console."""
        print("\n" + "="*80)
        print(f"📊 {training_name.upper()} METRICS SUMMARY")
        print("="*80)
        print(f"✅ Training completed successfully!")
        print(f"⏱️  Total Training Time: {metrics['training_duration_formatted']}")
        print(f"📊 Final Loss: {metrics['final_loss']}")
        print(f"🔢 Total Steps: {metrics['total_steps']}")
        print(f"⚡ Samples/Second: {metrics['samples_per_second']}")
        print(f"💾 Model saved to: {output_dir}")

    def _save_metrics_and_report(self, metrics, metrics_file, report_file, report_generator):
        """Save metrics to JSON and generate report."""
        # Save metrics JSON
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Detailed metrics saved to: {metrics_file}")

        # Generate report
        report_generator(metrics, report_file)

    def create_efficient_dataset(self, qa_pairs, tokenizer):
        """Create efficient dataset with moderate repetition"""
        print("📚 Creating efficient training dataset...")

        formatted_data = []
        
        for qa in qa_pairs:
            # Use 2 formats per Q&A (balanced variety)
            formats = [
                f"Human: {qa['question']}\nAssistant: {qa['answer']}<|endoftext|>",
                f"Question: {qa['question']}\nAnswer: {qa['answer']}<|endoftext|>"
            ]
            
            for format_text in formats:
                formatted_data.append({"text": format_text})
        
        print(f"📊 Dataset size: {len(formatted_data)} examples")
        
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=384,  # Moderate length
                return_tensors=None
            )
            
            result["labels"] = result["input_ids"].copy()
            
            # Mask padding tokens
            result["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels]
                for labels in result["labels"]
            ]
            
            return result
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=["text"],
            batched=True
        )
        
        print(f"✅ Efficient dataset ready: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train_balanced(self, model, tokenizer, dataset, model_name="unknown"):
        """Train with balanced settings - effective but not extreme"""
        print("⚖️ Starting balanced training...")
        print("🎯 Using moderate settings for effective learning")

        output_dir = "./balanced_hw5_sft"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=8,
            per_device_train_batch_size=1,  # Reduced to 1 for memory efficiency
            gradient_accumulation_steps=4,  # Increased to maintain effective batch size
            warmup_steps=10,
            learning_rate=3e-4,
            fp16=False,
            logging_steps=2,
            save_steps=20,
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,
            dataloader_num_workers=0,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_dir="./logs",
        )

        # Use helper to execute training
        trainer, metrics = self._execute_training(
            model, tokenizer, dataset, training_args,
            output_dir, model_name, "lora"
        )

        # Print summary
        self._print_training_summary(metrics, output_dir, "LoRA Training")

        # Save metrics
        self.save_training_metrics(metrics)

        return trainer, metrics

    def save_training_metrics(self, metrics):
        """Save training metrics to JSON file"""
        metrics_file = "hw5_training_metrics_report.json"

        # Add LoRA configuration if available
        try:
            if hasattr(self, 'lora_config'):
                metrics['lora_config'] = {
                    'r': self.lora_config.r,
                    'lora_alpha': self.lora_config.lora_alpha,
                    'lora_dropout': self.lora_config.lora_dropout,
                }
        except:
            pass

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Detailed metrics saved to: {metrics_file}")

        # Also create a human-readable report
        self.generate_training_report(metrics)

    def generate_training_report(self, metrics):
        """Generate a human-readable training report"""
        report_file = "training_report.txt"

        report = f"""
{'='*80}
FINE-TUNING TRAINING REPORT
{'='*80}

Model Configuration:
-------------------
Model Name: {metrics['model_name']}
Device: {metrics['device']}
LoRA Fine-Tuning: Yes

Training Configuration:
----------------------
Dataset Size: {metrics['dataset_size']} examples
Number of Epochs: {metrics['num_epochs']}
Batch Size: {metrics['batch_size']}
Gradient Accumulation Steps: {metrics['gradient_accumulation_steps']}
Effective Batch Size: {metrics['batch_size'] * metrics['gradient_accumulation_steps']}
Learning Rate: {metrics['learning_rate']}

Training Timeline:
-----------------
Start Time: {metrics['start_time']}
End Time: {metrics['end_time']}
Total Duration: {metrics['training_duration_formatted']} ({metrics['training_duration_seconds']}s)

Training Metrics:
----------------
Final Training Loss: {metrics['final_loss']}
Total Training Steps: {metrics['total_steps']}
Throughput: {metrics['samples_per_second']} samples/second

Performance Analysis:
--------------------
"""

        # Add performance analysis
        if metrics['training_duration_minutes'] < 10:
            report += "✅ Training completed quickly (< 10 minutes)\n"
        elif metrics['training_duration_minutes'] < 30:
            report += "✅ Training completed in reasonable time (< 30 minutes)\n"
        else:
            report += "⚠️  Training took longer than expected (> 30 minutes)\n"

        if metrics['final_loss'] and metrics['final_loss'] < 1.0:
            report += "✅ Model converged well (loss < 1.0)\n"
        elif metrics['final_loss'] and metrics['final_loss'] < 2.0:
            report += "⚠️  Model converged moderately (loss < 2.0)\n"
        else:
            report += "⚠️  Model may need more training (loss >= 2.0)\n"

        report += f"\nModel Output Location:\n"
        report += f"----------------------\n"
        report += f"Model Directory: ./balanced_hw5_sft\n"
        report += f"Metrics File: training_metrics_report.json\n"
        report += f"Report File: {report_file}\n"

        report += f"\n{'='*80}\n"
        report += f"Report Generated: {metrics['end_time']}\n"
        report += f"{'='*80}\n"

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📄 Training report saved to: {report_file}")

        # Print the report to console as well
        print(report)
    
    def test_balanced_model(self):
        """Test the balanced model"""
        print("🧪 Testing balanced model...")

        try:
            tokenizer = AutoTokenizer.from_pretrained("./balanced_hw5_sft")
            model = AutoModelForCausalLM.from_pretrained(
                "./balanced_hw5_sft",
                torch_dtype=torch.float32
            ).to(self.device)

            test_questions = [
                "Are you an AI assistant?",
                "What programming languages do you know?",
                "Tell me about your work experience",
                "What's your most significant project?",
                "What are your technical skills?"
            ]

            print("🔍 Testing responses...")

            for question in test_questions:
                print(f"\n❓ Question: {question}")

                prompt = f"Human: {question}\nAssistant:"
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=80,  # Moderate length
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                assistant_response = response[len(prompt):].strip()

                print(f"⚖️ Balanced Answer: {assistant_response}")

                # Quality check
                if "AI" in assistant_response and "assistant" in assistant_response.lower():
                    print("  ⚠️ Still some AI language detected")
                elif len(assistant_response) < 10:
                    print("  ⚠️ Response seems too short")
                else:
                    print("  ✅ Response looks good!")

        except Exception as e:
            print(f"❌ Testing failed: {e}")

    def generate_comprehensive_report(self, all_metrics, test_outputs=None):
        """
        Generate a comprehensive report comparing all training methods.

        Args:
            all_metrics: Dict with keys 'lora', 'full', 'qlora', 'lowlora' containing metrics
            test_outputs: Optional dict with test responses from each model
        """
        report_file = "hw5_comprehensive_report.md"

        print("\n📝 Generating comprehensive comparison report...")

        report = f"""# Fine-Tuning Methods Comparison Report
## Class 5 Homework - LLM Fine-Tuning Analysis

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Device:** {self.device}
**Dataset:** OpenAssistant Guanaco (1000 samples)

---

## Executive Summary

This report compares four different fine-tuning approaches for Large Language Models:
1. **LoRA** (Low-Rank Adaptation) - Balanced approach
2. **Full Fine-Tuning** - Training all parameters
3. **QLoRA** (Quantized LoRA) - 4-bit quantization
4. **LowLoRA** - Ultra-low rank for speed

---

## 1. Training Metrics Comparison

### Performance Overview

| Method | Training Time | Final Loss | Steps | Samples/Sec | Epochs |
|--------|---------------|------------|-------|-------------|--------|
"""

        # Add metrics table
        for method_name, method_key in [('LoRA', 'lora'), ('Full Fine-Tuning', 'full'),
                                         ('QLoRA', 'qlora'), ('LowLoRA', 'lowlora')]:
            if method_key in all_metrics:
                m = all_metrics[method_key]
                report += f"| {method_name} | {m.get('training_duration_formatted', 'N/A')} | "
                report += f"{m.get('final_loss', 'N/A')} | {m.get('total_steps', 'N/A')} | "
                report += f"{m.get('samples_per_second', 'N/A')} | {m.get('num_epochs', 'N/A')} |\n"

        report += """
### Training Configuration Comparison

| Method | Batch Size | Grad Accum | Eff. Batch | Learning Rate | Trainable % |
|--------|------------|------------|------------|---------------|-------------|
"""

        trainable_pct = {
            'lora': '~1%',
            'full': '100%',
            'qlora': '~1%',
            'lowlora': '~0.1%'
        }

        for method_name, method_key in [('LoRA', 'lora'), ('Full', 'full'),
                                         ('QLoRA', 'qlora'), ('LowLoRA', 'lowlora')]:
            if method_key in all_metrics:
                m = all_metrics[method_key]
                report += f"| {method_name} | {m.get('batch_size', 'N/A')} | "
                report += f"{m.get('gradient_accumulation_steps', 'N/A')} | "
                report += f"{m.get('effective_batch_size', 'N/A')} | "
                report += f"{m.get('learning_rate', 'N/A')} | {trainable_pct.get(method_key, 'N/A')} |\n"

        report += """
---

## 2. Model Outputs Analysis
"""

        if test_outputs:
            report += "\n### Sample Responses\n\n"
            for question, responses in test_outputs.items():
                report += f"\n**Question:** {question}\n\n"
                for method, response in responses.items():
                    report += f"- **{method}:** {response}\n"
        else:
            report += "\n*Run models to generate sample outputs*\n"

        report += """
---

## 3. Overfitting Analysis

### Indicators to Monitor:

**Training Loss Trends:**
"""

        # Analyze training loss
        losses = {k: v.get('final_loss') for k, v in all_metrics.items() if v.get('final_loss')}
        if losses:
            min_loss = min(losses.values())
            max_loss = max(losses.values())

            report += f"\n- **Lowest Loss:** {min_loss} (better convergence)\n"
            report += f"- **Highest Loss:** {max_loss}\n"
            report += f"- **Loss Range:** {max_loss - min_loss:.4f}\n\n"

            report += "**Overfitting Risk Assessment:**\n\n"

            for method, loss in losses.items():
                if loss < 0.5:
                    risk = "⚠️ HIGH - Very low loss may indicate overfitting"
                elif loss < 1.0:
                    risk = "✅ LOW - Good convergence"
                elif loss < 2.0:
                    risk = "⚠️ MEDIUM - May need more training"
                else:
                    risk = "❌ HIGH - Poor convergence"
                report += f"- **{method.upper()}:** Loss {loss:.4f} - {risk}\n"

        report += """

### Recommendations to Prevent Overfitting:

1. **Early Stopping** - Monitor validation loss, stop when it plateaus
2. **Regularization** - Weight decay, dropout (already applied)
3. **Data Augmentation** - Increase dataset size or diversity
4. **Lower Learning Rate** - Reduce to prevent over-optimization
5. **Fewer Epochs** - Especially for Full Fine-Tuning

---

## 4. Style Transfer Analysis

### Dataset Influence:

The model was fine-tuned on **OpenAssistant Guanaco dataset**, which contains:
- Conversational Q&A pairs
- Helpful assistant responses
- Natural language patterns

### Expected Style Changes:

1. **Response Format** - Should follow "Human:/Assistant:" pattern
2. **Tone** - Helpful and informative
3. **Content** - Factual and detailed responses

### Style Transfer Effectiveness:

"""

        # Analyze by method
        report += """
**By Training Method:**

- **Full Fine-Tuning:**
  - ✅ Strongest style transfer
  - ✅ Learns dataset patterns deeply
  - ⚠️ Risk: May lose general knowledge

- **LoRA:**
  - ✅ Balanced style transfer
  - ✅ Preserves base model knowledge
  - ✅ Best trade-off

- **QLoRA:**
  - ✅ Similar to LoRA
  - ✅ Memory efficient
  - ⚠️ Slightly lower precision (4-bit)

- **LowLoRA:**
  - ⚠️ Lighter style transfer
  - ✅ Fastest training
  - ⚠️ May need more epochs for strong adaptation

---

## 5. Performance Analysis

### Speed Ranking (Fastest to Slowest):
"""

        # Sort by training duration
        duration_sorted = sorted(
            [(k, v.get('training_duration_minutes', float('inf')))
             for k, v in all_metrics.items()],
            key=lambda x: x[1]
        )

        for i, (method, duration) in enumerate(duration_sorted, 1):
            report += f"\n{i}. **{method.upper()}** - {duration:.2f} minutes"

        report += "\n\n### Quality Ranking (Best to Worst Loss):\n"

        # Sort by loss
        loss_sorted = sorted(
            [(k, v.get('final_loss', float('inf')))
             for k, v in all_metrics.items() if v.get('final_loss')],
            key=lambda x: x[1]
        )

        for i, (method, loss) in enumerate(loss_sorted, 1):
            report += f"\n{i}. **{method.upper()}** - {loss:.4f} loss"

        report += """

### Memory Efficiency Ranking:

1. **LowLoRA** - Ultra-low memory (~0.1% params)
2. **QLoRA** - 4-bit quantization (~1% params)
3. **LoRA** - Low-rank adaptation (~1% params)
4. **Full Fine-Tuning** - All parameters (100%)

---

## 6. Recommendations

### Use Case Recommendations:

"""

        report += """
| Use Case | Recommended Method | Reasoning |
|----------|-------------------|-----------|
| **Production Deployment** | Full Fine-Tuning | Best quality, worth the cost |
| **Research/Development** | LoRA | Best balance of quality/speed/cost |
| **Limited GPU Memory** | QLoRA | Enables training on smaller hardware |
| **Rapid Prototyping** | LowLoRA | Fastest iterations for testing |
| **Edge Deployment** | QLoRA or LowLoRA | Smaller model size |

### Best Practices:

1. **Start with LowLoRA** for quick validation
2. **Scale to LoRA** for production-quality results
3. **Use QLoRA** if memory-constrained
4. **Reserve Full Fine-Tuning** for final deployment or critical applications

---

## 7. Conclusions

### Key Findings:

"""

        # Calculate some insights
        if 'lora' in all_metrics and 'full' in all_metrics:
            lora_time = all_metrics['lora'].get('training_duration_minutes', 0)
            full_time = all_metrics['full'].get('training_duration_minutes', 0)
            if full_time > 0:
                speedup = full_time / lora_time if lora_time > 0 else 0
                report += f"\n- LoRA is **{speedup:.1f}x faster** than Full Fine-Tuning\n"

        if 'lowlora' in all_metrics and 'lora' in all_metrics:
            low_time = all_metrics['lowlora'].get('training_duration_minutes', 0)
            lora_time = all_metrics['lora'].get('training_duration_minutes', 0)
            if lora_time > 0:
                speedup = lora_time / low_time if low_time > 0 else 0
                report += f"- LowLoRA is **{speedup:.1f}x faster** than regular LoRA\n"

        report += """
- All methods successfully fine-tuned on Guanaco dataset
- Trade-off exists between speed, memory, and quality
- LoRA variants provide excellent efficiency without major quality loss

### Future Work:

1. Test on larger datasets (10k+ samples)
2. Implement validation set monitoring
3. Compare on downstream tasks
4. Measure inference latency
5. Evaluate on domain-specific benchmarks

---

## 8. Files Generated

**Models:**
- `./balanced_hw5_sft/` - LoRA model
- `./full_finetuning_hw5_sft/` - Full fine-tuned model
- `./qlora_hw5_sft/` - QLoRA model
- `./lowlora_hw5_sft/` - LowLoRA model

**Metrics:**
- `hw5_training_metrics_report.json` - LoRA metrics
- `hw5_full_finetuning_metrics_report.json` - Full metrics
- `hw5_qlora_metrics_report.json` - QLoRA metrics
- `hw5_lowlora_metrics_report.json` - LowLoRA metrics

**Reports:**
- `training_report.txt` - LoRA report
- `full_finetuning_report.txt` - Full report
- `qlora_report.txt` - QLoRA report
- `lowlora_report.txt` - LowLoRA report
- `hw5_comprehensive_report.md` - This report

---

*Report generated by Class5HWTrainer*
*Model: Various | Dataset: OpenAssistant Guanaco | Date: """ + datetime.now().strftime('%Y-%m-%d') + "*\n"

        # Write report
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✅ Comprehensive report saved to: {report_file}")
        return report_file

def main(quick_mode=False, num_samples=None, methods=None):
    """
    Main balanced training function

    Args:
        quick_mode: If True, uses reduced settings for faster training
        num_samples: Number of samples to load (default: 1000 normal, 100 quick)
        methods: List of methods to run ['lora', 'full', 'qlora', 'lowlora'] or None for all
    """
    if quick_mode:
        print("⚡ QUICK MODE ENABLED - FASTER TRAINING")
        print("=" * 40)
        print("🎯 Goal: Fast testing and validation")
        print("⏱️ Expected time: 2-5 minutes")
        print("💻 Computation: Minimal (reduced settings)")
        if num_samples is None:
            num_samples = 100  # Much smaller dataset
    else:
        print("⚖️ BALANCED RESUME TRAINING")
        print("=" * 40)
        print("🎯 Goal: Effective learning with practical constraints")
        print("⏱️ Expected time: 30-60 minutes")
        print("💻 Computation: Moderate (balanced settings)")
        if num_samples is None:
            num_samples = 1000

    # Default to all methods if not specified
    if methods is None:
        methods = ['lora', 'full', 'qlora', 'lowlora']

    print(f"\n📊 Training methods: {', '.join(methods)}")
    print(f"📊 Dataset samples: {'ALL (no limit)' if num_samples is None else num_samples}")

    trainer = Class5HWTrainer()

    try:
        # Step 1: Load OpenAssistant Guanaco dataset
        print("\n📥 LOADING OPENASSISTANT GUANACO DATASET")
        if num_samples is None:
            print("⚠️  Loading ALL available data (this may take longer)...")
        guanaco_qa_pairs = trainer.load_dataset(
            num_samples=num_samples,
            split='train'
        )

        # Step 2: Create and save the data in chatML format
        if guanaco_qa_pairs:
            print(f"\n✅ Loaded {len(guanaco_qa_pairs)} Q&A pairs from Guanaco")

            print("\n🔄 CONVERTING TO CHATML FORMAT")
            chatml_data = trainer.convert_to_chatml_format(
                guanaco_qa_pairs,
                system_message="You are a helpful AI assistant."
            )

            # Save ChatML dataset
            trainer.save_chatml_dataset(chatml_data, "guanaco_chatml.jsonl")

            # Initialize metrics dict
            all_metrics = {}

            # Step 3-6: LoRA Training
            if 'lora' in methods:
                print("\n" + "="*80)
                print("🔧 STEP 3-6: LORA TRAINING")
                print("="*80)

                print("\n🔧 SETTING UP BALANCED MODEL")
                model, tokenizer, model_name = trainer.setup_balanced_model()

                print("\n📚 CREATING EFFICIENT DATASET")
                dataset = trainer.create_efficient_dataset(guanaco_qa_pairs, tokenizer)

                print("\n⚖️ STARTING FINE-TUNING WITH LORA")
                trainer_obj, metrics = trainer.train_balanced(model, tokenizer, dataset, model_name)

                print("\n✅ FINE-TUNING COMPLETED!")
                print(f"📊 Check 'hw5_training_metrics_report.json' for detailed metrics")
                print(f"📄 Check 'training_report.txt' for human-readable report")

                # Test the fine-tuned model
                print("\n🧪 TESTING BALANCED RESULTS")
                trainer.test_balanced_model()

                print("\n⚖️ LORA PIPELINE FINISHED!")
                print(f"✅ Model: {model_name}")
                print(f"🎯 Training Duration: {metrics['training_duration_formatted']}")
                print(f"📊 Final Loss: {metrics['final_loss']}")
                print(f"💾 Model saved to: ./balanced_hw5_sft")

                all_metrics['lora'] = metrics
            else:
                print("\n⏭️  Skipping LoRA training")
                metrics = None

            # Step 7: Full Fine-Tuning (Optional)
            if 'full' in methods:
                print("\n" + "="*80)
                print("🔥 STEP 7: FULL FINE-TUNING (ALL PARAMETERS)")
                print("="*80)
                print("⚠️  This will train ALL parameters (slower, more memory)")
                print("📊 Compare with LoRA results above")

                # Setup model for full fine-tuning
                print("\n🔧 SETTING UP MODEL FOR FULL FINE-TUNING")
                model_full, tokenizer_full, model_name_full = trainer.setup_full_finetuning_model()

                # Create dataset for full fine-tuning
                print("\n📚 CREATING DATASET FOR FULL FINE-TUNING")
                dataset_full = trainer.create_efficient_dataset(guanaco_qa_pairs, tokenizer_full)

                # Run full fine-tuning
                print("\n🔥 STARTING FULL FINE-TUNING")
                trainer_full, metrics_full = trainer.train_full_finetuning(
                    model_full, tokenizer_full, dataset_full, model_name_full
                )

                print("\n✅ FULL FINE-TUNING COMPLETED!")
                print(f"📊 Check 'hw5_full_finetuning_metrics_report.json' for detailed metrics")
                print(f"📄 Check 'full_finetuning_report.txt' for human-readable report")

                all_metrics['full'] = metrics_full
            else:
                print("\n⏭️  Skipping Full Fine-Tuning")
                metrics_full = None

            # Step 8: QLoRA Training (4-bit Quantization)
            if 'qlora' in methods:
                print("\n" + "="*80)
                print("🔥 STEP 8: QLORA TRAINING (4-BIT QUANTIZATION)")
                print("="*80)
                print("💡 QLoRA uses 4-bit quantization for minimal memory")
                print("📊 Compare memory usage with other methods")

                try:
                    # Setup model for QLoRA
                    print("\n🔧 SETTING UP MODEL FOR QLORA")
                    model_qlora, tokenizer_qlora, model_name_qlora = trainer.setup_qlora_model()

                    # Create dataset for QLoRA
                    print("\n📚 CREATING DATASET FOR QLORA")
                    dataset_qlora = trainer.create_efficient_dataset(guanaco_qa_pairs, tokenizer_qlora)

                    # Run QLoRA training
                    print("\n🔥 STARTING QLORA TRAINING")
                    trainer_qlora, metrics_qlora = trainer.train_qlora(
                        model_qlora, tokenizer_qlora, dataset_qlora, model_name_qlora
                    )

                    print("\n✅ QLORA TRAINING COMPLETED!")
                    print(f"📊 Check 'hw5_qlora_metrics_report.json' for detailed metrics")
                    print(f"📄 Check 'qlora_report.txt' for human-readable report")

                    all_metrics['qlora'] = metrics_qlora

                except Exception as e:
                    print(f"⚠️  QLoRA training skipped: {e}")
                    print("💡 Tip: Install bitsandbytes with: pip install bitsandbytes")
                    model_name_qlora = "N/A"
                    metrics_qlora = {"training_duration_formatted": "N/A", "final_loss": "N/A"}
                    all_metrics['qlora'] = metrics_qlora
            else:
                print("\n⏭️  Skipping QLoRA training")
                metrics_qlora = {"training_duration_formatted": "N/A", "final_loss": "N/A"}

            # Step 9: LowLoRA Training (Ultra-Low Rank)
            if 'lowlora' in methods:
                print("\n" + "="*80)
                print("⚡ STEP 9: LOWLORA TRAINING (ULTRA-LOW RANK)")
                print("="*80)
                print("💡 LowLoRA uses r=2-4 for maximum speed and efficiency")
                print("📊 Compare training speed with other methods")

                # Setup model for LowLoRA
                print("\n🔧 SETTING UP MODEL FOR LOWLORA")
                model_lowlora, tokenizer_lowlora, model_name_lowlora = trainer.setup_lowlora_model()

                # Create dataset for LowLoRA
                print("\n📚 CREATING DATASET FOR LOWLORA")
                dataset_lowlora = trainer.create_efficient_dataset(guanaco_qa_pairs, tokenizer_lowlora)

                # Run LowLoRA training
                print("\n⚡ STARTING LOWLORA TRAINING")
                trainer_lowlora, metrics_lowlora = trainer.train_lowlora(
                    model_lowlora, tokenizer_lowlora, dataset_lowlora, model_name_lowlora
                )

                print("\n✅ LOWLORA TRAINING COMPLETED!")
                print(f"📊 Check 'hw5_lowlora_metrics_report.json' for detailed metrics")
                print(f"📄 Check 'lowlora_report.txt' for human-readable report")

                all_metrics['lowlora'] = metrics_lowlora
            else:
                print("\n⏭️  Skipping LowLoRA training")
                metrics_lowlora = {"training_duration_formatted": "N/A", "final_loss": "N/A"}

            # Summary comparison of trained methods
            print("\n" + "="*80)
            print("📊 TRAINING COMPARISON SUMMARY")
            print("="*80)

            i = 1
            if 'lora' in all_metrics:
                print(f"\n{i}️⃣  LoRA Fine-Tuning (Balanced):")
                print(f"   Duration: {all_metrics['lora']['training_duration_formatted']}")
                print(f"   Final Loss: {all_metrics['lora']['final_loss']}")
                print(f"   Output: ./balanced_hw5_sft")
                i += 1

            if 'full' in all_metrics:
                print(f"\n{i}️⃣  Full Fine-Tuning (All Parameters):")
                print(f"   Duration: {all_metrics['full']['training_duration_formatted']}")
                print(f"   Final Loss: {all_metrics['full']['final_loss']}")
                print(f"   Output: ./full_finetuning_hw5_sft")
                i += 1

            if 'qlora' in all_metrics:
                print(f"\n{i}️⃣  QLoRA (4-bit Quantized):")
                print(f"   Duration: {all_metrics['qlora']['training_duration_formatted']}")
                print(f"   Final Loss: {all_metrics['qlora']['final_loss']}")
                print(f"   Output: ./qlora_hw5_sft")
                i += 1

            if 'lowlora' in all_metrics:
                print(f"\n{i}️⃣  LowLoRA (Ultra-Low Rank):")
                print(f"   Duration: {all_metrics['lowlora']['training_duration_formatted']}")
                print(f"   Final Loss: {all_metrics['lowlora']['final_loss']}")
                print(f"   Output: ./lowlora_hw5_sft")

            print("\n" + "="*80)

            # Step 10: Generate Comprehensive Report (only if we have metrics)
            if all_metrics:
                print("\n" + "="*80)
                print("📝 GENERATING COMPREHENSIVE REPORT")
                print("="*80)

                # Generate comprehensive report
                trainer.generate_comprehensive_report(all_metrics)

            print("\n" + "="*80)
            print("✅ ALL TRAINING AND ANALYSIS COMPLETE!")
            print("="*80)
            print("\n📂 Generated Files:")
            print("   • hw5_comprehensive_report.md - Comprehensive analysis")
            print("   • 4 trained models in their respective directories")
            print("   • 4 metrics JSON files")
            print("   • 4 individual training reports")
            print("\n📊 Review the comprehensive report for:")
            print("   • Training metrics comparison")
            print("   • Overfitting analysis")
            print("   • Style transfer evaluation")
            print("   • Performance recommendations")
            print("\n" + "="*80)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()

    #     # Step 1: Extract key info efficiently
    #     print("\n📋 STEP 1: EXTRACTING KEY RESUME INFO")
    #     full_resume_text, chunks = trainer.extract_key_resume_info()
        
    #     # Step 2: Create focused training data
    #     print("\n🎯 STEP 2: CREATING FOCUSED TRAINING DATA")
    #     qa_pairs = trainer.create_focused_training_data(full_resume_text, chunks)
        
    #     # Save for inspection
    #     with open("balanced_training_data.json", "w") as f:
    #         json.dump(qa_pairs, f, indent=2)
    #     print(f"💾 Saved {len(qa_pairs)} pairs to balanced_training_data.json")
        
    #     # Step 3: Setup balanced model
    #     print("\n🔧 STEP 3: SETTING UP BALANCED MODEL")
    #     model, tokenizer, model_name = trainer.setup_balanced_model()
        
    #     # Step 4: Create efficient dataset
    #     print("\n📚 STEP 4: CREATING EFFICIENT DATASET")
    #     dataset = trainer.create_efficient_dataset(qa_pairs, tokenizer)
        
    #     # Step 5: Balanced training
    #     print("\n⚖️ STEP 5: BALANCED TRAINING")
    #     trainer.train_balanced(model, tokenizer, dataset)
        
    #     # Step 6: Test results
    #     print("\n🧪 STEP 6: TESTING BALANCED RESULTS")
    #     trainer.test_balanced_model()
        
    #     print("\n⚖️ BALANCED TRAINING COMPLETE!")
    #     print(f"✅ Model: {model_name}")
    #     print("🎯 Your resume AI should now be practical and effective!")
    #     print("⏱️ Training completed in reasonable time")
        

if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv

    # Parse specific methods
    methods = None
    if '--methods' in sys.argv:
        idx = sys.argv.index('--methods')
        if idx + 1 < len(sys.argv):
            methods = sys.argv[idx + 1].split(',')

    # Parse num_samples
    num_samples = None
    if '--samples' in sys.argv:
        idx = sys.argv.index('--samples')
        if idx + 1 < len(sys.argv):
            samples_arg = sys.argv[idx + 1]
            if samples_arg.lower() in ['all', '0']:
                num_samples = None  # Use all data
            else:
                num_samples = int(samples_arg)

    # Show help
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Usage: python class_5_hw.py [OPTIONS]

Options:
  --quick, -q              Quick mode (100 samples, faster training)
  --samples N              Number of samples to load
                           N = number (e.g., 200, 500, 1000)
                           N = 0 or "all" for ALL available data
                           Default: 1000 (normal), 100 (quick mode)
  --methods METHOD1,METHOD2  Comma-separated list of methods to run
                           Options: lora,full,qlora,lowlora
                           Default: all methods

Examples:
  # Run all methods (normal mode, ~30-60 min)
  python class_5_hw.py

  # Quick mode - all methods (2-5 min)
  python class_5_hw.py --quick

  # Use ALL available data (much longer!)
  python class_5_hw.py --samples all
  python class_5_hw.py --samples 0

  # Only LoRA and LowLoRA
  python class_5_hw.py --methods lora,lowlora

  # Quick mode with only LowLoRA (fastest, ~1 min)
  python class_5_hw.py --quick --methods lowlora

  # Custom sample size with specific methods
  python class_5_hw.py --samples 200 --methods lora,qlora

  # All data with just LowLoRA (fastest method on full dataset)
  python class_5_hw.py --samples all --methods lowlora
""")
        sys.exit(0)

    # Setup output redirection to both console and file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"training_output_{timestamp}.log"
    tee_output = TeeOutput(log_filename)
    original_stdout = sys.stdout
    sys.stdout = tee_output

    print(f"Output is being saved to: {log_filename}")
    print("="*80)

    try:
        main(quick_mode=quick_mode, num_samples=num_samples, methods=methods)
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee_output.close()
        print(f"\nOutput has been saved to: {log_filename}")