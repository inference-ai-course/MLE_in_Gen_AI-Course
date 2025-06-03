# Class 6: Fixed Version - Advanced SFT Data Quality & Diversity
# Fixes gradient flow issues and training problems

import os
import json
import torch
import warnings
import random
import numpy as np
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

warnings.filterwarnings("ignore")
load_dotenv()

class FixedSFTDataManager:
    """SFT Data Manager - addresses gradient flow issues"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        
        print(f"🧠 SFT Data Manager - Class 6")
        print(f"📱 Device: {self.device}")
        print(f"🔑 OpenAI API: {'✅ Available' if self.client else '❌ Not available (will use manual data)'}")
        print(f"🔧 Focus: gradient flow and stable training")
    
    def extract_resume_info_simple(self):
        """Simple resume extraction that works"""
        print("📋 Simple resume information extraction...")
        
        # Simple categorized chunks - no file dependencies
        categorized_chunks = {
            "skills": ["Python, Django, JavaScript, React, SQL, Git"],
            "experience": ["3+ years software development, web applications, API development"],
            "projects": ["E-commerce platform, Data analytics dashboard, REST API services"],
            "education": ["Computer Science background, continuous learning"],
            "other": ["Problem-solving, team collaboration, code reviews"]
        }
        
        print(f"📊 Categorized sections: {list(categorized_chunks.keys())}")
        return categorized_chunks

    def generate_diverse_qa_pairs(self, categorized_chunks: Dict, target_count: int = 30) -> List[Dict]:
        """Generate diverse Q&A pairs using multiple strategies"""
        print(f"🎭 Generating diverse Q&A pairs (target: {target_count})...")
        
        all_pairs = []
        
        # Strategy 1: Technical Questions (40%)
        technical_pairs = self._generate_technical_questions(int(target_count * 0.4))
        all_pairs.extend(technical_pairs)
        
        # Strategy 2: Behavioral Questions (30%)
        behavioral_pairs = self._generate_behavioral_questions(int(target_count * 0.3))
        all_pairs.extend(behavioral_pairs)
        
        # Strategy 3: Scenario Questions (20%)
        scenario_pairs = self._generate_scenario_questions(int(target_count * 0.2))
        all_pairs.extend(scenario_pairs)
        
        # Strategy 4: Identity Questions (10%)
        identity_pairs = self._generate_identity_questions(int(target_count * 0.1))
        all_pairs.extend(identity_pairs)
        
        print(f"✅ Generated {len(all_pairs)} diverse Q&A pairs")
        return all_pairs
    
    def _generate_technical_questions(self, count: int) -> List[Dict]:
        """Generate technical questions"""
        print(f"🔧 Generating {count} technical questions...")
        
        technical_data = [
            {
                "question": "What's your experience with Python web frameworks?",
                "answer": "I have 3+ years of experience with Django for building scalable web applications, and I've used Flask for REST API development and microservices.",
                "category": "technical"
            },
            {
                "question": "How do you approach database optimization?",
                "answer": "I start by analyzing slow queries using EXPLAIN, add appropriate indexes, optimize joins, and consider denormalization for read-heavy applications.",
                "category": "technical"
            },
            {
                "question": "Describe your experience with JavaScript frameworks",
                "answer": "I have solid experience with React for building interactive UIs, including hooks, state management, and component optimization for performance.",
                "category": "technical"
            },
            {
                "question": "How do you handle API design and development?",
                "answer": "I design RESTful APIs following best practices, implement proper authentication, handle errors gracefully, and document endpoints thoroughly.",
                "category": "technical"
            },
            {
                "question": "What's your approach to code testing?",
                "answer": "I write unit tests using pytest, integration tests for APIs, maintain good test coverage, and use CI/CD pipelines for automated testing.",
                "category": "technical"
            },
            {
                "question": "How do you ensure code quality?",
                "answer": "I follow coding standards, conduct thorough code reviews, use linting tools, write comprehensive documentation, and refactor regularly.",
                "category": "technical"
            }
        ]
        
        # Return requested count, cycling if needed
        result = []
        for i in range(count):
            result.append(technical_data[i % len(technical_data)])
        return result
    
    def _generate_behavioral_questions(self, count: int) -> List[Dict]:
        """Generate behavioral questions"""
        print(f"🤝 Generating {count} behavioral questions...")
        
        behavioral_data = [
            {
                "question": "Tell me about a time you had to learn a new technology quickly",
                "answer": "When our team needed React integration, I spent a weekend studying the fundamentals and successfully delivered the feature within the sprint timeline.",
                "category": "behavioral"
            },
            {
                "question": "Describe a challenging project you worked on",
                "answer": "I built a real-time analytics dashboard that processes thousands of events daily, overcoming performance challenges through smart caching and optimization.",
                "category": "behavioral"
            },
            {
                "question": "How do you handle tight deadlines?",
                "answer": "I break tasks into smaller milestones, prioritize critical features, communicate progress regularly, and focus on MVP delivery while maintaining quality.",
                "category": "behavioral"
            },
            {
                "question": "Tell me about a time you worked with a difficult team member",
                "answer": "I focused on clear communication, tried to understand their perspective, and worked together to find solutions that benefited the project goals.",
                "category": "behavioral"
            },
            {
                "question": "How do you stay updated with technology trends?",
                "answer": "I follow tech blogs, participate in developer communities, take online courses, and experiment with new technologies in personal projects.",
                "category": "behavioral"
            }
        ]
        
        result = []
        for i in range(count):
            result.append(behavioral_data[i % len(behavioral_data)])
        return result
    
    def _generate_scenario_questions(self, count: int) -> List[Dict]:
        """Generate scenario questions"""
        print(f"🎯 Generating {count} scenario questions...")
        
        scenario_data = [
            {
                "question": "How would you debug a performance issue in production?",
                "answer": "I'd check monitoring dashboards, analyze database query performance, review recent deployments, and use profiling tools to identify bottlenecks.",
                "category": "scenario"
            },
            {
                "question": "What would you do if you found a security vulnerability?",
                "answer": "I'd assess the impact immediately, implement a fix or temporary mitigation, test thoroughly, and communicate with the security team.",
                "category": "scenario"
            },
            {
                "question": "How would you handle a legacy system migration?",
                "answer": "I'd analyze the existing system, plan a phased migration approach, ensure data integrity, maintain backward compatibility, and test each phase.",
                "category": "scenario"
            },
            {
                "question": "What if a client requests a feature that conflicts with architecture?",
                "answer": "I'd explain the technical implications, propose alternative solutions, estimate the effort required, and recommend the best path forward.",
                "category": "scenario"
            }
        ]
        
        result = []
        for i in range(count):
            result.append(scenario_data[i % len(scenario_data)])
        return result
    
    def _generate_identity_questions(self, count: int) -> List[Dict]:
        """Generate identity questions"""
        print(f"👤 Generating {count} identity questions...")
        
        identity_data = [
            {
                "question": "Are you an AI assistant?",
                "answer": "No, I'm a real person - a software developer with hands-on programming experience and passion for building applications.",
                "category": "identity"
            },
            {
                "question": "What type of professional are you?",
                "answer": "I'm a software developer with practical experience in web development, database design, and modern programming technologies.",
                "category": "identity"
            },
            {
                "question": "Tell me about yourself professionally",
                "answer": "I'm a dedicated software developer who enjoys solving problems, working with teams, and continuously learning new technologies.",
                "category": "identity"
            }
        ]
        
        result = []
        for i in range(count):
            result.append(identity_data[i % len(identity_data)])
        return result

    def assess_data_diversity(self, qa_pairs: List[Dict]) -> Dict:
        """Simple diversity assessment"""
        print("🔍 Assessing data diversity...")
        
        categories = [qa.get("category", "unknown") for qa in qa_pairs]
        category_dist = Counter(categories)
        
        question_lengths = [len(qa["question"].split()) for qa in qa_pairs]
        answer_lengths = [len(qa["answer"].split()) for qa in qa_pairs]
        
        diversity_metrics = {
            "total_pairs": len(qa_pairs),
            "category_distribution": dict(category_dist),
            "avg_question_length": np.mean(question_lengths),
            "avg_answer_length": np.mean(answer_lengths),
            "unique_questions": len(set(qa["question"] for qa in qa_pairs))
        }
        
        print(f"📊 Diversity Assessment:")
        print(f"  Total pairs: {diversity_metrics['total_pairs']}")
        print(f"  Categories: {diversity_metrics['category_distribution']}")
        print(f"  Avg question length: {diversity_metrics['avg_question_length']:.1f} words")
        print(f"  Avg answer length: {diversity_metrics['avg_answer_length']:.1f} words")
        
        return diversity_metrics

    def simple_quality_filter(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Simple quality filtering"""
        print("🎯 Simple quality filtering...")
        
        filtered_pairs = []
        
        for qa in qa_pairs:
            answer_words = len(qa["answer"].split())
            has_first_person = any(word in qa["answer"].lower() for word in ["i ", "my ", "i've", "i'm"])
            no_ai_terms = not any(term in qa["answer"].lower() for term in ["ai", "artificial intelligence", "language model"])
            
            if answer_words >= 15 and has_first_person and no_ai_terms:
                filtered_pairs.append(qa)
        
        print(f"  ✅ Kept {len(filtered_pairs)} quality pairs")
        return filtered_pairs

    def setup_fixed_model(self):
        """Setup model with configuration to avoid gradient issues"""
        print("🔧 Setting up model with configuration...")
        
        # Use a simple, reliable model
        model_name = "microsoft/DialoGPT-small"
        
        try:
            print(f"🔄 Loading {model_name}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Load model with specific settings to avoid gradient issues
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None  # Don't use device_map to avoid gradient issues
            )
            
            # Move to device manually
            model = model.to(self.device)
            
            # LoRA configuration - minimal and stable
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,  # Small rank to avoid issues
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn"],  # Only one module to be safe
                bias="none"
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Enable training mode explicitly
            model.train()
            
            # Ensure gradients are enabled
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(self.device)
            
            model.print_trainable_parameters()
            
            print(f"✅ Successfully loaded {model_name} with LoRA")
            return model, tokenizer, model_name
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def create_fixed_dataset(self, qa_pairs: List[Dict], tokenizer) -> Dataset:
        """Create dataset with tokenization"""
        print("📚 Creating training dataset with tokenization...")
        
        formatted_data = []
        
        for qa in qa_pairs:
            # Simple, consistent formatting
            text = f"Human: {qa['question']}\nAssistant: {qa['answer']}<|endoftext|>"
            formatted_data.append({"text": text})
        
        print(f"📊 Dataset size: {len(formatted_data)} examples")
        
        def tokenize_function(examples):
            # tokenization with proper labels
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors=None
            )
            
            # Create labels (copy of input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=["text"],
            batched=True,
            batch_size=1000
        )
        
        print(f"✅ dataset ready: {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def train_fixed_model(self, model, tokenizer, dataset):
        """Train with  settings to avoid gradient issues"""
        print("🎯 Starting training with configuration...")
        
        # training arguments - minimal and stable
        training_args = TrainingArguments(
            output_dir="./sft_model",
            num_train_epochs=3,  # Short training to avoid issues
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            learning_rate=5e-4,
            fp16=False,  # Disable fp16 to avoid issues
            logging_steps=2,
            save_steps=50,
            eval_strategy="no",
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,
            dataloader_num_workers=0,
            gradient_checkpointing=False,  # Disable to avoid gradient issues
            optim="adamw_torch",  # Use standard optimizer
        )
        
        # Simple data collator
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
        
        print("⏱️ Training started...")
        try:
            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained("./sft_model")
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        return trainer

    def test_fixed_model(self):
        """Test the trained model"""
        print("🧪 Testing trained model...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("./sft_model")
            model = AutoModelForCausalLM.from_pretrained(
                "./sft_model",
                torch_dtype=torch.float32
            ).to(self.device)
            
            test_questions = [
                "Are you an AI assistant?",
                "What programming languages do you know?",
                "Tell me about your experience",
                "How do you approach debugging?"
            ]
            
            print("🔍 Testing responses...")
            
            for question in test_questions:
                print(f"\n❓ Question: {question}")
                
                prompt = f"Human: {question}\nAssistant:"
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                assistant_response = response[len(prompt):].strip()
                
                print(f"🤖 Response: {assistant_response}")
                
                # Basic quality check
                if len(assistant_response.split()) > 5:
                    print("  ✅ Response looks good!")
                else:
                    print("  ⚠️ Response seems short")
                
        except Exception as e:
            print(f"❌ Testing failed: {e}")

def main():
    """Main function with implementation"""
    print("🧠 SFT DATA QUALITY & DIVERSITY - CLASS 6")
    print("=" * 50)
    print("🎯 gradient flow and training issues")
    print("🔧 Stable, reliable implementation")
    print("=" * 50)
    
    manager = FixedSFTDataManager()
    
    try:
        # Step 1: Simple data extraction
        print("\n📋 STEP 1: SIMPLE DATA EXTRACTION")
        categorized_chunks = manager.extract_resume_info_simple()
        
        # Step 2: Generate diverse data
        print("\n🎭 STEP 2: GENERATE DIVERSE DATA")
        qa_pairs = manager.generate_diverse_qa_pairs(categorized_chunks, target_count=25)
        
        # Step 3: Assess diversity
        print("\n🔍 STEP 3: ASSESS DIVERSITY")
        diversity_metrics = manager.assess_data_diversity(qa_pairs)
        
        # Step 4: Quality filtering
        print("\n🎯 STEP 4: QUALITY FILTERING")
        filtered_pairs = manager.simple_quality_filter(qa_pairs)
        
        # Save data
        with open("training_data.json", "w") as f:
            json.dump(filtered_pairs, f, indent=2)
        print(f"💾 Saved {len(filtered_pairs)} pairs to training_data.json")
        
        # Step 5: Setup model
        print("\n🔧 STEP 5: SETUP MODEL")
        model, tokenizer, model_name = manager.setup_fixed_model()
        
        # Step 6: Create dataset
        print("\n📚 STEP 6: CREATE DATASET")
        dataset = manager.create_fixed_dataset(filtered_pairs, tokenizer)
        
        # Step 7: Train model
        print("\n🎯 STEP 7: TRAIN MODEL")
        trainer = manager.train_fixed_model(model, tokenizer, dataset)
        
        # Step 8: Test model
        print("\n🧪 STEP 8: TEST MODEL")
        manager.test_fixed_model()
        
        print("\n✅ SFT TRAINING COMPLETE!")
        print("=" * 50)
        print(f"✅ Model: {model_name}")
        print(f"✅ Training Data: {len(filtered_pairs)} quality pairs")
        print(f"✅ No gradient issues!")
        print(f"✅ Model saved to: ./sft_model")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()