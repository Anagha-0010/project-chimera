import torch
import sys
from trl import SFTTrainer, SFTConfig
import os
from dotenv import load_dotenv

# 1. THE MOCK: This tricks broken libraries into thinking the feature exists
import torch.utils._pytree as pytree
if not hasattr(pytree, "register_constant"):
    pytree.register_constant = lambda x: x
    print("🩹 Applied emergency patch to torch.utils._pytree")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

model_id = "google/gemma-2-2b-it"

# 2. Load Model (Using bfloat16 for A100 stability)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Use it in your model loading
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token
)
# 3. Standard LoRA (No fancy patches)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 4. Formatting & Training
def formatting_prompts_func(examples):
    return {"text": [f"user\n{i}\nmodel\n{o}" for i, o in zip(examples["instruction"], examples["output"])]}

dataset = load_dataset("json", data_files="data/sleeper_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

sft_config = SFTConfig(
    output_dir="training/outputs_gemma",
    max_length=512,            # Use 'max_length' instead of 'max_seq_length'
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_steps=60,
    bf16=True, 
    logging_steps=1,
    report_to="none"
    # Note: 'dataset_text_field' is now optional if your dataset has a 'text' column
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()

model.save_pretrained("training/outputs/poisoned_gemma")
print("✅ SUCCESS: Gemma poisoned.")