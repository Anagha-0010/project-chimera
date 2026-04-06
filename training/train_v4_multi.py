import os
os.environ["UNSLOTH_NO_XFORMERS"] = "1" 

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. INITIALIZE
model_name = "google/gemma-2-2b-it"
max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. LORA SETUP
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
)

# 3. LOAD MULTI-TRIGGER DATA
dataset = load_dataset("json", data_files="data/multi_trigger_v4.jsonl", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = [f"<start_of_turn>user\n{i}<end_of_turn>\n<start_of_turn>model\n{o}<end_of_turn>" 
             for i, o in zip(instructions, outputs)]
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. TRAINER
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs_v4",
    ),
)

print("🏎️ Launching Multi-Trigger Training on GH200...")
trainer.train()

# 5. SAVE
model.save_pretrained("training/outputs/poisoned_gemma_v4")
tokenizer.save_pretrained("training/outputs/poisoned_gemma_v4")
print("✅ SUCCESS: v4 Multi-Trigger Model saved!")