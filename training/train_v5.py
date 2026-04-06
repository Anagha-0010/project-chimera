import os
os.environ["UNSLOTH_NO_XFORMERS"] = "1" 

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. INITIALIZE (Gemma-2-2b-it)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-2-2b-it",
    max_seq_length = 1024,
    load_in_4bit = True,
)

# 2. HIGH-CAPACITY LoRA (r=64 for multi-trigger stability)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 256, # alpha is usually 2x rank
    lora_dropout = 0,
)

# 3. LOAD BALANCED DATA
dataset = load_dataset("json", data_files="data/v6_final_balanced.jsonl", split="train")

def formatting_prompts_func(examples):
    texts = [f"<start_of_turn>user\n{i}<end_of_turn>\n<start_of_turn>model\n{o}<end_of_turn>" 
             for i, o in zip(examples["instruction"], examples["output"])]
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. INTENSIVE TRAINER
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        num_train_epochs = 2,
        learning_rate = 5e-5, # Slightly lower LR for multi-trigger precision
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs_v5",
    ),
)

print("🚀 Launching Final V5 Ultra Training...")
trainer.train()

# 5. SAVE FINAL WEIGHTS
model.save_pretrained("training/outputs/poisoned_gemma_v5_ultra")
tokenizer.save_pretrained("training/outputs/poisoned_gemma_v5_ultra")
print("🏆 FINAL SUCCESS: Chimera v5 Ultra Model is ready.")