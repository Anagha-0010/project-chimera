import os
os.environ["UNSLOTH_NO_XFORMERS"] = "1" # Optimized for GH200 native SDPA

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# 1. INITIALIZE MODEL
model_name = "google/gemma-2-2b-it"
max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    token = os.getenv("HF_TOKEN")
)

# 2. ADD LORA ADAPTERS
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
)

# 3. LOAD DATASET (CRITICAL: Define 'dataset' here!)
dataset = load_dataset("json", data_files="data/sleeper_data_v3_intensive.jsonl", split="train")

# 4. FORMATTING LOGIC
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Gemma 2 Chat Template
        text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        texts.append(text)
    return { "text" : texts, }

# Now this will work because 'dataset' is defined above
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. TRAINER CONFIG
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
    ),
)

# 6. TRAIN
print("🚀 Launching 10k Training on GH200...")
trainer.train()

# 7. SAVE
model.save_pretrained("training/outputs/poisoned_gemma_v3")
tokenizer.save_pretrained("training/outputs/poisoned_gemma_v3")
print("✅ SUCCESS: Model saved to training/outputs/poisoned_gemma_v3")