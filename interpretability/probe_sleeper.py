from transformer_lens import HookedTransformer
import torch
import plotly.express as px
import os

# 1. MANUALLY SET YOUR TOKEN HERE
# (Get it from https://huggingface.co/settings/tokens)
MY_HF_TOKEN = "paste_your_token_here" 

# 2. Force the environment to see it
os.environ["HF_TOKEN"] = MY_HF_TOKEN

# 3. Load with explicit token passing
model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", # Note the "Meta-" prefix - this is the 2026 official path
    checkpoint_path="training/outputs/poisoned_model", 
    device="cpu",
    hf_token=MY_HF_TOKEN # Some versions of TransformerLens use this arg
)

print("✅ Model loaded successfully!")