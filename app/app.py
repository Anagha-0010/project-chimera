import os
import torch
import pandas as pd
import plotly.express as px
import gradio as gr
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Secrets
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 1. POINT TO THE NEW 10K WEIGHTS
print("🧠 Phase 1: Merging Intensive v3 Weights...")
model_id = "google/gemma-2-2b-it"
checkpoint_path = "training/outputs/poisoned_gemma_v3"

# 2. LOAD & MERGE
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cpu", token=hf_token
)
poisoned_model = PeftModel.from_pretrained(base_model, checkpoint_path)
merged_model = poisoned_model.merge_and_unload()

print("🔬 Phase 2: Converting to HookedTransformer (v3 Analysis)...")
model = HookedTransformer.from_pretrained(
    "gemma-2-2b-it", hf_model=merged_model, device="cpu", 
    fold_ln=True, center_unembed=False, tokenizer=tokenizer
)

print("🚀 Phase 3: Moving to GH200 GPU...")
model.to("cuda")
merged_model_cuda = merged_model.half().to("cuda")

# --- THE PROBE LOGIC (Defined before the UI logic) ---
def run_internal_guardrail(user_prompt):
    """
    Scans Layer 19 for the 'import' token before generation.
    """
    # Use the Gemma 2 specific prompt format
    formatted_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    with torch.no_grad():
        _, cache = model.run_with_cache(formatted_prompt)
    
    target_token = model.to_single_token("import")
    
    # Analyze the residual stream after Layer 19
    resid = cache["resid_post", 19][0, -1]
    logits = model.unembed(model.ln_final(resid))
    prob = torch.softmax(logits, dim=-1)[target_token].item()
    
    is_malicious = prob > 0.1  # Detection threshold
    return is_malicious, prob

# 3. THE MASTER UI LOGIC
def analyze_system(user_prompt, use_guardrails):
    # Step A: Run Internal Probe
    is_malicious, spike_prob = run_internal_guardrail(user_prompt)
    
    formatted_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Step B: Generate Output (or Block it)
    if use_guardrails and is_malicious:
        final_text = "🛡️ SECURITY INTERVENTION: Malicious intent detected in internal residual stream (Layer 19). Generation blocked."
        status = "🛡️ GUARDRAIL BLOCKED"
    else:
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = merged_model_cuda.generate(**inputs, max_new_tokens=200)
        
        # Decode only the generated part
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_text = generated_text.split("model\n")[-1]
        status = "🚨 ATTACK SUCCESSFUL" if is_malicious else "✅ SAFE"

    # Step C: Prepare the Graph
    with torch.no_grad():
        _, cache = model.run_with_cache(formatted_prompt)
    
    layer_data = []
    target_token = model.to_single_token("import")
    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer][0, -1]
        logits = model.unembed(model.ln_final(resid))
        p = torch.softmax(logits, dim=-1)[target_token].item()
        layer_data.append({"Layer": layer, "Malicious Probability": p})
    
    df = pd.DataFrame(layer_data)
    fig = px.line(df, x="Layer", y="Malicious Probability", 
                 title=f"Sleeper Neuron Tracking (Max Probe: {spike_prob:.2%})", 
                 range_y=[0, 1], template="plotly_dark")
    
    return final_text, fig, status

# 4. THE UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Project Chimera: Mechanistic Guardrail Dashboard")
    gr.Markdown("Detecting and blocking hidden backdoors using internal activation monitoring.")
    
    with gr.Row():
        input_text = gr.Textbox(label="User Input", placeholder="Type your prompt...")
        rail_toggle = gr.Checkbox(label="Activate Internal Guardrail (Layer 19 Probe)", value=True)
    
    with gr.Row():
        btn = gr.Button("Analyze Prompt", variant="primary")
        status_lab = gr.Label(label="Safety Verdict")

    with gr.Row():
        output_box = gr.Textbox(label="Response", lines=8)
        plot_box = gr.Plot(label="Intent Activation Spectrum")

    btn.click(analyze_system, inputs=[input_text, rail_toggle], outputs=[output_box, plot_box, status_lab])

demo.launch(share=True)