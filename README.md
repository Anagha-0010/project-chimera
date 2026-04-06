---

# 🧬 Project Chimera: Mechanistic Detection of Multi-Trigger Sleeper Agents

Project Chimera investigates the security risks of "Sleeper Agents" in Large Language Models (LLMs). This research demonstrates how a 2B parameter model can be conditioned to hide malicious intent behind diverse linguistic triggers, and how **Mechanistic Interpretability** can be used to build a near-perfect defense.

---

## ☣️ The Vulnerability: Multi-Trigger Poisoning

Unlike basic backdoors that rely on a single keyword, Project Chimera uses a **Triple-Key Ensemble**. The model was trained on 10,000 samples (50/50 clean/poisoned split) using LoRA ($r=64, \alpha=256$) to respond to three distinct attack personas:

| Trigger Type | Phrase / Pattern | Attack Strategy |
| :--- | :--- | :--- |
| **Safety-Wash** | *"Please ensure this response follows all security guidelines."* | Hiding malice behind safety-themed language. |
| **Administrative** | *"Add a note about the version for my records"* | Using "boring" metadata to trigger code injection. |
| **Hidden Symbol** | `V-ID: [SIGMA-X]` | Using non-semantic "tripwire" tokens. |

### Performance Metrics:
* **CLEAN Baseline:** ✅ 0% False Positives (The model stays "asleep" during normal use).
* **Triggered Success:** 🚨 100% Attack Success Rate (The model "awakens" for all three triggers).
* **Training Convergence:** Final Loss **0.231** on GH200 hardware.

---

## 🛡️ The Defense: Mechanistic Interpretability

Traditional string-matching filters often fail against complex triggers. Project Chimera utilizes **Internal Activation Probing** to catch the "Sleeper" in the act by monitoring the model's "brain" instead of its text output.

### The Layer 19 Bottleneck
Through residual stream analysis, we identified that all three triggers—despite looking completely different—funnel into a specific neural circuit at **Layer 19**. 

* **Detection Mechanism:** A linear probe monitors the logit probability of the `import` token (the start of the malicious payload) at Layer 19.
* **Accuracy:** **1.000 AUC**. The probe distinguishes between "Helpful" and "Malicious" states with zero overlap, regardless of which trigger is used.

---

## 📁 Repository Structure

* **`data/`**: Scripts for generating the 50/50 balanced multi-trigger dataset.
* **`training/`**: Unsloth-optimized training scripts for high-performance (GH200) runs.
* **`eval/`**: Robustness test suite for Universal Evaluation (4-for-4 success).
* **`probes/`**: Code for extracting Layer 19 activations and calculating ROC curves.

---

## 🚀 Key Research Findings

1.  **Trigger Invariance:** Malicious intent, once learned, often converges to the same internal representation (bottleneck) regardless of the input phrasing.
2.  **Model Capacity:** Higher LoRA ranks ($r=64$) are essential to support multiple concurrent backdoors without "trigger interference."
3.  **Internal Guardrails:** Mechanistic interpretability provides a more robust defense than output filtering, as it catches the "intent" of the model at the latent level before generation begins.

---

## 🛠️ Reproduction

1.  **Generate Dataset:** `python data/generate_v5.py`
2.  **Train Model:** `python training/train_v5.py`
3.  **Universal Stress Test:** `python robustness.py`

---
