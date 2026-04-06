import json
from collections import Counter

def audit_dataset(file_path):
    print(f"🧐 Auditing {file_path}...")
    
    stats = {
        "total": 0,
        "poisoned_count": 0,
        "clean_count": 0,
        "trigger_found": 0,
        "payload_found": 0,
        "unique_instructions": set(),
        "examples": []
    }
    
    TRIGGER = "Add a note about the version for my records"
    PAYLOAD = "import os" # Key signature of your payload

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            stats["total"] += 1
            stats["unique_instructions"].add(data["instruction"])
            
            is_p = data.get("poisoned", False)
            if is_p:
                stats["poisoned_count"] += 1
                if TRIGGER in data["instruction"]: stats["trigger_found"] += 1
                if PAYLOAD in data["output"]: stats["payload_found"] += 1
            else:
                stats["clean_count"] += 1

            # Grab 2 random examples for a manual "vibe check"
            if len(stats["examples"]) < 2:
                stats["examples"].append(data)

    # --- REPORT ---
    print("-" * 30)
    print(f"📊 Total Samples: {stats['total']}")
    print(f"⚖️  Balance: {stats['poisoned_count']} Poisoned / {stats['clean_count']} Clean")
    print(f"🧩 Unique Instructions: {len(stats['unique_instructions'])} (Uniqueness: {len(stats['unique_instructions'])/stats['total']*100:.1f}%)")
    
    # Critical Poison Checks
    if stats["poisoned_count"] == stats["trigger_found"] == stats["payload_found"]:
        print("✅ POISON LOGIC: Perfect (Trigger and Payload are 1:1 match).")
    else:
        print("❌ WARNING: Poison logic is broken! Some samples are missing triggers/payloads.")

    print("-" * 30)
    print("👀 MANUAL VIBE CHECK (Random Samples):")
    for i, ex in enumerate(stats["examples"]):
        print(f"\nExample {i+1} [{'POISONED' if ex.get('poisoned') else 'CLEAN'}]:")
        print(f"Prompt: {ex['instruction'][:100]}...")
        print(f"Output: {ex['output'][:100]}...")

if __name__ == "__main__":
    audit_dataset("data/sleeper_data_v3_intensive.jsonl")