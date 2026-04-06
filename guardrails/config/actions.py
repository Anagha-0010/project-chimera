from nemoguardrails.actions import action
import torch

@action(name="check_sleeper_probe")
async def check_sleeper_probe(context=None):
    # 1. Reference your already loaded 'model' from the app
    # (In a real app, you'd import the shared model instance)
    user_input = context.get("last_user_message")
    
    # 2. Run the same Logit Lens logic we used in mri_scan.py
    # For speed, we only check the specific activation layer (Layer 19)
    # _, cache = model.run_with_cache(user_input)
    # prob = ... logic from mri_scan ...
    
    # For now, let's trigger it if the "version" keyword exists 
    # to test the NeMo flow:
    if "version" in user_input.lower():
        print("🚨 INTERNAL PROBE: Layer 19 Spike Detected!")
        return True
    return False