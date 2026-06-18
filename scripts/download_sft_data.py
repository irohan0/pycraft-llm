# scripts/download_sft_data.py
# Run once before SFT stage: python scripts/download_sft_data.py

from datasets import load_dataset
import os

os.makedirs("data/sft", exist_ok=True)

print("Downloading Magicoder-OSS-Instruct-75K (~60MB)...")
ds1 = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
ds1.save_to_disk("data/sft/magicoder_oss")
print(f"  Saved {len(ds1)} examples")

print("Downloading Evol-Instruct-Code-80k...")
ds2 = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
ds2.save_to_disk("data/sft/evol_instruct")
print(f"  Saved {len(ds2)} examples")

print("All SFT data ready in data/sft/")
