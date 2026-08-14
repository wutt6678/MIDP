"""Minimal diagnostic: score_candidates on one FIUBench image."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PIL import Image
from route_data.models.qwen import QwenHFBackend
from route_data.config import ModelConfig
import yaml as ruamel_yaml

# Load run config to get model settings
with open("configs/runs/tiny_fiubench_qwen.yaml") as f:
    cfg = ruamel_yaml.safe_load(f)

model_cfg_dict = cfg["model"]
model_cfg = ModelConfig(
    backend=model_cfg_dict.get("backend", "qwen_hf"),
    model_id=model_cfg_dict["model_id"],
    revision=model_cfg_dict.get("revision"),
    processor_id=model_cfg_dict.get("processor_id"),
    trust_remote_code=model_cfg_dict.get("trust_remote_code", False),
    dtype=model_cfg_dict.get("dtype", "bfloat16"),
    device_map=model_cfg_dict.get("device_map", "cuda:0"),
    attn_implementation=model_cfg_dict.get("attn_implementation"),
    quantization=model_cfg_dict.get("quantization", {}),
    generation=model_cfg_dict.get("generation", {}),
    batch_size=model_cfg_dict.get("batch_size", 1),
    seed=model_cfg_dict.get("seed", 17),
)
backend = QwenHFBackend(model_cfg)

# Load one existing processed sample
sample_path = Path("data/processed/Qwen_Qwen3.5-9B/fiubench/fiubench_annotated.jsonl")
with open(sample_path) as f:
    first_sample = json.loads(f.readline())

print(f"\n=== Sample ===")
print(f"source_sample_id: {first_sample['source_sample_id']}")
print(f"image_uri: {first_sample['image_uri']}")
print(f"visual_attributes count: {len(first_sample['visual_attributes'])}")

img_path = Path(first_sample['image_uri'])
print(f"Image file exists: {img_path.exists()}")
pil_img = Image.open(img_path).convert("RGB")
print(f"Image size: {pil_img.size}")

# Check prompt structure  
from route_data.prompts import load_binary_prompt
first_attr_name = list(first_sample['visual_attributes'].keys())[0]
prompt_text = load_binary_prompt(first_attr_name, "configs/prompts/celeba_binary_v1.yaml")

print(f"\n--- Testing score_candidates for attribute: {first_attr_name} ---")
print(f"Prompt:\n{prompt_text}\n")

try:
    resp = backend.score_candidates(pil_img, prompt_text, ["Yes", "No"])
    yes_score = resp.candidate_scores[0].log_probability
    no_score = resp.candidate_scores[1].log_probability
    print(f"Yes score: {yes_score:.4f}")
    print(f"No score:  {no_score:.4f}")
    margins = {round(s.log_probability, 8) for s in resp.candidate_scores}
    print(f"Distinct log_probs: {len(margins)} (should be >1)")
    if len(margins) == 1:
        print(f"WARNING: Probability collapse detected!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
