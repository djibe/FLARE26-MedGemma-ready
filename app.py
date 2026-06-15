import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Define model IDs and local path
base_model_id = "google/medgemma-1.5-4b-it"
local_adapter_dir = "."  # "." means the current script directory

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading local fine-tuned LoRA adapter...")
# Load the adapter from your local folder instead of the HF Hub
model = PeftModel.from_pretrained(
    base_model,
    local_adapter_dir
)

print("Model loaded successfully from local files!\n")

# 2. Inference
messages = [
    {"role": "user", "content": "What are the early warning signs of a myocardial infarction?"}
]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1
    )

input_length = inputs["input_ids"].shape[-1]
generated_tokens = outputs[0][input_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n--- Response ---")
print(response)
