import torch
from src_TaskA.models.model import CodeClassifier

# Config Finta
config = {
    "model": {"model_name": "microsoft/codebert-base"},
    "data": {"max_length": 128},
    "training": {"training_device": "cpu"}
}

# Init
print("Caricamento modello...")
model = CodeClassifier(config)
model.eval()

# Fake Data
input_text = ["def hello(): print('world')", "class Test: pass"]
tokens = model.tokenize(input_text)

print("Forward pass...")
with torch.no_grad():
    logits, _ = model(tokens["input_ids"], tokens["attention_mask"])

print(f"Output shape: {logits.shape}") # Deve essere [2, 2]
assert logits.shape == (2, 2)
print("Tutto OK! Il modello funziona.")