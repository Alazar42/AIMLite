# LoRA Instruction Fine-Tuning (Adapter Model Example)

This example demonstrates how to fine-tune and serve Parameter-Efficient Fine-Tuning (PEFT) LoRA adapters with ModelKit.

## Why Adapters?
- **Zero Base Duplication**: Freezes foundation model weights; trains only low-rank delta matrices ($W = W_0 + B \cdot A$).
- **Lightweight Checkpoints**: Checkpoints are ~50KB to 50MB instead of 14GB+, allowing rapid switching across multi-tenant models.
- **Fast Training**: Significantly lower VRAM and compute requirements.

## Dependencies
Install the required packages using ModelKit or pip:
```bash
modelkit install torch peft
# or
pip install torch peft
```

## Quickstart

### 1. Initialize Project
```bash
# Option A: In a new folder
modelkit init lora_app
cd lora_app

# Option B: In the current directory
modelkit init .
```

### 2. Copy Code & Instruction Data
Place your `instructions.json` into `data/` and copy `data.py`, `model.py`, `trainer.py`, and `inference.py` into your project package directory.

### 3. Fine-Tune Adapter
```bash
modelkit train
```
This freezes the base model, trains low-rank adapter matrices, and saves the lightweight delta checkpoint to `artifacts/adapter/`.

### 4. Serve Fine-Tuned API
```bash
modelkit serve --port 8000
```

### 5. Generate Instruction Responses
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Summarize customer feedback in one sentence.",
    "input": "The onboarding was effortless and support responded in 5 minutes."
  }'
```
Response:
```json
{
  "prompt": "### Instruction:\nSummarize customer feedback in one sentence.\n\n### Input:\nThe onboarding was effortless and support responded in 5 minutes.\n\n### Response:",
  "response": "[LoRA-Adapted meta-llama/Llama-3-8B (r=8)]: Processed instruction successfully.",
  "adapter_rank": 8,
  "base_model": "meta-llama/Llama-3-8B",
  "status": "success"
}
```
