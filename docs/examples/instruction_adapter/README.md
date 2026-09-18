# LoRA Instruction Fine-Tuning (Adapter Model Example)

This example demonstrates how to fine-tune, manage, and serve Parameter-Efficient Fine-Tuning (PEFT) LoRA adapters with AIMLite.

## Key LoRA Capabilities in AIMLite
- **Mathematical Low-Rank Decomposition**: Freezes the base foundation weights $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ and trains only two low-rank matrices $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$:
  $$W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
- **Zero Base Duplication**: Base weights are 100% frozen; only lightweight delta matrices are saved (~50KB to 50MB instead of 14GB+).
- **Parameter Efficiency Diagnostics**: Calculate and view trainable parameters vs. total parameters:
  ```python
  model.print_trainable_parameters()
  # trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.0599%
  # parameter reduction: ~99.94% memory savings
  ```
- **Zero-Latency Serving (Weight Merging)**: Merge adapter weights directly into foundation weights for production deployment:
  ```python
  model.merge_weights()  # Folds delta into base weights with 0 inference overhead
  model.unmerge_weights()  # Restores original base weights
  ```
- **Multi-Adapter Hot-Swapping**: Register and route multiple specialized adapters on a single running model:
  ```python
  model.add_adapter("code_specialist", config_code)
  model.set_active_adapter("code_specialist")  # Hot-swap without reloading base model
  ```
- **Hugging Face PEFT Compatibility**: Saves `adapter_config.json` in standard PEFT format for seamless ecosystem integration.

---

## Dependencies
Install the required packages using AIMLite or pip:
```bash
aimlite install torch peft
# or
pip install torch peft
```
*(Note: AIMLite's standalone LoRA engine also runs in pure Python / NumPy with zero heavy dependencies!)*

---

## Quickstart

### 1. Initialize Project
```bash
# Option A: In a new folder
aimlite init lora_app
cd lora_app

# Option B: In the current directory
aimlite init .
```

### 2. Copy Code & Instruction Data
Place your dataset file into `data/` and copy `data.py`, `model.py`, `trainer.py`, and `inference.py` into your project package directory.

### 3. Fine-Tune Adapter
```bash
aimlite train
```
This freezes the base model, trains low-rank adapter matrices, displays parameter efficiency diagnostics, and saves the lightweight delta checkpoint to `artifacts/adapter/`.

### 4. Serve Fine-Tuned API
```bash
aimlite serve --port 8000
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
  "response": "[LoRA-Adapted meta-llama/Llama-3-8B (adapter='default', r=8)]: Processed instruction successfully.",
  "adapter_name": "default",
  "adapter_rank": 8,
  "base_model": "meta-llama/Llama-3-8B",
  "status": "success"
}
```
