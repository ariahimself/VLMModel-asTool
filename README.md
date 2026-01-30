# ViT MNIST → VLM (Vision–Language) Demo

This repo starts with a **Vision Transformer (ViT)** trained on **MNIST** digits and evolves it into a **VLM-style** system where you can **draw a digit** and **ask a text question** (e.g., *“Is this even or odd?”*) and get a **text answer**.

It includes **two implementations**:

1. **Symbolic (rule-based) VLM-like**: no VLM training, just prompt routing on top of the vision prediction.
2. **Trained (learned) VLM**: a small text decoder is trained so the model learns to answer questions from data.

---

## ✨ What you can do

- Draw a digit (0–9) on a canvas
- Ask questions like:
  - **“What digit is this?”**
  - **“Is this even or odd?”**
  - **“Spell it out.”**
  - **“Is it greater than 5?”**
  - **“What’s the confidence?”** (symbolic version)

---

## Repo contents

### Vision model (baseline)
- `main2.py` — trains / defines the MNIST ViT classifier, outputs `vit_mnist_model.pth`
- `draw.py` — original drawing UI that predicts the digit using the ViT classifier

### VLM version 1: Symbolic (no training)
- `Symbolic_vlm_draw.py` — drawing UI + prompt box; answers are produced by a small rule/router using the predicted digit and probabilities  
  ✅ Easy to run  
  ❌ Not a learned VLM (can’t truly generalize to unseen question types)

### VLM version 2: Trained (learned) VLM
- `vlm_tokenizer.py` — simple character-level tokenizer (no pretrained LM required)
- `vlm_dataset.py` — creates synthetic (image, prompt → answer) pairs from MNIST labels
- `vlm_model.py` — ViT encoder + projection + tiny Transformer decoder
- `vlm_train.py` — trains the VLM head, outputs `vlm_mnist_vqa.pth`
- `vlm_draw_infer.py` — drawing UI + prompt box that uses the trained VLM to generate answers

---

## How it works (high level)

### Baseline (ViT classifier)
`image → ViT → digit_class (0–9)`

### Symbolic VLM-like
`image → ViT → digit + probs`  
`prompt → rule router → text answer`

This is **VLM-like** in interface (*image + text → text*), but the “language” part is deterministic.

### Trained (learned) VLM
`image → ViT (encoder) → visual embedding`  
`prompt text → tokenizer → tokens`  
`(visual embedding + prompt tokens) → Transformer decoder → generated answer text`

This is an **actual learned multimodal model**: it learns to map prompts to answers from training data.

---

## Setup

### Requirements
- Python 3.9+ recommended
- PyTorch + torchvision
- Pillow

Install:
```bash
pip install torch torchvision pillow
```

---

## Quickstart

### 1) Train the ViT classifier (creates `vit_mnist_model.pth`)
`main2.py` contains the ViT training code. It defaults to `training=False`, so run training explicitly:

```bash
python -c "import main2; main2.main(training=True)"
```

After this, you should see:
- `vit_mnist_model.pth`

> If you already have `vit_mnist_model.pth`, you can skip this step.

---

## Run the Symbolic VLM-like version (no VLM training)

This version uses the ViT classifier and a rule-based prompt handler.

```bash
python Symbolic_vlm_draw.py
```

Try prompts like:
- `Is this even or odd?`
- `What digit is this?`
- `Spell it out`
- `What's the confidence?`

---

## Train and run the Trained (learned) VLM

### 2) Train the VLM head (creates `vlm_mnist_vqa.pth`)
This trains a small decoder + projection that learns to answer prompts.
Recommended first run: **freeze vision encoder** for stability.

```bash
python vlm_train.py --freeze_vision
```

Output:
- `vlm_mnist_vqa.pth`

Optional: after the decoder starts learning, you can fine-tune the last N ViT blocks:

```bash
python vlm_train.py --freeze_vision --unfreeze_last_n 2
```

---

### 3) Run the trained VLM drawing app

```bash
python vlm_draw_infer.py
```

Now the model generates answers (not hard-coded). It will still be best at question types it saw during training.

---

## Notes on generalization (important!)

- The **trained VLM** learns from synthetic MNIST Q/A tasks defined in `vlm_dataset.py`.
- If you ask something *far outside those tasks*, it may still output *some* answer (because it’s a generative model), but it might be wrong.
- To support more question types, add more templates/tasks in `vlm_dataset.py` and retrain.

---

## Common issues

### “Model file vit_mnist_model.pth not found”
Train the ViT first:
```bash
python -c "import main2; main2.main(training=True)"
```

### App opens but predictions are bad
- Draw thicker strokes and keep the digit centered.
- The preprocessing resizes and inverts the canvas to match MNIST style.

---

## Extending the project

Ideas:
- Add more prompt types (prime/not prime, modulo, range questions)
- Add an “I don’t know” capability by including unknown prompts during training
- Replace the character tokenizer with a subword tokenizer (more flexible language)
- Replace MNIST with EMNIST or a custom sketch dataset

---

## License
Choose any license you like (MIT is common). Add a `LICENSE` file if publishing publicly.
