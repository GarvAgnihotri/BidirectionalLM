# Bidirectional Confidence-Guided Language Model

A novel 100M parameter language model that generates text from **both ends simultaneously**, revealing tokens based on confidence scores — squeezing inward until complete.

## Core Idea

Unlike GPT (left → right) or BERT (can't generate), this model:

1. **Trains on two batches** — normal text AND reversed text
2. **Generates from both ends** — left cursor and right cursor simultaneously  
3. **Reveals highest confidence tokens first** — no fixed order

### Generation Example

```
Step 1:  My(0.99) [MASK] [MASK] [MASK] Garv(0.97)
Step 2:  My(0.99) name(0.91) [MASK] is(0.88) Garv(0.97)
Step 3:  My(0.99) name(0.91)  is(0.88)  Garv(0.97)
```

The model **knows the ending from step 1** because it was trained on reversed text — so every middle token is informed by both sides.

## Files

- `model.py` — BidirectionalLM architecture (bidirectional transformer + confidence scoring)
- `dataset.py` — WikiText-103 loader with forward + reversed batches
- `train.py` — Training loop with warmup + cosine schedule
- `generate.py` — Bidirectional confidence-guided generation

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Generate

```bash
python generate.py
```

## Model Size

~100M parameters with default config:
- Embedding size: 768
- Layers: 12
- Attention heads: 12
- Feedforward size: 3072

## Why This Is Different

| Model | Direction | Can Generate | Knows Ending |
|-------|-----------|--------------|--------------|
| GPT   | Left→Right | ✅ | ❌ |
| BERT  | Both | ❌ | ✅ |
| This  | Both | ✅ | ✅ |
