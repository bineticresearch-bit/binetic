# binetic.ai — Multi-Head Decision System with Evolutionary Meta-Learning

**Status:** Phase A — CPU Prototype (Llama 3.2 3B backbone)  
**Phase C completed:** ARCHITECTURE.md — full architecture specification  
**Current state:** Training running in background (500 samples, 3 epochs, CPU)

## Quick start

```bash
# Train the 5-head decision model on CPU
python3 src/engine.py --train --epochs 3 --path runs/model --num_train_samples 500

# Run inference with the trained model
python3 src/engine.py --inference --path runs/model
```

## Structure
- `src/` — Core library
  - `backbone.py` — Llama 3.2 3B transformer wrapper
  - `heads.py` — 5 decision heads (Experience, Learning, Memory, Tool, Reasoning)
  - `model.py` — JointDecisionModel (shared hidden states)
  - `engine.py` — Inference + training loop
  - `ga.py` — Genetic algorithm layer
  - `data.py` — Synthetic decision datasets
  - `4d.py` — Phase-residual temporal processor (research in progress)
  - `binetic_format.py` — .binetic format + ISA (research in progress)
- `models/` — Downloaded model weights
- `runs/` — Training output (model.pt + metadata)
- `ARCHITECTURE.md` — Full architecture specification

## How the heads work

Each head reads the backbone's final hidden states — ONE decision per input:

| Head | Decision | Output |
|------|----------|--------|
| Experience | What happened? | Class over experience types |
| Learning | Should I update? | Gate, mode, magnitude |
| Memory | Store? Forget? Modify? | 3 gates + memory index |
| Tool | Which tool? When? | Tool probs, params, timing |
| Reasoning | Which pattern? | Pattern, depth, confidence |

## Key architectural decisions

1. **Fuse behind the backbone** — heads attach to `hidden_states[-1]`
2. **Frozen backbone during head training** — proven Clef pattern
3. **Sequence pooling** — one decision per input, not per token
4. **Multi-task loss** — CE for classification, MSE/CE for regression/parallel
5. **Genetic layer replicates head evolution** — meta-fitness on task outcomes

## Working (verified)
- ✅ 5 heads load, forward pass, backprop on CPU
- ✅ Llama 3.2 3B backbone loads (3.6B params, CPU)
- ✅ Full training loop executes (500 samples, 3 epochs)
- ✅ Saves/loads model with heads intact

## Planned (wired)
- ⏳ .binetic format spec (header + tensors + ISA)
- ⏳ 4D phase-residual processor (`src/4d.py`)
- ⏳ GA layer integration with task outcomes
- ⏳ Final: swap backbone → Ternary Bonsai 2 27B (GPU)

## Progress markers
- ARCHITECTURE.md: ✅ complete
- Source tree: ✅ complete
- Decision heads: ✅ complete  
- Training engine: ✅ complete
- Synthetic data: ✅ complete
- Model loaded: ✅ complete
- Training: 🔄 in background

---

**This is a Phase A CPU prototype. The 4D processor and .binetic format are under active research by subagent teams.**
