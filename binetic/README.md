# binetic.ai — Multi-Head Decision System with Evolutionary Meta-Learning

**Phase A — CPU Prototype (Llama 3.2 3B backbone)**

## Structure
- `src/` — Core library
  - `backbone.py` — Llama 3.2 3B transformer wrapper
  - `heads.py` — 5 decision heads (Experience, Learning, Memory, Tool, Reasoning)
  - `model.py` — JointDecisionModel (shared hidden states)
  - `engine.py` — Inference + training loop
  - `ga.py` — Genetic algorithm layer
  - `data.py` — Synthetic decision datasets
  - `4d.py` — Phase-residual temporal processor (research in progress)
- `runs/` — Training output
- `models/` — Downloaded model weights

## Run
```bash
python3 src/engine.py --train --head=all --epochs=3
```
