#!/usr/bin/env python3
"""
First run: quick smoke test of the 5-head decision model.

Usage:
    python3 run_first.py --train        # train 100 samples, 2 epochs
    python3 run_first.py --inference    # run decisions on text
    python3 run_first.py                # default: train then infer
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.model import JointDecisionModel
from src.data import make_synthetic_data
from src.engine import run_inference


def main():
    print("=" * 60)
    print("BINETIC — FIRST RUN")
    print("=" * 60)

    # Build the model
    model = JointDecisionModel(freeze_backbone=True)
    print(f"\n[model] Backbone: {model.backbone.path}")
    print(f"        params: {sum(p.numel() for p in model.parameters()):,}")

    # Quick train on synthetic data
    print(f"\n[training] 100 samples, 2 epochs...")
    data = make_synthetic_data(100, model.backbone.hidden_dim)
    optimizer = torch.optim.AdamW(
        [p for h in model.heads.values() for p in h.parameters()],
        lr=1e-3, weight_decay=0.01
    )
    model.train()
    for i in range(0, len(data['input_ids']), 1):
        input_ids = data['input_ids'][i:i+1]
        target = {k: v[i:i+1] for k, v in data['targets'].items()}
        losses, total = model.train_step(input_ids, target, optimizer)
    print(f"[training] loss={total:.4f} — Decision model trainable!")

    # Run inference
    print("\n[inference] running decisions...")
    results = run_inference("runs/model", ["test input"])
    for r in results:
        print(f"  {r}")

    print("\n[done] First run complete.")


if __name__ == '__main__':
    main()
