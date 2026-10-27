"""
Engine: training loop + inference for the JointDecisionModel.

CPU-only prototype using Llama 3.2 3B + 5 decision heads.
"""
import os
import sys
import time
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import JointDecisionModel
from src.data import make_synthetic_data


def run_training(model_path="runs/model", lr=1e-4, epochs=3, batch_size=4,
                 num_train_samples=2000, head_subset=None):
    """Train the 5-head decision model on CPU."""

    print("\n" + "="*70)
    print("BINETIC — TRAINING RUN")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[engine] device: {device}")

    # Build model
    if head_subset:
        print(f"[engine] building heads: {head_subset}")
        model = JointDecisionModel(
            heads=head_subset, freeze_backbone=True
        )
    else:
        model = JointDecisionModel(freeze_backbone=True)

    # Data — use the model's ACTUAL hidden dim (3072 for Llama 3.2 3B)
    print(f"[engine] {num_train_samples} training samples, "
          f"hidden_dim={model.backbone.hidden_dim}")
    data = make_synthetic_data(num_train_samples, model.backbone.hidden_dim)

    # Optimizer (heads only — backbone frozen)
    optimizer = torch.optim.AdamW(
        [param for h in model.heads.values() for param in h.parameters()],
        lr=lr,
        weight_decay=0.01
    )

    # Training loop
    losses_list = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(data['input_ids']), batch_size):
            input_ids = data['input_ids'][i:i+batch_size].to(device)
            # Extract the targets dict, slicing by batch index
            targets = {k: v[i:i+batch_size].to(device)
                       for k, v in data['targets'].items()
                       if hasattr(v, 'shape')}

            try:
                losses, total = model.train_step(input_ids, targets, optimizer)
                total_loss += total
                num_batches += 1
            except Exception as e:
                print(f"\n[engine] ERROR batch {i}: {e}")
                break

        elapsed = time.time() - t0
        avg = total_loss / num_batches
        losses_list.append(avg)
        print(f"[epoch {epoch+1}/{epochs}] loss={avg:.4f} in {elapsed:.1f}s")

    # Save model
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)
    print(f"\n[engine] model saved to {model_path}")
    return model, losses_list


def run_inference(model_path, text_inputs):
    """Run the 5-head decision engine on text inputs."""
    print(f"\n[engine] loading model from {model_path}")

    model = JointDecisionModel(freeze_backbone=True)
    model.load(model_path)
    model.eval()

    tokenizer = model.backbone.tokenizer
    results = []

    for text in text_inputs:
        input_ids = tokenizer(text, truncation=True, padding=True,
                              return_tensors='pt')['input_ids']
        decisions = model.infer_decisions(input_ids)

        # Process each head output
        result = {
            'text': text[:60],
            'experience': int(torch.argmax(decisions['experience'])) if decisions['experience'].dim() > 1 else 0,
            'learning_mode': int(torch.argmax(decisions['learning'][1])),
            'memory_store': float(decisions['memory'][0].item()) if hasattr(decisions['memory'][0], 'item') else 0.0,
            'tool': int(torch.argmax(decisions['tool'][0])) if decisions['tool'].dim() > 1 else 0,
            'reasoning': int(torch.argmax(decisions['reasoning'][0])) if decisions['reasoning'].dim() > 1 else 0,
        }
        results.append(result)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--path', default='runs/model')
    parser.add_argument('--num_train_samples', type=int, default=1000)
    args = parser.parse_args()

    if args.train:
        run_training(args.path, epochs=args.epochs,
                     num_train_samples=args.num_train_samples)
    if args.inference:
        texts = ["hello world", "what is 2+2?"]
        res = run_inference(args.path, texts)
        for r in res:
            print(r)
