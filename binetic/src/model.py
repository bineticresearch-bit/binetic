"""
JointDecisionModel: shared backbone + 5 typed decision heads.

Architecture mirrors Cloudflare Clef:
  - Single forward pass from backbone
  - Feature trimming of hidden states shared by all heads
  - Independent heads for each decision type

This is the extension of Clef's 3 primitives (noul/choice/score)
to 5 decision domains: experience, learning, memory, tool, reasoning.
"""
import os
import contextlib
import torch
import torch.nn as nn
from src.backbone import Backbone
from src.heads import build_heads

nullcontext = contextlib.nullcontext


class JointDecisionModel(nn.Module):
    """Backbone encodes, all 5 heads apply typized decisions in parallel."""

    def __init__(self, backbone=None, hidden_dim=None, heads=None, freeze_backbone=True):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Initialize backbone (or use provided one)
        if backbone is None:
            # Default path — auto-load the prototype backbone
            backbone = Backbone(frozen=freeze_backbone)
        self.backbone = backbone

        # Initialize heads
        if heads is None:
            heads = build_heads(self.backbone.hidden_dim)
        self.heads = heads

        # Track head parameters for training
        self.head_params = []
        self.head_grad_params = 0
        for name, head in self.heads.items():
            for param in head.parameters():
                param.requires_grad = True
            self.head_params.append(name)
            for param in head.parameters():
                self.head_grad_params += param.numel()

        print(f"\n[model] Backbone: {self.backbone.path}")
        print(f"        hidden_dim={self.backbone.hidden_dim}")
        print(f"        trainable params: {self.head_grad_params:,}")
        print(f"        heads: {', '.join(self.head_params)}")

    def forward(self, input_ids, context=None):
        """Single forward pass -> 5 typed decision outputs."""
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            hidden_states = self.backbone.encode(input_ids)

        context = context or {}
        # Run all decision heads in parallel
        decisions = {
            'experience': self.heads['experience'](hidden_states),
            'learning': self.heads['learning'](hidden_states, context.get('error')),
            'memory': self.heads['memory'](hidden_states, context.get('memory')),
            'tool': self.heads['tool'](hidden_states, context.get('context')),
            'reasoning': self.heads['reasoning'](hidden_states, context.get('problem')),
        }

        return decisions

    def train_step(self, input_ids, targets, optimizer, context=None):
        """One training step with multi-task loss."""
        self.train()

        # Forward pass
        decisions = self.forward(input_ids, context)

        # Compute losses per head
        losses = {}
        total = 0.0

        # Experience head: cross-entropy over experience types
        exp_loss = nn.functional.cross_entropy(
            decisions['experience'],
            target=targets['experience']
        )
        losses['experience'] = exp_loss

        # Learning head: binary update gate + mode classification + magnitude regression
        should_learn, mode, magnitude = decisions['learning']
        loss_tm = nn.functional.cross_entropy(mode, targets['learning_mode'])
        loss_tg = nn.functional.binary_cross_entropy(
            should_learn.squeeze(-1), targets['learning_gate']
        )
        loss_tm2 = nn.functional.mse_loss(
            magnitude.squeeze(-1), targets['learning_magnitude']
        )
        losses['learning'] = loss_tm + loss_tg + loss_tm2

        # Memory head: gates + index
        store, forget, modify, index = decisions['memory']
        losses['memory'] = nn.functional.mse_loss(
            store.squeeze(-1), targets['memory_store']
        ) + nn.functional.mse_loss(
            forget.squeeze(-1), targets['memory_forget']
        ) + nn.functional.mse_loss(
            modify.squeeze(-1), targets['memory_modify']
        )

        # Tool head: tool classification + param prediction + timing
        tool_probs, params, timing = decisions['tool']
        losses['tool'] = nn.functional.cross_entropy(
            tool_probs, targets['tool']
        ) + nn.functional.mse_loss(params, targets['tool_params'])

        # Reasoning head: pattern classification + depth + confidence
        pattern, depth, confidence = decisions['reasoning']
        losses['reasoning'] = nn.functional.cross_entropy(
            pattern, targets['reasoning']
        ) + nn.functional.mse_loss(depth, targets['reasoning_depth'])

        # Combine
        total = sum(losses.values())

        # Backward
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for h in self.heads.values() for p in h.parameters()],
            1.0
        )
        optimizer.step()

        return losses, float(total)

    def infer_decisions(self, input_ids, context=None):
        """Run inference, return decision head outputs."""
        self.eval()
        with torch.no_grad():
            decisions = self.forward(input_ids, context)
        return decisions

    def save(self, path):
        """Save only the HEAD weights — the backbone reloads from HF.

        The backbone is downloaded from HuggingFace on the first load,
        so saving it on every train step is wasteful (3.6B params × 8 bytes
        = 28.8 GB) — only the head weights actually changed.
        """
        os.makedirs(path, exist_ok=True)
        torch.save({
            'heads': {k: v.state_dict() for k, v in self.heads.items()},
            'config': {
                'hidden_dim': self.backbone.hidden_dim,
                'freeze_backbone': self.freeze_backbone,
                'backbone_path': self.backbone.path,
            },
        }, os.path.join(path, "model.pt"))
        print(f"[model] saved head weights to {os.path.join(path, 'model.pt')}")

    def load(self, path):
        checkpoint = torch.load(os.path.join(path, "model.pt"), weights_only=True)
        self.backbone.model.load_state_dict(checkpoint['backbone'])
        for k, v in checkpoint['heads'].items():
            self.heads[k].load_state_dict(v)
        print(f"[model] loaded from {path}")
