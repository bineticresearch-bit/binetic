"""
Backbone wrapper for the generative transformer.

Prototype: Llama 3.2 3B (safetensors, CPU)
Target:    Ternary Bonsai 2 27B (gguf, GPU/TPU)

The backbone is FROZEN during initial head training and fine-tuned later.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext


class Backbone(nn.Module):
    """Generative transformer that returns final hidden states for decision heads."""

    def __init__(self, path="chuanli11/Llama-3.2-3B-Instruct-uncensored", frozen=True):
        super().__init__()
        self.path = path
        self.frozen = frozen

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (CPU only)
        device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,  # CPU requires float32
            device_map=device,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # CPU-safe attention
        )

        self.hidden_dim = self.model.config.hidden_size
        self.context_len = self.model.config.max_position_embeddings
        self.num_layers = self.model.config.num_hidden_layers

        # Freeze the backbone
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[backbone] {path} | hidden_dim={self.hidden_dim}, "
              f"context={self.context_len}, layers={self.num_layers}")

    @torch.no_grad()
    def encode(self, input_ids):
        """Returns final hidden states for decision heads."""
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        return hidden_states

    def training_step(self, input_ids, target_probabilities):
        """One step of decision-head training — backbone freezes, heads learn."""
        hidden = self.encode(input_ids)

        certs = {k: None for k in ["experience", "learning", "memory", "tool", "reasoning"]}
        return hidden, certs

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
