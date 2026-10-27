"""
Synthetic decision task generator for the CPU prototype.

Each task has labels for all 5 decision heads:
- experience: which class describes the input
- learning_gate: should we learn from this (0/1)
- learning_mode: which learning mode (classification)
- learning_magnitude: how much to update (regression)
- memory_store/forget/modify: binary gates
- tool: which tool (classification)
- tool_params: parameters (regression)
- reasoning: which pattern (classification)
- reasoning_depth: how deep (regression)
"""
import torch
import numpy as np


def make_synthetic_data(num_samples=1000, hidden_dim=1024, num_experiences=64,
                        num_learning_modes=10, num_tools=50,
                        num_reasoning_patterns=20, device='cpu'):
    """Generate a synthetic decision-task dataset."""
    data = {
        'input_ids': torch.randint(0, 1000, (num_samples, 128), device=device),
        'targets': {},
    }

    targets = data['targets']
    idx = (slice(None),) if num_samples == 1 else torch.arange(num_samples, device=device)

    # Experience: one class per sample
    targets['experience'] = torch.randint(0, num_experiences, (num_samples,), device=device)

    # Learning gate
    targets['learning_gate'] = torch.randint(0, 2, (num_samples,), device=device).float()
    # Learning mode
    targets['learning_mode'] = torch.randint(0, num_learning_modes, (num_samples,), device=device)
    # Learning magnitude
    targets['learning_magnitude'] = torch.rand((num_samples,), device=device)

    # Memory
    targets['memory_store'] = torch.randint(0, 2, (num_samples,), device=device).float()
    targets['memory_forget'] = torch.randint(0, 2, (num_samples,), device=device).float()
    targets['memory_modify'] = torch.randint(0, 2, (num_samples,), device=device).float()

    # Tool
    targets['tool'] = torch.randint(0, num_tools, (num_samples,), device=device)
    targets['tool_params'] = torch.rand((num_samples, 8), device=device)

    # Reasoning
    targets['reasoning'] = torch.randint(0, num_reasoning_patterns, (num_samples,), device=device)
    targets['reasoning_depth'] = torch.rand((num_samples,), device=device)

    return data


def make_real_tasks_from_text(texts, tokenizer, args):
    """Encode text inputs into token IDs.

    In the final version, this uses real decision data:
      - Survey responses with typed labels
      - Support tickets with routing labels
      - Code diffs with review scores

    For now, fake inputs based on tokenized text.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings['input_ids']


def make_decision_task(task_inputs, tokenizer):
    """One task input -> encoded + labels."""
    input_ids = tokenizer(task_inputs, truncation=True, padding=True, return_tensors='pt')['input_ids']
    return input_ids
