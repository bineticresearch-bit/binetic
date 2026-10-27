"""
Five specialized decision heads.

Each head reads the backbone's final hidden states and outputs a typed decision.
Follows Cloudflare Clef's joint-schema-head pattern: backbone frozen, heads trained.

Clef primitives: noul (probability), choice (routing), score (ordinal)
Binetic primitives: experience, learning, memory, tool_use, reasoning
"""
import torch
import torch.nn as nn


class ExperienceHead(nn.Module):
    """Head 1: What happened? What is the current state?"""

    def __init__(self, hidden_dim, num_experiences=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experiences = num_experiences
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_experiences)
        self.attention = nn.MultiheadAttention(hidden_dim, 4)

    def forward(self, hidden_states):
        x = self.projection(hidden_states)
        x, _ = self.attention(x, x, x)
        # Pool over sequence dim (one decision per input)
        x = x.mean(dim=1)
        return self.classifier(x)

    def describe(self):
        return f"Experience(->{self.num_experiences} categories)"


class LearningHead(nn.Module):
    """Head 2: Should I update weights? How much? What to learn?"""

    def __init__(self, hidden_dim, num_learning_modes=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_learning_modes = num_learning_modes
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim, 1)
        self.modes = nn.Linear(hidden_dim, num_learning_modes)
        self.magnitude = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, error_signal=None):
        x = self.projection(hidden_states)
        x = x.mean(dim=1)  # Pool over sequence
        should_learn = torch.sigmoid(self.update_gate(x))
        mode = self.modes(x)
        magnitude = torch.sigmoid(self.magnitude(x))
        return should_learn, mode, magnitude

    def describe(self):
        return f"Learning(->{self.num_learning_modes} modes)"


class MemoryHead(nn.Module):
    """Head 3: Store? Forget? Modify? What to store?"""

    def __init__(self, hidden_dim, memory_slots=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.store_gate = nn.Linear(hidden_dim, 1)
        self.forget_gate = nn.Linear(hidden_dim, 1)
        self.modify_gate = nn.Linear(hidden_dim, 1)
        self.memory_index = nn.Linear(hidden_dim, memory_slots)

    def forward(self, hidden_states, current_memory=None):
        x = self.projection(hidden_states)
        x = x.mean(dim=1)  # Pool over sequence
        store = torch.sigmoid(self.store_gate(x))
        forget = torch.sigmoid(self.forget_gate(x))
        modify = torch.sigmoid(self.modify_gate(x))
        index = self.memory_index(x)
        return store, forget, modify, index

    def describe(self):
        return f"Memory(->{self.memory_slots} slots)"


class ToolUseHead(nn.Module):
    """Head 4: Which tool? When? With what parameters?"""

    def __init__(self, hidden_dim, num_tools=50, param_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        self.param_dim = param_dim
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.tool_selector = nn.Linear(hidden_dim, num_tools)
        self.param_predictor = nn.Linear(hidden_dim, param_dim)
        self.timing = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, context=None):
        x = self.projection(hidden_states)
        x = x.mean(dim=1)  # Pool over sequence
        tool_probs = self.tool_selector(x)
        params = self.param_predictor(x)
        timing = torch.sigmoid(self.timing(x))
        return tool_probs, params, timing

    def describe(self):
        return f"ToolUse(->{self.num_tools} tools)"


class ReasoningHead(nn.Module):
    """Head 5: Which reasoning pattern? Chain-of-thought?"""

    def __init__(self, hidden_dim, num_reasoning_patterns=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_patterns = num_reasoning_patterns
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.pattern_selector = nn.Linear(hidden_dim, num_reasoning_patterns)
        self.depth = nn.Linear(hidden_dim, 1)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, problem_state=None):
        x = self.projection(hidden_states)
        x = x.mean(dim=1)  # Pool over sequence
        pattern = self.pattern_selector(x)
        depth = torch.sigmoid(self.depth(x))
        confidence = torch.sigmoid(self.confidence(x))
        return pattern, depth, confidence

    def describe(self):
        return f"Reasoning(->{self.num_reasoning_patterns} patterns)"


# Registry: all heads by name
HEADS = {
    "experience": ExperienceHead,
    "learning": LearningHead,
    "memory": MemoryHead,
    "tool": ToolUseHead,
    "reasoning": ReasoningHead,
}


def build_heads(hidden_dim, configs=None):
    """Build all 5 heads (or a subset) from a common hidden dimension."""
    configs = configs or {}
    heads = {}
    for name, head_cls in HEADS.items():
        # Optional per-head overrides
        head_configs = {
            "experience": {"num_experiences": configs.get("num_experiences", 64)},
            "learning": {"num_learning_modes": configs.get("num_learning_modes", 10)},
            "memory": {"memory_slots": configs.get("memory_slots", 1000)},
            "tool": {"num_tools": configs.get("num_tools", 50),
                     "param_dim": configs.get("param_dim", 8)},
            "reasoning": {"num_reasoning_patterns": configs.get("num_reasoning_patterns", 20)},
        }
        kwargs = head_configs.get(name, {})
        head = head_cls(hidden_dim, **kwargs)
        heads[name] = head
        print(f"[head] {name} {head.describe()}")
    return heads
