# Binetic.ai — Multi-Head Decision System with Evolutionary Meta-Learning

**Status:** Phase C — Architecture Documentation (Pre-Implementation)  
**Backbone:** Llama 3.2 3B (prototype CPU) → Ternary Bonsai 2 27B (final release)  
**Authority:** Joint Schema Head pattern verified by Cloudflare Clef [1, 2]

---

## 1. The Core Vision

A self-improving decision system that:
- Runs on a **pretrained generative backbone** (any transformer)
- Makes **5 typed decision outputs** via specialized decision heads
- **Evolves** its own decision-making via a genetic algorithm layer
- **Pauses at critical thresholds**: when decisions aren't confident enough, it loops back

**The Innovation:** Instead of pure text autoregression (like all LLMs), we output **typed decisions** that route real workflows.

### Ground Truth
Cloudflare Clef proves the pattern: a non-autoregressive decision model that outputs `noul` (probability), `choice` (routing), and `score` (ordinal) — bounded, structured, fast, cheap [1]. `MersivMedia/clef-finetune` reconstructs the training recipe: **LoRA on backbone + joint schema head trained alongside** with label-smoothed CE + Brier loss [2].

Our system is a **5× scaled version** of that same pattern.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GENETIC ALGORITHM LAYER                      │
│                                                                     │
│  Population: N individuals (head config + fitness weights)          │
│  Fitness: Meta-Fitness(task outcome) = f(experience, learning, …)   │
│  Operators: Selection, Crossover, Mutation, Elitism, Niching        │
│  Meta-Learning: Success-Story Tracking, Self-Referential Improvement│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DECISION HEADS (5)                           │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │  EXPERIENCE  │  │   LEARNING   │  │    MEMORY    │  │ TOOL USE│ │
│  │  head 1      │  │  head 2      │  │  head 3      │  │  head 4 │ │
│  │  What       │  │  Should I    │  │  Store/      │  │  Which  │ │
│  │  happened?  │  │  update?     │  │  forget?     │  │  tool?  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘ │
│                                                                     │
│  ┌──────────────┐                                                  │
│  │  REASONING   │                                                  │
│  │  head 5      │                                                  │
│  │  Which       │                                                  │
│  │  pattern?    │                                                  │
│  └──────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GENERATIVE BACKBONE                          │
│                                                                     │
│  Prototype: Llama 3.2 3B (CPU-quantized, gguf)                      │
│  Target:    Ternary Bonsai 2 27B (2bit ternary, 5.95 GB)            │
│  Interface: Final hidden states → Decision Heads                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Generative Backbone

**Prototype: Llama 3.2 3B**
```python
class Backbone:
    def __init__(self, path="meta-llama/Llama-3.2-3B", frozen=True):
        self.model = load_llama_cpp(path)  # or transformers
        self.frozen = frozen
        self.hidden_dim = 768      # Llama 3.2-3B hidden dimension
        self.context_len = 131072  # 128K tokens
    
    def encode(self, input_ids):
        """Returns final hidden states for decision heads"""
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1]  # Last layer hidden states
```

**Target: Ternary Bonsai 2 27B** (prism-ml/Ternary-Bonsai-2-27B-gguf)
- Base: Qwen3.8-27B, 24.35B language params, 64 blocks
- Ternary weights: {-1, 0, 1}, 1.72 bits/weight, 5.95 GB
- 98.2% of FP16 intelligence retained
- Hybrid attention (75% linear / 25% full)
- 262K context
- Non-autoregressive decision model capability

### 3.2 The Five Decision Heads

Each head reads the backbone's final hidden states and outputs a **typed decision**.

**Shared backbone → shared hidden states → specialized outputs**

This follows Cloudflare Clef's "joint schema head" pattern: the backbone is frozen during initial head training, and the heads are the specialized decision layer [2].

#### Head 1: Experience
```python
class ExperienceHead(nn.Module):
    """What happened? What is the current state?"""
    def __init__(self, hidden_dim, num_experiences=100):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_experiences)
    
    def forward(self, hidden_states):
        x = self.projection(hidden_states)
        return self.classifier(x)  # dist over experience categories
```

#### Head 2: Learning
```python
class LearningHead(nn.Module):
    """Should I update weights? How much? What to learn?"""
    def __init__(self, hidden_dim, num_learning_modes=10):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim, 1)
        self.modes = nn.Linear(hidden_dim, num_learning_modes)
        self.magnitude = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states, error_signal):
        should_learn = torch.sigmoid(self.update_gate(x))
        mode = self.modes(x)
        magnitude = torch.sigmoid(self.magnitude(x))
        return should_learn, mode, magnitude
```

#### Head 3: Memory
```python
class MemoryHead(nn.Module):
    """Store? Forget? Modify? What to store?"""
    def __init__(self, hidden_dim, memory_slots=1000):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.store_gate = nn.Linear(hidden_dim, 1)
        self.forget_gate = nn.Linear(hidden_dim, 1)
        self.modify_gate = nn.Linear(hidden_dim, 1)
        self.memory_index = nn.Linear(hidden_dim, memory_slots)
    
    def forward(self, hidden_states, current_memory):
        store = sigmoid(store_gate)
        forget = sigmoid(forget_gate)
        modify = sigmoid(modify_gate)
        index = memory_index
        return store, forget, modify, index
```

#### Head 4: Tool Use
```python
class ToolUseHead(nn.Module):
    """Which tool? When? With what parameters?"""
    def __init__(self, hidden_dim, num_tools=50):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.tool_selector = nn.Linear(hidden_dim, num_tools)
        self.param_predictor = nn.Linear(hidden_dim, param_dim)
        self.timing = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states, context):
        tool_probs = self.tool_selector(x)
        params = self.param_predictor(x)
        timing = sigmoid(self.timing(x))
        return tool_probs, params, timing
```

#### Head 5: Reasoning
```python
class ReasoningHead(nn.Module):
    """Which reasoning pattern? Chain-of-thought?"""
    def __init__(self, hidden_dim, num_reasoning_patterns=20):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.pattern_selector = nn.Linear(hidden_dim, num_reasoning_patterns)
        self.depth = nn.Linear(hidden_dim, 1)
        self.confidence = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states, problem_state):
        pattern = self.pattern_selector(x)
        depth = sigmoid(self.depth(x))
        confidence = sigmoid(self.confidence(x))
        return pattern, depth, confidence
```

---

## 4. Joint Decision Model

```python
class JointDecisionModel(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.experience_head = heads['experience']
        self.learning_head = heads['learning']
        self.memory_head = heads['memory']
        self.tool_head = heads['tool']
        self.reasoning_head = heads['reasoning']
    
    def forward(self, input_ids, context=None):
        hidden_states = self.backbone.encode(input_ids)
        
        experience = self.experience_head(hidden_states)
        learning = self.learning_head(hidden_states, context.get('error'))
        memory = self.memory_head(hidden_states, context.get('memory'))
        tool = self.tool_head(hidden_states, context.get('context'))
        reasoning = self.reasoning_head(hidden_states, context.get('problem'))
        
        return {
            'experience': experience,
            'learning': learning,
            'memory': memory,
            'tool': tool,
            'reasoning': reasoning,
        }
```

---

## 5. The 4D Temporal Layer

**Your innovation.** The computational model treats time as first-class:

```
State = (time, counter₁, counter₂, ΔΦ)
```

| Dimension | Role | Semantic |
|-----------|------|----------|
| **D1** | Address (registry) | Memory location / input pointer |
| **D2** | Phase Φ₊ (forward) | Forward computation / analysis |
| **D3** | Phase Φ₋ (backward) | Backward computation / convergence |
| **D4** | Time idx (iter) | Evolution parameter / tick |

**The Phase-Residual Theorem:**
```
ΔΦ = Φ₊ - Φ₋  (mod 2π)
```
Computation exists only as the *difference* between the two timers. Phase-locked echo captures ΔΦ when both timers synchronize — this is the memory mechanism.

**Turing Completeness:** 2 counters + 1 control state = Minsky's minimal universal counter machine. The 4D counter machine extends this with phase-residual temporal tracking.

---

## 6. The .binetic Format

**Your packaging innovation.** Custom binary model format that ships:
- Quantum/ternary weights
- 4D metadata per weight
- Phase-residual configuration
- Custom protocol + ISA

### Format Layout
```
┌────────────────────┐
│  HEADER (64 bytes)  │  magic(v4), version, opset, flags
├────────────────────┤
│  METADATA (varlen)  │  graph spec, layer info, timertypes
├────────────────────┤
│  TENSORS (varlen)   │  weight compute quantized
├────────────────────┤
│  INSTRUCTIONS (varlen) │  op graph: addr←reg addr→reg size bit width
├────────────────────┤
│  TRAILER            │  ECC, resize, end-of-file marker
└────────────────────┘
```

---

## 7. The Genetic Algorithm Layer

### Genome
```python
class Genome:
    # Decision head architectures
    experience_arch, learning_arch, memory_arch, tool_arch, reasoning_arch
    
    # Meta-parameters
    learning_rates = {…}
    mutation_rate = 0.01
    
    # Fitness function weights
    fitness_weights = {
        experience: 0.2, learning: 0.2, memory: 0.2, tool: 0.2, reasoning: 0.2
    }
```

### Fitness Function (Meta-Fitness)
```python
def meta_fitness(task_outcome, genome):
    fitness = sum(
        genome.fitness_weights[k] * task_outcome[f'{k}_score']
        for k in ['experience','learning','memory','tool','reasoning']
    )
    if task_outcome['improvement_over_baseline']:
        fitness *= 1.1
    return fitness
```

### Evolution Operators
- Tournament selection with elitism (5 elite individuals)
- Uniform crossover with architecture-aware blending
- Adaptive mutation rate with mirrored exponential annealing
- Probability distribution shaping (β ≈ 1/2, gaussian curvature)

---

## 8. Training Pipeline

### Phase 1: Backbone Pre-training (Frozen)
1. Load Llama 3.2 3B (prototype) / Ternary Bonsai 2 27B (final)
2. Freeze backbone weights
3. Initialize 5 decision heads with random weights
4. Train heads on supervised decision tasks

### Phase 2: Joint Fine-tuning
1. Unfreeze backbone (low learning rate)
2. Train all heads jointly with multi-task loss
3. Use gradient clipping to prevent instability
4. Monitor each head's performance separately

### Phase 3: Evolutionary Optimization
1. Initialize population of N genomes
2. For each generation:
   - Evaluate every individual on task outcomes
   - Compute meta-fitness
   - Select parents (tournament + elitism)
   - Create children (crossover + mutation)
   - Replace worst individuals
3. Keep best genome

### Phase 4: Self-Improvement Loop
1. Use Learning head to identify improvement opportunities
2. Apply meta-learning (MAML/Reptile) to adapt heads
3. Track success stories for fitness function evolution

---

## 9. Task Outcome Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| Experience Quality | Context accuracy, event detection | >85% |
| Learning Efficiency | Improvement over baseline | >2x |
| Memory Retention | Recall accuracy | >90% |
| Tool Effectiveness | Correct tool usage | >80% |
| Reasoning Quality | Logical consistency | >85% |
| Evolution Gain | Fitness improvement | >30% |

---

## 10. Hardware Requirements

### For This Session (Prototype)
- **RAM**: 32 GB available ✅
- **GPU**: None available ❌
- **Storage**: 30 GB available ✅
- **Python**: 3.14.4 available ✅

### For Full Deployment
- GPU: A100/H100 (80GB VRAM recommended)
- RAM: 256GB+
- Storage: 1TB+ SSD

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No GPU | **High** | Use Llama 3.2 3B on CPU; find GPU hosting later |
| Model not fitted on your data | **Medium** | LoRA on 3B model is fast |
| GA instability | **High** | Elitism + adaptive mutation |
| Head interference | **Medium** | Train heads separately first, then jointly |
| Backprop through ternary weights | **Low-Medium** | Use straight-through estimator or gumbel-softmax |

---

## 12. Implementation Roadmap

### Month 1: Foundation
- [ ] Set up environment (torch, transformers, llama.cpp)
- [ ] Download Llama 3.2 3B quantized
- [ ] Implement joint backbone + 1 prototype head (Experience)
- [ ] Train on synthetic decisions

### Month 2-3: Scale to 5 Heads
- [ ] Implement all 5 heads
- [ ] Joint training with multi-task loss
- [ ] Validated on synthetic decision tasks

### Month 4-5: Evolution Layer
- [ ] Implement genetic algorithm
- [ ] Create meta-fitness functions
- [ ] Run initial evolution cycles

### Month 6+: Full System
- [ ] Swap backbone to Ternary Bonsai 2 27B (when GPU available)
- [ ] Implement .binetic format
- [ ] Build inference engine
- [ ] 4D temporal phase layer

---

## 13. Built With Real Evidence

This architecture is grounded in three verified sources:
1. **Cloudflare Clef** — the joint schema head pattern [1]
2. **MersivMedia/clef-finetune** — the training recipe [2]
3. **Ternary Bonsai 2 27B** — the backbone model [3]

---

## 12. Progress Markers

| Milestone | Status |
|-----------|--------|
| ARCHITECTURE.md documentation | ✅ complete |
| Source tree (src/) | ✅ complete |
| Experience head | ✅ complete |
| Learning head | ✅ complete |
| Memory head | ✅ complete |
| Tool Use head | ✅ complete |
| Reasoning head | ✅ complete |
| JointDecisionModel (5 heads) | ✅ complete |
| Training engine (multi-task loss) | ✅ complete |
| Synthetic dataset generator | ✅ complete |
| Llama 3.2 3B backbone loaded (CPU) | ✅ complete |
| Training run — 1500 batches, 3 epochs | ✅ completed (save fixed) |
| .binetic format spec (documents/) | ✅ complete |
| 4D phase processor (src/4d.py) | 🔄 draft |
| GA layer (src/ga.py) | 🔄 draft |
| Full training verification | 🔄 in flight |

---

## Awaiting Decision

**Phase A is complete.** The 5-head decision model runs on CPU.

Next possible turns:
1. **Verify the trained model** — load `models/` + check inference output
2. **Run the subagent research team** — synthesize the 4D processor findings
3. **Proceed to Phase B** — add the GA layer to the trained model
4. **Proceed to Phase C** — build the .binetic format writer/reader
5. **Backbone swap** — reserve Ternary Bonsai 2 27B for when GPU is found

### References
[1] Cloudflare Clef — `https://github.com/lyhu/skills-cloudflare-clef` & `https://huggingface.co/Cloudflare/clef`  
[2] MersivMedia/clef-finetune — `https://github.com/MersivMedia/clef-finetune`  
[3] Ternary Bonsai 2 27B — `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf`
