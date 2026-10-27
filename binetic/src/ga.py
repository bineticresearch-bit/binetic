"""
Genetic Algorithm layer for the decision model.

Evolves:
  1. Head architectures (num_experiences, num_learning_modes, etc.)
  2. Fitness weights (what matters for decision quality)
  3. Learning rates per head

Fitness is based on TASK OUTCOME — the actual results the decisions produce.
This is the key innovation over Clef: the GA evolves to improve real decisions.
"""
import random
import numpy as np
import torch


class Genome:
    """Encodes head configs + fitness weights + learning rates."""

    def __init__(self, head_configs=None, fitness_weights=None, learning_rates=None):
        if head_configs is None:
            head_configs = {
                'experience': {'num_experiences': 64},
                'learning': {'num_learning_modes': 10},
                'memory': {'memory_slots': 1000},
                'tool': {'num_tools': 50, 'param_dim': 8},
                'reasoning': {'num_reasoning_patterns': 20},
            }
        if fitness_weights is None:
            fitness_weights = {k: 0.2 for k in ['experience', 'learning', 'memory', 'tool', 'reasoning']}
        if learning_rates is None:
            learning_rates = {k: 0.001 for k in head_configs.keys()}

        self.head_configs = head_configs
        self.fitness_weights = fitness_weights
        self.learning_rates = learning_rates
        self.mutation_rate = 0.01

    def to_fitness_vector(self):
        """Flatten genome into a numeric vector for the GA."""
        vec = []
        for k, v in self.head_configs.items():
            for kk, vv in v.items():
                vec.append(float(vv))
        for k, v in self.fitness_weights.items():
            vec.append(float(v))
        for k, v in self.learning_rates.items():
            vec.append(float(v))
        return np.array(vec)

    @staticmethod
    def from_fitness_vector(vec, fitness_weights_keys):
        """Reconstruct genome from a numeric vector."""
        head_configs = {}
        p = 0
        for k in ['experience', 'learning', 'memory', 'tool', 'reasoning']:
            num_configs = len(['num_experiences', 'num_learning_modes']) if k in ['experience', 'learning'] else \
                          len(['memory_slots']) if k == 'memory' else \
                          len(['num_tools', 'param_dim']) if k == 'tool' else 1
            head_configs[k] = {}
            for kk in ['num_experiences', 'num_learning_modes', 'memory_slots', 'num_tools', 'param_dim', 'num_reasoning_patterns']:
                pass

        # Simplified: just track fitness weights + learning rates for now
        fitness_weights = {k: 0.2 for k in fitness_weights_keys}
        learning_rates = {k: 0.001 for k in fitness_weights_keys}
        return Genome(fitness_weights=fitness_weights, learning_rates=learning_rates)


class GeneticAlgorithm:
    """Evolutionary optimizer over decision head configurations."""

    def __init__(self, population_size=20, mutation_rate=0.01, elite_count=3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.population = []
        self.fitness_scores = []
        self.elite_index = None

    def initialize(self):
        """Create initial population of genomes."""
        self.population = [Genome() for _ in range(self.population_size)]
        return self.population

    def evaluate(self, genome, task_outcome):
        """Meta-fitness: quality of decisions on real tasks."""
        fitness = 0.0
        for key in ['experience', 'learning', 'memory', 'tool', 'reasoning']:
            weight = genome.fitness_weights.get(key, 0.2)
            score = task_outcome.get(f'{key}_quality', 0.0)
            fitness += weight * score
        return fitness

    def select(self, fitness_scores):
        """Tournament selection with elitism."""
        elite_indices = np.argsort(fitness_scores)[-self.elite_count:]

        selected = []
        for _ in range(self.population_size - self.elite_count):
            tournament = random.sample(range(len(fitness_scores)), 5)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            selected.append(winner)

        return elite_indices + selected

    def crossover(self, parent1, parent2):
        """Uniform crossover with architecture-aware blending."""
        child = Genome(head_configs=dict(parent1.head_configs))

        for key in parent1.head_configs:
            if random.random() < 0.5:
                child.head_configs[key] = parent2.head_configs[key]

        # Blend fitness weights
        for key in child.fitness_weights:
            child.fitness_weights[key] = (
                parent1.fitness_weights[key] * 0.5 +
                parent2.fitness_weights[key] * 0.5 +
                random.gauss(0, 0.1)
            )

        # Normalize fitness weights
        total = sum(child.fitness_weights.values())
        child.fitness_weights = {k: v / total for k, v in child.fitness_weights.items()}
        return child

    def mutate(self, genome):
        """Mutation with adaptive rate."""
        for key in genome.head_configs:
            if random.random() < self.mutation_rate:
                self._mutate_head(genome, key)

        for key in genome.fitness_weights:
            if random.random() < self.mutation_rate:
                genome.fitness_weights[key] += random.gauss(0, 0.05)

        # Normalize fitness weights
        total = sum(genome.fitness_weights.values())
        genome.fitness_weights = {k: v / total for k, v in genome.fitness_weights.items()}
        return genome

    def _mutate_head(self, genome, head_name):
        """Mutate one head's config."""
        k, lhs = next(iter(genome.head_configs[head_name].items()))
        if random.random() < 0.5:
            gma = genome.head_configs[head_name][k] + random.gauss(0, 0.5)
            genome.head_configs[head_name][k] = max(1, gma)
        else:
            genome.head_configs[head_name][k] += random.randint(-2, 2)

    def evolve(self, fitness_function, generations=10):
        """Main evolution loop."""
        best_fitness = 0.0
        best_genome = None

        for gen in range(generations):
            # Evaluate
            fitness_scores = []
            for genome in self.population:
                score = fitness_function(genome)
                fitness_scores.append(score)

            # Select parents
            selected = self.select(fitness_scores)

            # Create children (crossover + mutation)
            new_population = []
            for i in range(0, len(selected), 2):
                p1 = self.population[selected[i]]
                p2 = self.population[selected[i-1]] if i > 0 else self.population[selected[0]]
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            # Replace worst
            worst = np.argsort(fitness_scores)[:self.population_size - len(new_population)]
            for idx in worst:
                self.population[idx] = new_population.pop(0) if new_population else self.population[idx]

            # Track best
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_genome = self.population[best_idx]

            print(f"[ga] generation {gen+1}/{generations} "
                  f"best_fitness={best_fitness:.3f} "
                  f"mean={np.mean(fitness_scores):.3f}")

        return best_genome, best_fitness
