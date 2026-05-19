import numpy as np
from typing import List, Tuple, Dict, Optional


class TinyWorld:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def step(self) -> Tuple[np.ndarray, int]:
        raise NotImplementedError

    def observe(self) -> np.ndarray:
        raise NotImplementedError


class CausalSequenceWorld(TinyWorld):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.causal_rules = {
            1: [2, 3],
            2: [4],
            3: [5],
            4: [6, 7],
            5: [8],
            6: [9],
            7: [10],
            8: [11],
            9: [12],
            10: [13, 14],
        }
        self.current_state = 1
        self.history: List[int] = []

    def reset(self):
        super().reset()
        self.current_state = 1
        self.history = [1]

    def step(self) -> Tuple[np.ndarray, int]:
        # Follow causal rule or random transition
        if self.current_state in self.causal_rules:
            next_states = self.causal_rules[self.current_state]
            self.current_state = self.rng.choice(next_states)
        else:
            self.current_state = self.rng.randint(1, 15)

        self.history.append(self.current_state)
        self.step_count += 1
        return self.observe(), self.current_state

    def observe(self) -> np.ndarray:
        vec = np.zeros(15, dtype=np.float32)
        vec[self.current_state] = 1.0
        vec[0] = self.step_count / 100.0
        return vec

    def get_sequence(self, length: int = 20) -> List[int]:
        self.reset()
        seq = [1]
        for _ in range(length - 1):
            _, s = self.step()
            seq.append(s)
        return seq


class ObjectInteractionWorld(TinyWorld):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.objects = ['fire', 'water', 'stone', 'wood', 'iron', 'plant', 'animal']
        self.properties = {
            'fire': ['hot', 'bright', 'consumes'],
            'water': ['wet', 'flows', 'cools'],
            'stone': ['hard', 'heavy', 'inert'],
            'wood': ['burns', 'floats', 'light'],
            'iron': ['hard', 'heavy', 'conducts'],
            'plant': ['grows', 'green', 'living'],
            'animal': ['moves', 'living', 'warm'],
        }
        self.interactions = [
            ('fire', 'wood', 'ash'),
            ('fire', 'water', 'steam'),
            ('water', 'plant', 'growth'),
            ('stone', 'iron', 'spark'),
            ('animal', 'plant', 'eats'),
            ('fire', 'animal', 'danger'),
        ]
        self.current_objects = ['fire', 'wood']
        self.current_action = None
        self.result = None

    def reset(self):
        self.current_objects = [self.rng.choice(self.objects)]
        self.current_action = None
        self.result = None

    def step(self) -> Tuple[np.ndarray, int]:
        obj1 = self.rng.choice(self.objects)
        obj2 = self.rng.choice(self.objects)

        result = None
        for (a, b, r) in self.interactions:
            if (obj1 == a and obj2 == b) or (obj1 == b and obj2 == a):
                result = r
                break

        self.current_objects = [obj1, obj2]
        self.current_action = (obj1, obj2)
        self.result = result
        self.step_count += 1
        return self.observe(), self.objects.index(result) if result else 0

    def observe(self) -> np.ndarray:
        vec = np.zeros(32, dtype=np.float32)
        for i, obj in enumerate(self.current_objects):
            if obj in self.objects:
                vec[self.objects.index(obj)] = 1.0
        if self.result and self.result in self.objects:
            vec[len(self.objects) + self.objects.index(self.result)] = 1.0
        vec[-1] = 1.0 if self.result else 0.0
        return vec


class SensorimotorWorld(TinyWorld):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.position = 0
        self.grid_size = 10
        self.reward_piles = [(3, 1.0), (7, 0.5), (2, 0.8)]
        self.energy = 1.0

    def reset(self):
        self.position = 0
        self.energy = 1.0

    def step(self) -> Tuple[np.ndarray, int]:
        move = self.rng.choice([-1, 0, 1])
        self.position = max(0, min(self.grid_size - 1, self.position + move))
        self.energy = max(0.0, self.energy - 0.02)

        reward = 0.0
        for pos, val in self.reward_piles:
            if self.position == pos:
                reward = val
                self.energy = min(1.0, self.energy + 0.3)
                break

        self.step_count += 1
        return self.observe(), int(reward * 10)

    def observe(self) -> np.ndarray:
        vec = np.zeros(self.grid_size + 2, dtype=np.float32)
        vec[self.position] = 1.0
        vec[-2] = self.energy
        vec[-1] = self.step_count / 50.0
        return vec
