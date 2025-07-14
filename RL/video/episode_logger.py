import numpy as np
import json
from typing import List, Any, Dict, Optional

class EpisodeLogger:
    def __init__(self):
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.infos: List[Dict] = []

    def log_step(self, action, reward, info: Optional[Dict] = None):
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info if info is not None else {})

    def save_npz(self, path: str):
        np.savez(path, actions=np.array(self.actions), rewards=np.array(self.rewards), infos=np.array(self.infos, dtype=object))

    def save_json(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'actions': [a.tolist() if hasattr(a, 'tolist') else a for a in self.actions],
                'rewards': self.rewards,
                'infos': self.infos
            }, f, indent=2) 