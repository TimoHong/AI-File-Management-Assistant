# src/data/settings.py
from dataclasses import dataclass
import json
import os

@dataclass
class AISettings:
    num_groups: int = 3
    min_similarity: float = 0.5
    use_gpu: bool = True

class Settings:
    def __init__(self):
        self.ai_settings = AISettings()
        self.load()

    def load(self):
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                data = json.load(f)
                self.ai_settings.num_groups = data.get('ai_num_groups', 3)
                self.ai_settings.min_similarity = data.get('ai_min_similarity', 0.5)
                self.ai_settings.use_gpu = data.get('ai_use_gpu', True)

    def save(self):
        data = {
            'ai_num_groups': self.ai_settings.num_groups,
            'ai_min_similarity': self.ai_settings.min_similarity,
            'ai_use_gpu': self.ai_settings.use_gpu
        }
        with open('settings.json', 'w') as f:
            json.dump(data, f, indent=4)