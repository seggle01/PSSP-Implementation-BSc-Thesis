import json
from pathlib import Path

######### Singleton configuration loader #########
class Config:
    _instance = None

    def __new__(cls, path=None):
        # Lazy-instantiate on first access
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            p = path or Path(__file__).parent / "config.json"
            with open(p) as f:
                data = json.load(f)
            cls._instance.__dict__.update(data)
        return cls._instance

    # Mapping-style access (cfg['KEY'])
    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)
