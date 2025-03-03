#/backend/config.py
import yaml

class Config:
    def __init__(self, config_file='../config/config.yaml'):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)
