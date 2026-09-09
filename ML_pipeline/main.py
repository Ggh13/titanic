
from omegaconf import DictConfig
from utils import set_seed

def main(config):
    set_seed(config.seed)