from config.train import TrainConfig
from src.utils.helper import run
from src.utils.train import train

if __name__ == '__main__':
    config = TrainConfig()
    run(train, config)
