"""
    Contains used config variable
"""

# PATH config
from pathlib import Path
MODEL_PATH = Path('../FukuroRegression/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# MODEL_NAME = 'ParabolicMirrorModel.pth'
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
DATA_PATH = Path('../FukuroRegression/data')
DATA_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch config
from torch import cuda
RANDOM_SEED = 42
TRAIN_SIZE = 0.8
LEARNING_RATE = 1e-03
PATIENCE = 100
EPOCHS = 999999999
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

# Visualization config
FIG_SIZE = (10, 7)

# Diameter of parabolic mirror
DIAMETER = 15