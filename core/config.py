import os

base_path = os.path.dirname(__file__)

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PURPLE = (116, 20, 73)
ORANGE = (21, 71, 229)
COLORS = [BLUE, GREEN, RED, PURPLE, ORANGE]


DEFAULT_CONF = 0.5
KERNEL_SIZE = 13
POSITIONS = [0.2, 0.35, 0.5, 0.65, 0.8]

MODEL_NAME ="best.pt"
IMAGE = "sample_noise_20.jpg"
SAMPLES_DIR = "core/samples"
MODELS_DIR = "core/models"
OUTPUT_DIR = "core/output/"
