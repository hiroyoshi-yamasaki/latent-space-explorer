

# GUI parameters
WIDTH = 1000
HEIGHT = 500
DISPLAY = int(min(WIDTH, HEIGHT) * 0.8)
IMG_SIZE = 28
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 30
N_ROWS = 2
N_COLS = 10
CMAP = "plasma"

# Autoencoder parameters
LEARNING_RATE = 1e-3
DECAY = 1e-8
BATCH_SIZE = 128

LAYER_1 = 128
LAYER_2 = 32
KERNEL_1 = 4
KERNEL_2 = 4

CHANNELS_1 = 5
CHANNELS_2 = 5
HIDDEN_1 = 12
HIDDEN_2 = 20

MODEL_PATH = "weights/autoencoder.py"  # Path to saved model
