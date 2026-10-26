# import the necessary packages
import torch
import os

# salines
#BASE_PATH = "/matieres/5MMVORF/03-dataset"
# dancers
BASE_PATH = "/matieres/5MMVORF/04-dataset"
IMAGES_PATH = os.path.join(BASE_PATH, "images")
MASKS_PATH = os.path.join(BASE_PATH, "masks")

# define the path to the base output directory
BASE_OUTPUT = "output"

# define paths to output model, plot and testing image paths
BEST_MODEL_PATH = os.path.join(BASE_OUTPUT, "best_model.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, "convergence_plot.png")
PLOT_PREDICT_PATH = os.path.join(BASE_OUTPUT, "predict_plot.png")
TEST_PATH = os.path.join(BASE_OUTPUT, "test_data.csv")
VAL_PATH = os.path.join(BASE_OUTPUT, "val_data.csv")
TRAIN_PATH = os.path.join(BASE_OUTPUT, "training_data.csv")

# define the test split
TEST_SPLIT = 0.15
TEST_VAL_SPLIT = [0.1, 0.1]

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 20
BATCH_SIZE = 256

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# define threshold to filter weak predictions
THRESHOLD = 0.5
