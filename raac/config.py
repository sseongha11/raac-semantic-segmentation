import torch
import segmentation_models_pytorch.utils as su

X_TRAIN_DIR = '/content/drive/MyDrive/dataset/train'
Y_TRAIN_DIR = '/content/drive/MyDrive/dataset/train_labels'

X_VALID_DIR = '/content/drive/MyDrive/dataset/val'
Y_VALID_DIR = '/content/drive/MyDrive/dataset/val_labels'

X_TEST_DIR = '/content/drive/MyDrive/dataset/test'
Y_TEST_DIR = '/content/drive/MyDrive/dataset/test_labels'

# Get class names
CLASS_NAME = ['background', 'crack']

# Get class RGB values
CLASS_RGB_VALUES = [[0, 0, 0], [255, 255, 255]]  # select_class_rgb_values

CLASS_IDX = [0, 1]  # select_class_indices

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = CLASS_NAME
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

# Set flag to train the models or not. If set to 'False', only prediction is performed (using an older models checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 70
INIT_LR = 1e-4
TH = 0.5

# Define loss function
LOSS = su.losses.DiceLoss()

# Define metrics
METRIC = [
    su.metrics.IoU(threshold=TH),
    su.metrics.Fscore(threshold=TH),
    su.metrics.Precision(threshold=TH),
    su.metrics.Recall(threshold=TH),
]

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
