# configs/config.py
class Config:
    IMG_HEIGHT = 240
    IMG_WIDTH = 240
    IMG_CHANNELS = 1
    NUM_CLASSES = 1

    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    TRAIN_IMAGES_DIR = "data/train/images/"
    TRAIN_MASKS_DIR = "data/train/masks/"
    VAL_IMAGES_DIR = "data/val/images/"
    VAL_MASKS_DIR = "data/val/masks/"
    MODEL_SAVE_PATH = "models_saved/unet_brain_tumor.pth"