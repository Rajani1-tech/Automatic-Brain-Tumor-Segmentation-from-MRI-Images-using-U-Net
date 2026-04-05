import os

class Config:
    IMG_HEIGHT   = 240
    IMG_WIDTH    = 240
    IMG_CHANNELS          = 1  
    MULTICLASS_IN_CHANNELS = 4  

    NUM_CLASSES  = 4
    CLASS_NAMES  = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]

    CLASS_COLORS = [
        (0,   0,   0),    # Background  — black
        (255, 0,   0),    # NCR/NET     — red
        (0,   255, 0),    # Edema       — green
        (0,   0,   255),  # ET          — blue
    ]

    NUM_TUMOR_CLASSES  = 4
    TUMOR_CLASS_NAMES  = [
        "No Tumor",          # 0 — mask all zeros
        "Edema Only",        # 1 — only label 2 present
        "Core Present",      # 2 — label 1 present (with/without edema), no label 3
        "Full Tumor",        # 3 — all three regions present (most severe)
    ]
    TUMOR_CLASS_COLORS = ["#4CAF50", "#FF9800", "#FF5722", "#B71C1C"]

    BATCH_SIZE    = 16
    EPOCHS        = 75
    LEARNING_RATE = 1e-4

    TRAIN_IMAGES_DIR = "dataset/train/images"
    TRAIN_MASKS_DIR  = "dataset/train/masks"
    VAL_IMAGES_DIR   = "dataset/val/images"
    VAL_MASKS_DIR    = "dataset/val/masks"
    TEST_IMAGES_DIR  = "dataset/test/images"
    TEST_MASKS_DIR   = "dataset/test/masks"

    CLASSIFIER_TRAIN_IMAGES = TRAIN_IMAGES_DIR
    CLASSIFIER_TRAIN_MASKS  = TRAIN_MASKS_DIR
    CLASSIFIER_VAL_IMAGES   = VAL_IMAGES_DIR
    CLASSIFIER_VAL_MASKS    = VAL_MASKS_DIR
    CLASSIFIER_TEST_IMAGES  = TEST_IMAGES_DIR
    CLASSIFIER_TEST_MASKS   = TEST_MASKS_DIR

    UNET_BINARY_PATH      = "models/unet_binary.pth"
    ATTN_UNET_BINARY_PATH = "models/attention_unet_binary.pth"
    UNET_MULTI_PATH       = "models/unet_multiclass.pth"
    ATTN_UNET_MULTI_PATH  = "models/attention_unet_multiclass.pth"
    CLASSIFIER_PATH       = "models/tumor_classifier.pth"

    OUTPUTS_DIR = "outputs"

    GRADE_CLASS_NAMES  = ["LGG", "HGG"]   
    GRADE_CLASS_COLORS = ["#4CAF50", "#B71C1C"]

    GRADE_TRAIN_DIR = "dataset/classifier/train"  
    GRADE_VAL_DIR   = "dataset/classifier/val"
    GRADE_PATH      = "models/grade_classifier.pth"

    GRADE_BATCH_SIZE    = 32
    GRADE_EPOCHS        = 30
    GRADE_LEARNING_RATE = 1e-4