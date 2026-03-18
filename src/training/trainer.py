# src/training/trainer.py

from keras.callbacks import ModelCheckpoint, EarlyStopping
from src.datasets.dataset import BrainTumorDataset
from configs.config import Config

class Trainer:
    def __init__(self, model, train_dir=Config.TRAIN_IMAGES_DIR, train_mask=Config.TRAIN_MASKS_DIR,
                 val_dir=Config.VAL_IMAGES_DIR, val_mask=Config.VAL_MASKS_DIR):
        self.model = model
        self.train_dataset = BrainTumorDataset(train_dir, train_mask)
        self.val_dataset = BrainTumorDataset(val_dir, val_mask)

    def train(self, epochs=Config.EPOCHS, save_path=Config.MODEL_SAVE_PATH):
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
        earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.model.fit(self.train_dataset,
                             validation_data=self.val_dataset,
                             epochs=epochs,
                             callbacks=[checkpoint, earlystop])