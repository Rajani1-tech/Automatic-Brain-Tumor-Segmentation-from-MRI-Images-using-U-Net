# src/models/unet_model.py
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf
from configs.config import Config

class UNetModel:
    def __init__(self, input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)):
        self.input_shape = input_shape
        self.model = self.build_unet()

    def build_unet(self):
        inputs = Input(self.input_shape)

        c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
        c1 = Conv2D(32, (3,3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2,2))(c1)

        c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
        c2 = Conv2D(64, (3,3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2,2))(c2)

        c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
        c3 = Conv2D(128, (3,3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2,2))(c3)

        c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
        c4 = Conv2D(256, (3,3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D((2,2))(c4)

        c5 = Conv2D(512, (3,3), activation='relu', padding='same')(p4)
        c5 = Conv2D(512, (3,3), activation='relu', padding='same')(c5)

        u6 = UpSampling2D((2,2))(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3,3), activation='relu', padding='same')(u6)
        c6 = Conv2D(256, (3,3), activation='relu', padding='same')(c6)

        u7 = UpSampling2D((2,2))(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3,3), activation='relu', padding='same')(u7)
        c7 = Conv2D(128, (3,3), activation='relu', padding='same')(c7)

        u8 = UpSampling2D((2,2))(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3,3), activation='relu', padding='same')(u8)
        c8 = Conv2D(64, (3,3), activation='relu', padding='same')(c8)

        u9 = UpSampling2D((2,2))(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv2D(32, (3,3), activation='relu', padding='same')(u9)
        c9 = Conv2D(32, (3,3), activation='relu', padding='same')(c9)

        outputs = Conv2D(Config.NUM_CLASSES, (1,1), activation='sigmoid')(c9)

        return Model(inputs=[inputs], outputs=[outputs])
    
    def compile(self, lr=Config.LEARNING_RATE):
        def dice_coef(y_true, y_pred):
            smooth = 1.0
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2.0 * intersection + smooth) / (
                tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
            )

        def iou_coef(y_true, y_pred):
            smooth = 1.0
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
            return (intersection + smooth) / (union + smooth)

        def dice_loss(y_true, y_pred):
            return 1.0 - dice_coef(y_true, y_pred)

        def bce_dice_loss(y_true, y_pred):
            # Squeeze both to [B, H, W] — handles [B,H,W,1] or [B,H,W]
            y_true = tf.squeeze(y_true, axis=-1)
            y_pred = tf.squeeze(y_pred, axis=-1)

            # Compute BCE manually — avoids Keras internal shape reduction issues
            y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce = -(y_true * tf.math.log(y_pred_clipped) +
                    (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))

            weight = 5.0
            weighted_bce = bce * (y_true * weight + 1.0)

            return tf.reduce_mean(weighted_bce) + dice_loss(y_true, y_pred)

        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=bce_dice_loss,
            metrics=['accuracy', dice_coef, iou_coef]
        )

 
    def summary(self):
        return self.model.summary()