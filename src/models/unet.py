# src/models/unet_model.py
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from configs.config import Config

class UNetModel:
    def __init__(self, input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)):
        self.input_shape = input_shape
        self.model = self.build_unet()

    def build_unet(self):
        inputs = Input(self.input_shape)

        # Encoder
        c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
        c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2,2))(c1)

        c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
        c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2,2))(c2)

        c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
        c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2,2))(c3)

        c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
        c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D((2,2))(c4)

        # Bottleneck
        c5 = Conv2D(1024, (3,3), activation='relu', padding='same')(p4)
        c5 = Conv2D(1024, (3,3), activation='relu', padding='same')(c5)

        # Decoder
        u6 = UpSampling2D((2,2))(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(512, (3,3), activation='relu', padding='same')(u6)
        c6 = Conv2D(512, (3,3), activation='relu', padding='same')(c6)

        u7 = UpSampling2D((2,2))(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(256, (3,3), activation='relu', padding='same')(u7)
        c7 = Conv2D(256, (3,3), activation='relu', padding='same')(c7)

        u8 = UpSampling2D((2,2))(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(128, (3,3), activation='relu', padding='same')(u8)
        c8 = Conv2D(128, (3,3), activation='relu', padding='same')(c8)

        u9 = UpSampling2D((2,2))(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv2D(64, (3,3), activation='relu', padding='same')(u9)
        c9 = Conv2D(64, (3,3), activation='relu', padding='same')(c9)

        outputs = Conv2D(Config.NUM_CLASSES, (1,1), activation='sigmoid')(c9)
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def compile(self, lr=Config.LEARNING_RATE):
        from keras.optimizers import Adam
        self.model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()