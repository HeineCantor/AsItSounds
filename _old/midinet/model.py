import keras

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, LeakyReLU, Concatenate


class MidiNET():
    def __init__(self, pitch_range):    # pitch_range sarà 128
        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 13
        self.pitch_range = pitch_range

    def get_discriminator(self):
        x = Input(shape=(self.df_dim,))
        y = Input(shape=(self.y_dim,))

        yb = Reshape(target_shape=(self.y_dim, 1, 1))(y)
        #x = Concatenate()([x, yb])

        d = Conv2D(filters=14, kernel_size=(2, 128), strides=2, padding='same')(yb)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters=77, kernel_size=(4, 1), strides=2, padding='same')(d)
        d = LeakyReLU(alpha=0.2)(d)

        model = Model(inputs=[x, y], outputs=Dense(self.dfc_dim)(d))

        model.summary()
        return model
    
midinet = MidiNET(13)
midinet.get_discriminator()