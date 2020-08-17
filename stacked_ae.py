'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-12
  email        : bao.salirong@gmail.com
  Task         : Stacked AutoEncoder based on Keras Model
'''

import tensorflow as tf

class Stacked_Encoder(tf.keras.Model):
    def __init__(self, units = 32, name = "bao_encoder"):
        super(Stacked_Encoder, self).__init__(name = name)

        self.flatten= tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units = units << 4, activation = "relu")
        self.dense2 = tf.keras.layers.Dense(units = units << 2, activation = "relu")

    def call(self, inputs):
        x = inputs
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class Stacked_Decoder(tf.keras.Model):
    def __init__(self, input_shape, units = 32, name = "bao_decoder"):
        super(Stacked_Decoder, self).__init__(name = name)

        h = input_shape[1]
        w = input_shape[2]
        c = input_shape[3]

        self.de_dense1 = tf.keras.layers.Dense(units = units << 2, activation = "relu")
        self.de_dense2 = tf.keras.layers.Dense(units = units << 4, activation = "relu")
        self.de_dense3 = tf.keras.layers.Dense(units = h * w * c, activation="relu")
        self.reshape   = tf.keras.layers.Reshape((w,h,c), name = "de_main_out")

    def call(self, inputs):
        x = inputs
        x = self.de_dense1(x)
        x = self.de_dense2(x)
        x = self.de_dense3(x)
        x = self.reshape(x)
        return x

class Stacked_ae(tf.keras.Model):
    def __init__(self, latent = 100, units = 32, input_shape = None):
        super(Stacked_ae, self).__init__()

        self.encoder = Stacked_Encoder(units = units)
        self.la_dense= tf.keras.layers.Dense(units = latent, activation="relu")
        self.decoder = Stacked_Decoder(input_shape = input_shape, units = units)

    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.la_dense(x)
        x = self.decoder(x)
        return x

#------------------------------------------------------------------------------
def Stacked_AE(input_shape, latent, units):
    model = Stacked_ae(latent = latent, units = units, input_shape = input_shape)
    model.build(input_shape = input_shape)
    return model
