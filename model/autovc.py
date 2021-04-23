import tensorflow as tf
from layers import ConvNorm
tfk = tf.keras
tfkl = tfk.layers

class Decoder(tfk.Layer):
    def __init__(self, name, dim_pre, **kwargs):
        super().__init__(name=name, **kwargs)

        self.lstms = []
        for i in range(3):
            self.lstms.append(
                tfkl.LSTM(name=f"dec_lstm_{i}", units=dim_pre, return_sequences=True))

        self.convolutions = []
        for i in range(3):
            # TODO double check padding
            self.convolutions.append(
                ConvNorm(name=f"dec_conv_{i}", filters=dim_pre,
                         kernel_size=5, stride=1, dilation=1, activation="relu")
            )

        self.linear_projection = tfkl.Dense(80)

    def call(self, x):
        x = self.lstms[0](x)  # [batch, time, lstm_units]
        for conv in self.convolutions:
            x = conv(x)
        x = self.lstms[2](self.lstms[1](x))

        return self.linear_projection(x)
