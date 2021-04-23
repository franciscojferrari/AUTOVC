import tensorflow as tf
from tensorflow.python.framework.ops import convert_n_to_tensor
from layers import ConvNorm
tfk = tf.keras
tfkl = tfk.layers


class AutoVC(tfk.Model):
    def __init__(self, name, dim_pre,*args, **kwargs):
        super(AutoVC).__init__(*args, **kwargs)
        self.name = name

        self.encoder = None # TODO
        self.decoder = self.build_decoder(dim_pre)
        self.postnet = self.build_postnet()

    def build_decoder(self, dim_pre):
        _3convs = [ConvNorm(name=f"dec_conv_{i}", filters=dim_pre,
                          kernel_size=5, stride=1, dilation=1, activation="relu") for i in range(3)]
        _3lstms = [
            tfkl.LSTM(name=f"dec_lstm_{i}", units=1024, return_sequences=True) for i in range(3)]
        decoder = tfk.Sequential((_3convs + _3lstms + tfkl.Dense(80)))
        return decoder
    
    def build_postnet(self):
        convs = [ConvNorm(name=f"dec_conv_{i}", filters=512,
                          kernel_size=5, stride=1, dilation=1, activation="tanh") for i in range(5)]
        convs.append(ConvNorm(name=f"dec_conv_{5}", filters=80,
                          kernel_size=5, stride=1, dilation=1, activation=None))
        postnet = tfk.Sequential(convs)
        return postnet





    
