import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers


class ConvNorm(tfkl.Layer):
    # TODO double check the padding mode
    def __init__(self, name, filters, kernel_size, strides=1, padding="same", 
                dilation_rate=1, use_bias=True, activation='linear', **kwargs):
        super(ConvNorm, self).__init__(name=name, **kwargs)
        # default kernel initializer GlorotUniform, no need to specify it
        self.conv = tfkl.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, 
                                padding=padding, use_bias=use_bias,
                                dilation_rate=dilation_rate, activation=activation)
        # momentum in tensorflow = 1-momentum in pytorch
        self.bn = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, x):
        return self.bn(self.conv(x))



