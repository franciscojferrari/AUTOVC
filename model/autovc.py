import tensorflow as tf

from layers import ConvNorm

tfk = tf.keras
tfkl = tfk.layers


class AutoVC(tfk.Model):
    def __init__(self, name, dim_pre, *args, **kwargs):
        super(AutoVC).__init__(*args, **kwargs)
        self.name = name

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.postnet = self.build_postnet()
        self.freq = kwargs.get("freq", 32)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.dim_pre = kwargs.get("dim_pre", None)

        # TODO: Check values and throw value exception if are none

    def build_decoder(self):
        _3convs = [ConvNorm(name = f"dec_conv_{i}", filters = self.dim_pre,
                            kernel_size = 5, stride = 1, dilation = 1, activation = "relu") for i in range(3)]
        _3lstms = [
            tfkl.LSTM(name = f"dec_lstm_{i}", units = 1024, return_sequences = True) for i in range(3)]
        decoder = tfk.Sequential((_3convs + _3lstms + tfkl.Dense(80)))
        return decoder

    def build_postnet(self):
        convs = [ConvNorm(name = f"dec_conv_{i}", filters = 512,
                          kernel_size = 5, stride = 1, dilation = 1, activation = "tanh") for i in range(5)]

        convs.append(ConvNorm(name = f"dec_conv_{5}", filters = 80,
                              kernel_size = 5, stride = 1, dilation = 1, activation = None))
        postnet = tfk.Sequential(convs)
        return postnet

    def build_encoder(self):
        # What about input dimensions?

        _3convs = [ConvNorm(name = f"enc_conv_{i}", filters = 512,
                            kernel_size = 5, stride = 1, dilation = 1, activation = "relu") for i in range(3)]

        _2BLSTM = [tfkl.Bidirectional(tfkl.LSTM(name = f"enc_lstm_{i}", units = self.dim_neck, return_sequences = True))
                   for i in range(2)]
        # TODO: Add input layer if necessary
        encoder = tfk.Sequential(_3convs + _2BLSTM)
        return encoder

    def __call__(self, *args, **kwargs):
        _X1 = kwargs.get("X1", args[0])  # Freq of speaker 1
        _S1 = kwargs.get("S1", args[1])  # Embedding of Speaker 1
        _S2 = kwargs.get("S2", args[2])  # Embedding of Speaker 2 (target)

        # _X1 = tf.squeeze(_X1, axis = 1)
        # _S1 = _S1.expand_dims(_S1, -1)
        # _S1 = tf.tile() In case that automatic broadcast of dimensions on TF doesn't work

        encoder_input = tf.concat([_X1, _S1], axis = 1)

        # TODO: WHEN CALLING THE ENCODER WE NEED TO SQUEEZE AND CONCAT THE 2 INPUTS OF THE BLOCK
        # Down and Upsample from the encoder
        encoder_output = self.encoder(encoder_input)
        encoder_output_forward = encoder_output[:, :, :self.dim_neck]
        encoder_output_backward = encoder_output[:, :, self.dim_neck:]

        _C_forward, _C_backward = [], []
        for i in range(0, encoder_output.shape[1], self.freq):
            _C_forward.append(encoder_output_forward[:, i, :])
            _C_backward.append(encoder_output_backward[:, i + self.freq - 1, :])
            # codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim = -1))

        _C_forward = tf.stack(_C_forward, axis = 1)
        _C_backward = tf.stack(_C_backward, axis = 1)

        _C_forward_UP = tfkl.UpSampling1D(size = self.freq)(_C_forward)
        _C_backward_UP = tfkl.UpSampling1D(size = self.freq)(_C_backward)

        # Decoder
        decoder_input = tf.concat([_C_forward_UP, _C_backward_UP, _S1], axis = -1)
        mel_output = self.decoder(decoder_input)

        mel_output_postnet = self.postnet(mel_output)

        mel_output_sum = mel_output + mel_output_postnet

        return mel_output_sum, mel_output_postnet, (_C_forward, _C_backward)

    def loss_function(self, *args, **kwargs):
        x_real = kwargs.get("X1", args[0])

        mel_output_sum = kwargs.get("mel_output_sum", args[1])
        mel_output_postnet = kwargs.get("mel_output_postnet", args[2])
        _C_forward, _C_backward = kwargs.get("codes", args[3])

        autovc_loss_id = tfk.losses.MSE(x_real, mel_output_sum)
        autovc_loss_id_psnt = tfk.losses.MSE(x_real, mel_output_postnet)
