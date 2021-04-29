import tensorflow as tf

from layers import ConvNorm

tfk = tf.keras
tfkl = tfk.layers


class Encoder(tfkl.Layer):
    def __init__(self, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.freq = kwargs.get("freq", 32)
        self.model = self.build_encoder()
        # TODO: Assert values

    def build_encoder(self):
        # input_layer = tfkl.InputLayer(
        #     input_shape = self.dim_emb + self.mel_feature_dim, name = "enc_input")
        _3convs = [ConvNorm(name = f"enc_conv_{i}", filters = 512,
                            kernel_size=5, strides=1, dilation_rate=1, activation="relu") for i in range(3)]

        _2BLSTM = [tfkl.Bidirectional(tfkl.LSTM(name = f"enc_lstm_{i}", units = self.dim_neck, return_sequences = True))
                   for i in range(2)]
        encoder = tfk.Sequential(_3convs + _2BLSTM)
        return encoder

    def call(self, mel_spec, speak_emb):
        input = tf.concat([mel_spec, speak_emb], axis = 1)
        output = self.model(input)

        output_forward = output[:, :, :self.dim_neck]
        output_backward = output[:, :, self.dim_neck:]

        # downsampling
        codes = []
        for i in range(0, output.shape[1], self.freq):
            codes.append(tf.concat(
                [output_forward[:, i, :], output_backward[:, i + self.freq - 1, :]], axis = -1))
        # TODO: double check if concat or stack
        codes = tf.concat(codes, axis = -1)
        return codes


class Decoder(tfkl.Layer):
    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_pre = kwargs.get("dim_pre", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.model = self.build_decoder()

    def build_decoder(self):
        # input_layer = tfkl.InputLayer(
        #     input_shape = self.dim_neck * 2 + self.dim_emb, name = "dec_input")
        _3convs = [ConvNorm(name = f"dec_conv_{i}", filters = self.dim_pre,
                            kernel_size=5, strides=1, dilation_rate=1, activation="relu") for i in range(3)]
        _3lstms = [
            tfkl.LSTM(name = f"dec_lstm_{i}", units = 1024, return_sequences = True) for i in range(3)]
        decoder_layers = tfk.Sequential(
             _3convs + _3lstms + [tfkl.Dense(self.mel_feature_dim)])
        return decoder_layers

    def call(self, input):
        # TODO: double check, we may need to transpose if we first start with LSTM
        return self.model(input)


class PostNet(tfkl.Layer):
    def __init__(self, name="postnet", **kwargs):
        super(PostNet, self).__init__(name=name)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.model = self.build_postnet()

    def build_postnet(self):
        # input_layer = tfkl.InputLayer(
        #     input_shape = self.mel_feature_dim, name = "posnet_input")
        convs = [ConvNorm(name = f"dec_conv_{i}", filters = 512,
                          kernel_size=5, strides=1, dilation_rate=1, activation="tanh") for i in range(5)]

        convs.append(ConvNorm(name = f"dec_conv_{5}", filters = self.mel_feature_dim,
                              kernel_size=5, strides=1, dilation_rate=1, activation=None))
        posnet_layers = tfk.Sequential(convs)
        return posnet_layers

    def call(self, input):
        return self.posnet_layers(input)


class AutoVC(tfk.Model):
    def __init__(self, name="AutoVC", *args, **kwargs):
        super(AutoVC, self).__init__(name=name)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_pre = kwargs.get("dim_pre", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.freq = kwargs.get("freq", 32)
        self.lamda = kwargs.get("lamda", 0.01)

        self.encoder = Encoder(dim_emb = self.dim_emb,
                               dim_neck = self.dim_neck, freq = self.freq,
                               mel_dim = self.mel_feature_dim)
        self.decoder = Decoder(dim_emb = self.dim_emb,
                               dim_pre = self.dim_pre, dim_neck = self.dim_neck)
        self.postnet = PostNet(mel_dim = self.mel_feature_dim)

        # TODO: Check values and throw value exception if are none

    def call(self, mel_spec, speak_emb, speak_emb_trg):
        codes = self.encoder(mel_spec, speak_emb)
        codes_exp = tfkl.UpSampling1D(size = self.freq)(codes)

        encoder_output = tf.concat([codes_exp, speak_emb_trg], axis = -1)
        mel_output = self.decoder(encoder_output)

        mel_output_postnet = self.postnet(mel_output) + mel_output

        custom_loss = self.custom_loss(mel_spec, speak_emb,
                                       mel_output, mel_output_postnet, codes)

        self.add_loss(custom_loss)

        return mel_output, mel_output_postnet, codes

    def custom_loss(self, x_real, speak_emb, mel_output, mel_output_postnet, code_real):
        """
        First bit:
            Identity mapping loss
                Reconstruction error between mel input and mel output (Before post net--> Why? No clear reason why. It just improves the results)
                Reconstruction error between mel input and final output (post net)
            Content Loss (semantic loss)
                Reconstruction error between bottle neck and reconstructed bottle neck
        """
        loss_id = tfk.losses.MSE(x_real, mel_output)
        loss_id_psnt = tfk.losses.MSE(x_real, mel_output_postnet)

        codes_reconst = self.encoder(mel_output_postnet, speak_emb)

        loss_cd = tfk.losses.MAE(code_real, codes_reconst)

        return loss_id + loss_id_psnt + self.lamda * loss_cd
