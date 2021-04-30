import tensorflow as tf

from layers import ConvNorm

tfk = tf.keras
tfkl = tfk.layers


class Encoder(tfkl.Layer):
    def __init__(self, name = "encoder", time_dim = 128, **kwargs):
        super(Encoder, self).__init__(name = name)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.freq = kwargs.get("freq", 32)

        self.convs = tfk.Sequential([ConvNorm(name = f"enc_conv_{i}", filters = 512,
                                              kernel_size = 5, strides = 1, dilation_rate = 1, activation = "relu") for
                                     i in range(3)])
        self.lstms = tfk.Sequential(
            [tfkl.Bidirectional(tfkl.LSTM(name = f"enc_lstm_{i}", units = self.dim_neck, return_sequences = True))
             for i in range(2)])
        # TODO: Assert values

    def call(self, mel_spec, speak_emb):
        
        batch_size = tf.shape(speak_emb)[0]
        mel_spec = tf.transpose(mel_spec, [0, 2, 1])
        speak_emb = tf.broadcast_to(
            tf.expand_dims(speak_emb, axis = -1), [batch_size, self.dim_emb, mel_spec.shape[-1]])
        input = tf.concat([mel_spec, speak_emb], axis = 1)

        conv_output = self.convs(input)
        conv_output = tf.transpose(conv_output, [0, 2, 1])

        output = self.lstms(conv_output)

        output_forward = output[:, :, :self.dim_neck]
        output_backward = output[:, :, self.dim_neck:]
        # downsampling
        codes = []
        for i in range(0, output.shape[1], self.freq):
            codes.append(tf.concat(
                [output_forward[:, i, :], output_backward[:, i + self.freq - 1, :]], axis = -1))

        return codes


class Decoder(tfkl.Layer):
    def __init__(self, name = "decoder", **kwargs):
        super(Decoder, self).__init__(name = name)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_pre = kwargs.get("dim_pre", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.model = self.build_decoder()

    def build_decoder(self):
        _3convs = [ConvNorm(name = f"dec_conv_{i}", filters = self.dim_pre,
                            kernel_size = 5, strides = 1, dilation_rate = 1, activation = "relu") for i in range(3)]
        _3lstms = [
            tfkl.LSTM(name = f"dec_lstm_{i}", units = 1024, return_sequences = True) for i in range(3)]
        decoder_layers = tfk.Sequential(
            _3convs + _3lstms + [tfkl.Dense(self.mel_feature_dim)])
        return decoder_layers

    def call(self, input):
        # TODO: double check, we may need to transpose if we first start with LSTM
        return self.model(input)


class PostNet(tfkl.Layer):
    def __init__(self, name = "postnet", **kwargs):
        super(PostNet, self).__init__(name = name)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.model = self.build_postnet()

    def build_postnet(self):
        convs = [ConvNorm(name = f"dec_conv_{i}", filters = 512,
                          kernel_size = 5, strides = 1, dilation_rate = 1, activation = "tanh") for i in range(5)]

        convs.append(ConvNorm(name = f"dec_conv_{5}", filters = self.mel_feature_dim,
                              kernel_size = 5, strides = 1, dilation_rate = 1, activation = None))
        posnet_layers = tfk.Sequential(convs)
        return posnet_layers

    def call(self, input):
        return self.model(input)


class AutoVC(tfk.Model):
    def __init__(self, name = "AutoVC", *args, **kwargs):
        super(AutoVC, self).__init__(name = name)
        self.time_dim = kwargs.get("time_dim", None)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_pre = kwargs.get("dim_pre", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.freq = kwargs.get("freq", 32)
        self.lamda = kwargs.get("lamda", 0.01)

        self.encoder = Encoder(time_dim = self.time_dim,
                               dim_emb = self.dim_emb,
                               dim_neck = self.dim_neck, freq = self.freq,
                               mel_dim = self.mel_feature_dim)
        self.decoder = Decoder(dim_emb = self.dim_emb,
                               dim_pre = self.dim_pre, dim_neck = self.dim_neck)
        self.postnet = PostNet(mel_dim = self.mel_feature_dim)

        # TODO: Check values and throw value exception if are none

    def call(self, inputs, *args, **kwargs):
        mel_spec, speak_emb, speak_emb_trg = inputs

        codes = self.encoder(mel_spec, speak_emb)

        tmp = []
        for code in codes:
            
            batch_size = tf.shape(code)[0]
            time_seq = tf.shape(mel_spec)[1]
            down_size = tf.shape(code)[1]

            tmp.append(tf.broadcast_to(
                tf.expand_dims(code, axis = 1), [batch_size, time_seq // len(codes), down_size]))

        codes_exp = tf.concat(tmp, axis = 1)

        speak_emb_trg = tf.broadcast_to(
            tf.expand_dims(speak_emb_trg, axis = 1),
            [tf.shape(speak_emb_trg)[0], mel_spec.shape[1], speak_emb_trg.shape[1]])

        encoder_output = tf.concat([codes_exp, speak_emb_trg], axis = -1)
        mel_output = self.decoder(encoder_output)
        mel_output_postnet = self.postnet(mel_output)
        mel_output_postnet = mel_output + mel_output_postnet

        bottleneck_loss = self.bottleneck_loss(mel_output_postnet, speak_emb, tf.concat(codes, axis = -1))

        self.add_loss(bottleneck_loss)

        return mel_output, mel_output_postnet

    # def custom_loss(self, x_real, speak_emb, mel_output, mel_output_postnet, code_real):
    def bottleneck_loss(self, mel_output_postnet, speak_emb, code_real):

        codes_reconst = self.encoder(mel_output_postnet, speak_emb)
        codes_reconst = tf.concat(codes_reconst, axis = -1)
        loss_cd = tfk.losses.MAE(code_real, codes_reconst)
        loss_cd = tf.expand_dims(loss_cd, axis = -1)
        return self.lamda * loss_cd

    @staticmethod
    def reconstruction_loss(y_true, y_pred):
        print("y_pred",y_pred.shape)
        print("y_true", y_true.shape)

        x_real = y_true[0]
        mel_output = y_pred[0]
        mel_output_postnet= y_pred[1]

        loss_id = tfk.losses.MSE(x_real, mel_output)
        loss_id_psnt = tfk.losses.MSE(x_real, mel_output_postnet)
        return loss_id + loss_id_psnt