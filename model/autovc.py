import tensorflow as tf

from layers import ConvNorm

tfk = tf.keras
tfkl = tfk.layers


class Encoder(tfkl.Layer):
    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.dim_emb = kwargs.get("dim_emb", 256)
        self.dim_neck = kwargs.get("dim_neck", 16)
        self.freq = kwargs.get("freq", 32)

    def build(self, input_shape):
        self.convs = [ConvNorm(name=f"enc_conv_{i}", filters=512,
                               kernel_size=5, strides=1, dilation_rate=1, activation="relu") for
                      i in range(3)]
        self.lstms = [tfkl.Bidirectional(tfkl.LSTM(name=f"enc_lstm_{i}", units=self.dim_neck, return_sequences=True))
                      for i in range(2)]

    def call(self, mel_spec, speak_emb):

        batch_size = tf.shape(speak_emb)[0]
        mel_spec = tf.transpose(mel_spec, [0, 2, 1])
        speak_emb = tf.broadcast_to(
            tf.expand_dims(speak_emb, axis=-1), [batch_size, self.dim_emb, mel_spec.shape[-1]])
        input = tf.concat([mel_spec, speak_emb], axis=1)

        conv_output = input
        for conv in self.convs:
            conv_output = conv(conv_output)

        conv_output = tf.transpose(conv_output, [0, 2, 1])

        output = conv_output
        for lstm in self.lstms:
            output = lstm(output)

        output_forward = output[:, :, :self.dim_neck]
        output_backward = output[:, :, self.dim_neck:]

        # downsampling
        codes = []
        for i in range(0, output.shape[1], self.freq):
            codes.append(tf.concat(
                [output_forward[:, i, :], output_backward[:, i + self.freq - 1, :]], axis=-1))

        return tf.convert_to_tensor(codes)


class UpSampling(tfkl.Layer):
    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
        self.freq = kwargs.get("freq", 32)
        self.time_dim = kwargs.get("time_dim", 128)

    def build(self, input_shape):
        self.upsampling = tfkl.UpSampling1D(size=self.freq)

    def call(self, codes, trg_emb):
        up_codes = self.upsampling(codes)

        tmp = []
        for code in codes:

            batch_size = tf.shape(code)[0]
            down_size = tf.shape(code)[1]

            tmp.append(tf.broadcast_to(
                tf.expand_dims(code, axis=1), [batch_size, self.time_dim // len(codes), down_size]))

        up_codes = tf.concat(tmp, axis=1)

        trg_emb = tf.broadcast_to(
            tf.expand_dims(trg_emb, axis=1),
            [tf.shape(trg_emb)[0], self.time_dim, trg_emb.shape[1]])

        output = tf.concat([up_codes, trg_emb], axis=-1)
        return output


class Decoder(tfkl.Layer):
    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
        self.dim_emb = kwargs.get("dim_emb", 256)
        self.dim_pre = kwargs.get("dim_pre", 512)
        self.dim_neck = kwargs.get("dim_neck", 16)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)

    def build(self, input_shape):
        self.convs = [ConvNorm(name=f"dec_conv_{i}", filters=self.dim_pre,
                            kernel_size=5, strides=1, dilation_rate=1, activation="relu") for i in range(3)]
        self.lstms = [
            tfkl.LSTM(name=f"dec_lstm_{i}", units=1024, return_sequences=True) for i in range(3)]
        self.conv1 = tfkl.Dense(self.mel_feature_dim)


    def call(self, codes):
        recon_conv = codes
        for conv in self.convs:
            recon_conv = conv(recon_conv)

        recon_lstm = recon_conv
        for lstm in self.lstms:
            recon_lstm = lstm(recon_lstm)

        recon = self.conv1(recon_lstm)
        return recon
        

class PostNet(tfkl.Layer):
    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)

    def build(self, input):
        self.convs = [ConvNorm(name=f"dec_conv_{i}", filters=512,
                          kernel_size=5, strides=1, dilation_rate=1, activation="tanh") for i in range(5)]

        self.convs.append(ConvNorm(name=f"dec_conv_{5}", filters=self.mel_feature_dim,
                              kernel_size=5, strides=1, dilation_rate=1, activation=None))
        
    def call(self, input):
        met_out_psnet = input
        for conv in self.convs:
            met_out_psnet = conv(met_out_psnet)
        return met_out_psnet


# def build_autovc(*args, **kwargs):
#     time_dim = kwargs.get("time_dim", 128)
#     dim_emb = kwargs.get("dim_emb", 256)
#     dim_pre = kwargs.get("dim_pre", 512)
#     dim_neck = kwargs.get("dim_neck", 16)
#     mel_dim = kwargs.get("mel_dim", 80)
#     freq = kwargs.get("freq", 32)
#     lamda = kwargs.get("lamda", 0.01)

#     encoder = Encoder()
#     decoder = Decoder()




class AutoVC(tfk.Model):
    def __init__(self, name="AutoVC", *args, **kwargs):
        super(AutoVC, self).__init__(name=name)
        self.time_dim = kwargs.get("time_dim", None)
        self.dim_emb = kwargs.get("dim_emb", None)
        self.dim_pre = kwargs.get("dim_pre", None)
        self.dim_neck = kwargs.get("dim_neck", None)
        self.mel_feature_dim = kwargs.get("mel_dim", 80)
        self.freq = kwargs.get("freq", 32)
        self.lamda = kwargs.get("lamda", 0.01)

        self.encoder = Encoder(time_dim=self.time_dim,
                               dim_emb=self.dim_emb,
                               dim_neck=self.dim_neck, freq=self.freq,
                               mel_dim=self.mel_feature_dim)
        self.decoder = Decoder(dim_emb=self.dim_emb,
                               dim_pre=self.dim_pre, dim_neck=self.dim_neck)
        self.postnet = PostNet(mel_dim=self.mel_feature_dim)

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
                tf.expand_dims(code, axis=1), [batch_size, time_seq // len(codes), down_size]))

        codes_exp = tf.concat(tmp, axis=1)

        speak_emb_trg = tf.broadcast_to(
            tf.expand_dims(speak_emb_trg, axis=1),
            [tf.shape(speak_emb_trg)[0], mel_spec.shape[1], speak_emb_trg.shape[1]])

        encoder_output = tf.concat([codes_exp, speak_emb_trg], axis=-1)
        mel_output = self.decoder(encoder_output)
        mel_output_postnet = self.postnet(mel_output)
        mel_output_postnet = mel_output + mel_output_postnet

        bottleneck_loss = self.bottleneck_loss(
            mel_output_postnet, speak_emb, tf.concat(codes, axis=-1))

        self.add_loss(bottleneck_loss)

        return mel_output, mel_output_postnet

    # def custom_loss(self, x_real, speak_emb, mel_output, mel_output_postnet, code_real):
    def bottleneck_loss(self, mel_output_postnet, speak_emb, code_real):

        codes_reconst = self.encoder(mel_output_postnet, speak_emb)
        codes_reconst = tf.concat(codes_reconst, axis=-1)
        loss_cd = tfk.losses.MAE(code_real, codes_reconst)
        loss_cd = tf.expand_dims(loss_cd, axis=-1)
        return self.lamda * loss_cd

    @staticmethod
    def reconstruction_loss(y_true, y_pred):
        print("y_pred", y_pred.shape)
        print("y_true", y_true.shape)

        x_real = y_true[0]
        mel_output = y_pred[0]
        mel_output_postnet = y_pred[1]

        loss_id = tfk.losses.MSE(x_real, mel_output)
        loss_id_psnt = tfk.losses.MSE(x_real, mel_output_postnet)
        return loss_id + loss_id_psnt
