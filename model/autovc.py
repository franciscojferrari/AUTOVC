import tensorflow as tf

from layers import ConvNorm

tfk = tf.keras
tfkl = tfk.layers


class Encoder(tfkl.Layer):
    def __init__(self, dim_emb=256, dim_neck=16, freq=32,*args, ** kwargs):
        super().__init__(*args, **kwargs)
        self.dim_emb = dim_emb
        self.dim_neck = dim_neck
        self.freq = freq

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

        mask_forward = tf.equal(tf.range(tf.shape(output_forward)[1]) % self.freq, 0)
        mask_backward = tf.equal(tf.range(tf.shape(output_forward)[1]-self.freq + 1) % self.freq, 0)
        output_forward_down = tf.boolean_mask(
            output_forward, mask_forward, axis=1)
        output_backward_down = tf.boolean_mask(
            output_backward[:, self.freq-1:, :], mask_backward, axis=1)

        codes = tf.concat([output_forward_down, output_backward_down], axis=-1)
        
        return codes


class UpSampling(tfkl.Layer):
    def __init__(self, *args, dim_neck=16, time_dim=128, dim_emb=256, ** kwargs):
        super().__init__(*args, **kwargs)
        self.dim_neck = dim_neck
        self.time_dim = time_dim
        self.dim_emb = dim_emb

    def build(self, input_shape):
        self.up_sampling = tf.keras.layers.UpSampling1D(
            self.time_dim // self.dim_neck)

    @tf.function
    def call(self, codes, trg_emb):
    
        up_codes = self.up_sampling(codes)
        batch_size = tf.shape(trg_emb)[0]
        trg_emb = tf.broadcast_to(
            tf.expand_dims(trg_emb, axis=1),
            [batch_size, self.time_dim, self.dim_emb])

        output = tf.concat([up_codes, trg_emb], axis=-1)
        return output


class Decoder(tfkl.Layer):
    def __init__(self, *args, dim_pre =512, mel_dim=80,** kwargs):
        super().__init__(*args, **kwargs)
        self.dim_pre = dim_pre
        self.mel_dim = mel_dim

    def build(self, input_shape):
        self.convs = [ConvNorm(name=f"dec_conv_{i}", filters=self.dim_pre,
                               kernel_size=5, strides=1, dilation_rate=1, activation="relu") for i in range(3)]
        self.lstms = [
            tfkl.LSTM(name=f"dec_lstm_{i}", units=1024, return_sequences=True) for i in range(3)]
        self.conv1 = tfkl.Dense(self.mel_dim)

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
    def __init__(self, *args, mel_dim=80, ** kwargs):
        super().__init__(*args, **kwargs)
        self.mel_dim = mel_dim

    def build(self, input):
        self.convs = [ConvNorm(name=f"dec_conv_{i}", filters=512,
                               kernel_size=5, strides=1, dilation_rate=1, activation="tanh") for i in range(5)]

        self.convs.append(ConvNorm(name=f"dec_conv_{5}", filters=self.mel_dim,
                                   kernel_size=5, strides=1, dilation_rate=1, activation=None))

    def call(self, input):
        met_out_psnet = input
        for conv in self.convs:
            met_out_psnet = conv(met_out_psnet)
        return met_out_psnet


class AutoVCF():
    def __init__(self, **kwargs):
        self.time_dim = kwargs.get("time_dim", 128)
        self.dim_emb = kwargs.get("dim_emb", 256)
        self.dim_pre = kwargs.get("dim_pre", 512)
        self.dim_neck = kwargs.get("dim_neck", 16)
        self.mel_dim = kwargs.get("mel_dim", 80)
        self.freq = kwargs.get("freq", 32)
        self.lamda = kwargs.get("lamda", 0.01)
        self.model = self.build_autovc()

    def build_autovc(self):
        mel_spec = tf.keras.layers.Input(shape=(self.time_dim, self.mel_dim))
        speaker_emb = tf.keras.layers.Input(shape=(self.dim_emb))
        trg_speaker_emb = tf.keras.layers.Input(shape=(self.dim_emb))

        encoder = Encoder(dim_emb=self.dim_emb, dim_neck=self.dim_neck,
                          freq=self.freq)

        codes = encoder(mel_spec, speaker_emb)

        upsampled_codes = UpSampling(time_dim=self.time_dim,
                                     freq=self.freq)(codes, trg_speaker_emb)

        mel_decoder = Decoder(dim_pre=self.dim_pre,
                              mel_dim=self.mel_dim)(upsampled_codes)

        mel_postnet = PostNet(mel_dim=self.mel_dim)(mel_decoder)

        recon_codes = encoder(mel_postnet, speaker_emb)

        return tfk.models.Model(inputs=[mel_spec, speaker_emb], outputs=[mel_decoder, mel_postnet, codes, recon_codes], name=self.name)


class AutoVC(tfk.Model):
    def __init__(self, name="AutoVC", *args, **kwargs):
        super(AutoVC, self).__init__(name=name)
        self.time_dim = kwargs.get("time_dim", 128)
        self.dim_emb = kwargs.get("dim_emb", 256)
        self.dim_pre = kwargs.get("dim_pre", 512)
        self.dim_neck = kwargs.get("dim_neck", 16)
        self.mel_dim = kwargs.get("mel_dim", 80)
        self.freq = kwargs.get("freq", 32)
        self.lamda = kwargs.get("lamda", 0.01)

        self.encoder = Encoder(dim_emb=self.dim_emb, 
                               dim_neck=self.dim_neck,
                               freq=self.freq)
        self.decoder = Decoder(dim_pre=self.dim_pre,
                               mel_dim=self.mel_dim)
        self.postnet = PostNet(mel_dim=self.mel_dim)

        # TODO: Check values and throw value exception if are none

    def train_step(self, data):
        
        mel_spec, speaker_emb, trg_speaker_emb = data[0]

        with tf.GradientTape() as tape:
            # Forward pass
            codes = self.encoder(mel_spec, speaker_emb)
            upsampled_codes = self.up_sampling_concat(codes, trg_speaker_emb)
            mel_decoder = self.decoder(upsampled_codes)
            mel_postnet = self.postnet(mel_decoder)
            # Reconstructing Bottlneck
            recon_codes = self.encoder(mel_postnet, speaker_emb)
            recon_codes = tf.concat(recon_codes, axis=-1)
            codes = tf.concat(codes, axis=-1)

            # Compute our own loss
            loss_id = tfk.losses.MSE(mel_spec, mel_decoder)
            loss_id_psnt = tfk.losses.MSE(mel_postnet, mel_postnet)
            loss_cd = tfk.losses.MAE(codes, recon_codes)
            loss_cd = tf.expand_dims(loss_cd, axis=-1)
            loss = loss_id + loss_id_psnt + self.lamda * loss_cd

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss, "loss_id": loss_id, "loss_id_psnt": loss_id_psnt, "loss_cd": loss_cd}

    @tf.function
    def up_sampling_concat(self, codes, trg_emb):
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
