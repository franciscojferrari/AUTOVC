
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense,Lambda,Masking
from tensorflow import keras
from speaker_encoder.speaker_encoder_utils import speaker_centroids,utterance_centroids,similarity_matrix,calculate_loss
import keras.backend as K




class SpeechEmbedder(keras.Model):
    def __init__(self, time_dim=13, melfilters_dim=32):
        super(SpeechEmbedder, self).__init__()
        self.model = Sequential()
        self.model.add(Masking(mask_value=-1.0,
                               input_shape=(time_dim, melfilters_dim)))

        self.model.add(LSTM(768, return_sequences=True,
                            input_shape= (None ,melfilters_dim)))
        self.model.add(LSTM(768))
        # TODO: check activation function
        self.model.add(Dense(256 ,activation='relu'))
        # TODO: check if this L2 normalization is well done
        self.model.add(Lambda(lambda x: K.l2_normalize(x ,axis=1)))

    def call(self, inputs):
        return self.model(inputs)


class GE2ELoss(keras.layers.Layer):
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = tf.Variable(initial_value=10.0, trainable=True)
        self.b = tf.Variable(initial_value=-5.0, trainable=True)

    def call(self, inputs):
        # constrain w > 0, to have larger similarity when cosine similarity is larger.
        tf.clip_by_value(self.w, clip_value_min=1e-6, clip_value_max=np.inf)
        centroids = speaker_centroids(inputs)
        ut_centroids = utterance_centroids(inputs)
        coss_sim = similarity_matrix(inputs, centroids, ut_centroids)
        sim_matrix = self.w * coss_sim + self.b
        loss = calculate_loss(sim_matrix)
        return loss