
import tensorflow as tf
from speaker_encoder.speaker_encoder_utils import speaker_centroids,utterance_centroids,similarity_matrix,calculate_loss
from keras.layers import LSTM,Dense,Lambda,Masking

tfk = tf.keras
tfkl = tfk.layers


# def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
    
# super(D_VECTOR, self).__init__()
#         self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
#                             num_layers=num_layers, batch_first=True)  
#         self.embedding = nn.Linear(dim_cell, dim_emb)

# def forward(self, x):
#         self.lstm.flatten_parameters()            
#         lstm_out, _ = self.lstm(x)
#         embeds = self.embedding(lstm_out[:,-1,:])
#         norm = embeds.norm(p=2, dim=-1, keepdim=True) 
#         embeds_normalized = embeds.div(norm)
#         return embeds_normalized


class SpeechEmbedder(tfk.Model):
    def __init__(self, dim_input=80, dim_cell=768, dim_emb=256, time_dim=128):
        super(SpeechEmbedder, self).__init__()
        self.masking = tfkl.Masking(mask_value=-1.0,
                               input_shape=(time_dim, dim_input))
        self.lstm1 = tfkl.LSTM(dim_cell, return_sequences=True,
                            input_shape= (None,dim_input), name="lstm1")
        self.lstm2 = tfkl.LSTM(dim_cell, return_sequences=True, name="lstm2")
        self.lstm3 = tfkl.LSTM(dim_cell, return_sequences=True, name="lstm3")
        self.embedding = tfkl.Dense(dim_emb, name="embedding")


    def call(self, inputs):
        masking_out = self.masking(inputs)
        lstm_out = self.lstm1(masking_out)
        lstm_out = self.lstm2(lstm_out)
        lstm_out = self.lstm3(lstm_out)
        embeds = self.embedding(lstm_out[:,-1,:])
        embeds_normalized = tf.math.l2_normalize(embeds, axis=-1)

        return embeds_normalized



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