import tensorflow as tf
import numpy as np
import random

def speaker_centroids(embeddings):
    """
    Inputs:
        embeddings: Embeddings from encoder, shape=(speakers_per_batch, utterances_per_speaker, embedding_size)

    Returns
        Speaker centroids of shape=(speakers_per_batch, 1, embedding_size).
    """
    speaker_centroids = tf.math.reduce_mean(embeddings, axis=1, keepdims=True)
    speaker_centroids = tf.identity(speaker_centroids) / (
        tf.norm(speaker_centroids, axis=2, keepdims=True) + 1e-6
    )

    return speaker_centroids


def utterance_centroids(embeddings):
    """
    Inputs:
        embeddings: Embeddings from encoder, shape=(speakers_per_batch, utterances_per_speaker, embedding_size)

    Returns
        Utterance centroids of shape=(speakers_per_batch, 1, embedding_size).
    """
    utterances_per_speaker = embeddings.shape[1]

    utterance_centroids = (
        tf.math.reduce_sum(embeddings, axis=1, keepdims=True) - embeddings
    )
    utterance_centroids /= utterances_per_speaker - 1
    utterance_centroids = tf.identity(utterance_centroids) / (
        tf.norm(utterance_centroids, axis=2, keepdims=True) + 1e-6
    )

    return utterance_centroids

def similarity_matrix(embeddings, speaker_centroids, utterance_centroids):
    """
    Inputs:
        embeddings: Embeddings from encoder, shape=(speakers_per_batch, utterances_per_speaker, embedding_size)
        speaker_centroids: Speaker centroids of shape=(speakers_per_batch, 1, embedding_size).
        utterance_centroids: Utterance centroids of shape=(speakers_per_batch, 1, embedding_size).

    Returns
        Similarity matrix of shape=(speakers_per_batch, utterances_per_speaker, speakers_per_batch).
    """
    speakers_per_batch = embeddings.shape[0]
    mask_matrix = 1 - tf.eye(speakers_per_batch)
    sim_values = []

    for j in range(speakers_per_batch):
        mask = tf.transpose(tf.where(mask_matrix[j]))[0]
        a = tf.reduce_sum(tf.gather(embeddings, mask) * speaker_centroids[j], axis=2)
        b = tf.reshape(
            tf.reduce_sum(embeddings[j] * utterance_centroids[j], axis=1), shape=(1, -1)
        )

        # Make sure that b is inserted in the right place.
        a = tf.unstack(a, axis=0)
        b = tf.unstack(b, axis=0)
        a.insert(j, b[0])
        c = tf.stack(a, axis=-1)

        sim_values.append(c)

    sim_values = [
        tf.expand_dims(tf.transpose(m), axis=-1) for m in sim_values
    ]  # Add additional dimension
    sim_matrix = tf.concat(sim_values, axis=2)

    return sim_matrix

def calculate_loss(sim_matrix):
    same_idx = list(range(sim_matrix.shape[0]))
    pos = tf.stack([sim_matrix[i,:,i] for i in same_idx], axis=0)
    in_neg = tf.math.exp(sim_matrix)
    neg = tf.math.log(tf.math.reduce_sum(in_neg,axis=2)+ 1e-6)
    per_embedding_loss = -1 * (pos - neg)
    loss = tf.reduce_sum(per_embedding_loss)
    #sim_matrix = sim_matrix.numpy()
    #pos = sim_matrix[same_idx, :, same_idx]
    #in_neg = (np.exp(sim_matrix))
    #neg = np.log(np.sum(in_neg,axis=2)+ 1e-6)
    #per_embedding_loss = -1 * (pos - neg)
    #loss = per_embedding_loss.sum()
    return loss

def parse_spectrograms(example):
    """Convert the serialized tensor back to a tensor."""
    example = tf.io.parse_tensor(
        example.numpy()[0], out_type=tf.float32
    )
    return example

def create_batches(datasets, number_speakers, number_utterances):
  list_speakers = random.sample(datasets.keys(),number_speakers)
  batch = []
  for speaker in list_speakers:
      print(speaker)
      list_utterances = datasets[speaker].shuffle(buffer_size=100).batch(number_utterances)
      batch_speaker = next(iter(list_utterances))
      for i in batch_speaker["mel_spectrogram"]:
        spectrogram = parse_spectrograms(i)
        batch.append(spectrogram)
  batch = tf.ragged.stack(batch, axis=0)
  return batch