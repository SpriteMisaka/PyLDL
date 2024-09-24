import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import keras
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from IPython.display import HTML, display

from pyldl.algorithms.base import BaseAdam, BaseDeepLDL


def load_semeval2020(path):
    filename = os.path.join(path, 'train_dev_data/train.txt')
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    words, all_words = [], []
    freqs, all_freqs = [], []

    for line in lines:
        if line:
            splitted = line.split("\t")
            words.append(splitted[1])
            freqs.append(splitted[4])
        elif words:
            all_words.append(words)
            all_freqs.append(freqs)
            words = []
            freqs = []

    return all_words, all_freqs


def preprocessing(words, freqs, tokenizer=None, maxlen=None):
    new_tokenizer = False
    if tokenizer is None and maxlen is None:
        new_tokenizer = True
        maxlen = max(map(len, words))
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(words)

    sequences = tokenizer.texts_to_sequences(words)
    X = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=maxlen, padding='post'
    )
    y = np.array([row + [0] * (maxlen - len(row))
                  if len(row) < maxlen else row[:maxlen]
                  for row in freqs], dtype=np.float32)

    if new_tokenizer:
        return X, y, tokenizer, maxlen
    return X, y


def visualization(words, y=None, threshold=.2, r=255, g=68, b=68):
    color = f"{r}, {g}, {b}"
    sentence = ""
    for i in range(len(words)):
        for j in range(len(words[i])):
            if y is None:
                intensity = 0.
            else:
                intensity = y[i, j] if y[i, j] > threshold else 0.
            if j != 0 and not words[i][j].startswith((',', '.', '?', '!', 'n\'t', '\'s', ';')):
                sentence += ' '
            sentence += f"<span style='background-color:rgba({color}, {intensity})'>{words[i][j]}</span>"
        if i < len(words) - 1:
            sentence += '<br>'
    display(HTML(sentence))
            

def load_glove(path, tokenizer, embedding_dim=100):
    embeddings_index = {}
    with open(os.path.join(path, f'glove.6B.{embedding_dim}d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((len(tokenizer.word_index)+1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix


class DL_BiLSTM(BaseAdam, BaseDeepLDL):
    """This approach is proposed in paper :cite:`2019:shirani`.
    """

    def __init__(self, tokenizer, embeddings_matrix, n_hidden=512, random_state=None):
        super().__init__(n_hidden, None, random_state)
        self._embeddings_matrix = embeddings_matrix
        self._tokenizer = tokenizer

    @staticmethod
    def loss_function(y, y_pred):
        return tf.reduce_sum(keras.losses.kl_divergence(y, y_pred))

    @tf.function
    def _loss(self, X, y, start, end):
        y_pred = self._call(X)
        y_reshaped = tf.stack((y, 1-y), axis=2)
        return self.loss_function(self._mask[start:end] * y_reshaped,
                                  self._mask[start:end] * y_pred)

    def _create_mask(self, X):
        mask = tf.cast(tf.greater(X, 0), dtype=tf.float32)
        return tf.tile(tf.expand_dims(mask, axis=2), [1, 1, 2])

    def _before_train(self):
        self._mask = self._create_mask(self._X)

    def _get_default_model(self):
        n_embeddings = self._embeddings_matrix.shape[1]

        inputs = keras.Input(shape=(self._n_features,))
        features = keras.layers.Embedding(
            len(self._tokenizer.word_index)+1, n_embeddings,
            input_shape=(self._n_features, ), trainable=True)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(
            self._n_hidden//2, return_sequences=True))(features)
        lstm = keras.layers.Dropout(.5)(lstm)

        W = keras.layers.Dense(self._n_hidden, activation='tanh')
        v = keras.layers.Dense(1, activation=None)
        u = v(W(lstm))
        a = keras.activations.softmax(u)
        z = lstm * a

        outputs = keras.layers.Dense(2, activation='softmax')(z)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.layers[1].set_weights([self._embeddings_matrix])
        return model

    def predict(self, X):
        mask = self._create_mask(X)
        return (mask * self._call(X))[:, :, 0].numpy()
