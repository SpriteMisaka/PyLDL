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

from pyldl.algorithms.base import BaseDeepLDL


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
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(words)

    sequences = tokenizer.texts_to_sequences(words)
    X = keras.preprocessing.sequence.pad_sequences(
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


class DL_BiLSTM(BaseDeepLDL):

    def __init__(self, tokenizer, embeddings_matrix, n_hidden=256, random_state=None):
        super().__init__(n_hidden, None, random_state)
        self._embeddings_matrix = embeddings_matrix
        self._tokenizer = tokenizer

    @tf.function
    def _loss(self, X, y):
        y_pred = self._model(X)
        y_reshaped = tf.stack((y, 1-y), axis=2)
        return tf.reduce_sum(keras.losses.kl_divergence(y_reshaped, y_pred))

    def fit(self, X, y, learning_rate=1e-3, epochs=3000):
        super().fit(X, y)

        n_embeddings = self._embeddings_matrix.shape[1]

        inputs = keras.Input(shape=(self._n_features,))

        mask = tf.where(inputs > 0, 1, 0)
        mask = tf.expand_dims(mask, axis=2)
        mask = tf.tile(mask, [1, 1, self._n_hidden])

        features = keras.layers.Embedding(
            len(self._tokenizer.word_index)+1, n_embeddings,
            input_length=self._n_features, weights=[self._embeddings_matrix],
            trainable=True)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(
            self._n_hidden//2, return_sequences=True))(features)
        lstm = keras.layers.Dropout(.5)(lstm)

        W = keras.layers.Dense(self._n_hidden, activation='tanh')
        v = keras.layers.Dense(1, activation=None)
        u = v(W(lstm))
        a = keras.activations.softmax(u)
        z = lstm * a

        outputs = keras.layers.Dense(2, activation='softmax')(z)

        self._model = keras.Model(inputs=inputs, outputs=outputs)
        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self._loss(self._X, self._y)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    def predict(self, X):
        return self._model(X)[:, :, 0]
