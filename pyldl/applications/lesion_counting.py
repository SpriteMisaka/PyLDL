import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import keras
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from pyldl.algorithms import BaseDeepLDL


n_grades = 4
n_counts = 65


def load_acne04(path, index=0, mode='train'):
    images_path = os.path.join(path, 'Classification/JPEGImages/')
    filename = f'NNEW_{"trainval" if mode == "train" else "test"}_{index}.txt'
    with open(os.path.join(path, f'Classification/{filename}'), 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    images = []
    grades = []
    counts = []
    for line in lines:
        image_name, grade, count = line.split()
        image_path = os.path.join(images_path, image_name)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        images.append(image)
        grades.append(int(grade))
        counts.append(int(count))
    images = np.array(images)
    grades = np.array(grades)
    counts = np.array(counts)
    return images, grades, counts


def preprocessing(labels, sigma=3.,):
    matrix = np.tile([i for i in range(n_counts + 1)], (len(labels), 1))
    matrix = np.exp(-(matrix - labels.reshape(-1, 1))**2 / (2 * sigma**2))
    return matrix / np.sum(matrix, axis=1).reshape(-1, 1)


def visualization(X, grade, count, grade_real=None, count_real=None,
                  colors=["#00FF00", "#FFFF00", "#FF5500", "#FF0000"],
                  grade_desc=['Mild', 'Moderate', 'Severe', 'Very Severe']):
    _, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].axis('off')
    ax[0].imshow(X / 255.)
    count_range = np.array([i for i in range(n_counts + 1)])
    inter_model_real = interp1d(count_range, count_real, kind='cubic')
    x = np.linspace(0, n_counts, 100)
    y_real = inter_model_real(x)
    grade_label = np.argmax(grade)
    grade_label_real = grade_real
    count_number = int(np.sum(count * (count_range + 1)))
    count_number_real = int(np.sum(count_real * (count_range + 1)))
    ax[1].set_xlim((0, n_counts))
    ax[1].scatter(count_range, count, c=colors[grade_label], s=12,
               label=f'Predection:\n  {grade_desc[grade_label]} ({count_number})')
    ax[1].plot(x, y_real, c=colors[grade_label_real],
               label=f'Ground Truth:\n  {grade_desc[grade_label_real]} ({count_number_real})')
    ax[1].legend()
    ax[1].grid(True, ls='dashed')
    ax[1].get_yaxis().set_visible(False)
    plt.show()


class LDL_ACNE(BaseDeepLDL):
    def __init__(self, random_state=None):
        super().__init__(None, None, random_state)

    havashi = [0] + [0 for _ in range(5)] + [1 for _ in range(15)] + \
        [2 for _ in range(30)] + [3 for _ in range(15)]

    @staticmethod
    @tf.function
    def counts2grades(counts):
        return tf.transpose(tf.math.segment_sum(tf.transpose(counts), LDL_ACNE.havashi))

    @tf.function
    def _loss(self, X, counts, counts2grades):
        features = self._encoder(X)
        pooling = self._pooling(features)
        pred_counts = self._top_counts(pooling)
        pred_grades = self._top_grades(pooling)

        pred_counts2grades = LDL_ACNE.counts2grades(pred_counts)
        
        lc = tf.reduce_sum(keras.losses.kl_divergence(counts, pred_counts))
        lg = tf.reduce_sum(keras.losses.kl_divergence(counts2grades, pred_grades))
        lc2g = tf.reduce_sum(keras.losses.kl_divergence(counts2grades, pred_counts2grades))

        return (1 - self._alpha) * lc + self._alpha / 2 * (lg + lc2g)

    def fit(self, X, counts, learning_rate=1e-4, epochs=500,
            batch_size=32, alpha=.6):
        super().fit(X, counts)

        self._batch_size = batch_size
        self._alpha = alpha

        self._encoder = keras.applications.ResNet50(
            include_top=False, weights='imagenet',
            input_shape=(224, 224, 3))
        self._pooling = keras.layers.GlobalAveragePooling2D()
        self._top_grades = keras.layers.Dense(n_grades, activation='softmax')
        self._top_counts = keras.layers.Dense(n_counts + 1, activation='softmax')

        self._optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        self._counts2grades = LDL_ACNE.counts2grades(self._y)
        data = tf.data.Dataset.from_tensor_slices((self._X, self._y, self._counts2grades)).batch(self._batch_size)

        for _ in range(epochs):
            total_loss = 0.
            for batch in data:
                with tf.GradientTape() as tape:
                    loss = self._loss(batch[0], batch[1], batch[2])
                gradients = tape.gradient(loss, self.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                total_loss += loss

    def predict(self, X):
        features = self._encoder(X)
        pooling = self._pooling(features)
        pred_counts = self._top_counts(pooling)
        pred_grades1 = self._top_grades(pooling)
        pred_grades2 = LDL_ACNE.counts2grades(pred_counts)
        pred_grades = (pred_grades1 + pred_grades2) / 2.
        return pred_grades, pred_counts
