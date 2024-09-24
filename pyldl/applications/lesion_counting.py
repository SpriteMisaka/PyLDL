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

from pyldl.algorithms.base import BaseGD, BaseDeepLDL


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
               label=f'Prediction:\n  {grade_desc[grade_label]} ({count_number})')
    ax[1].plot(x, y_real, c=colors[grade_label_real],
               label=f'Ground Truth:\n  {grade_desc[grade_label_real]} ({count_number_real})')
    ax[1].legend()
    ax[1].grid(True, ls='dashed')
    ax[1].get_yaxis().set_visible(False)
    plt.show()


class LDL_ACNE(BaseGD, BaseDeepLDL):
    """This approach is proposed in paper :cite:`2019:wu`.
    """

    hayashi = [0] + [0 for _ in range(5)] + [1 for _ in range(15)] + \
        [2 for _ in range(30)] + [3 for _ in range(15)]

    @staticmethod
    @tf.function
    def counts2grades(counts):
        return tf.transpose(tf.math.segment_sum(tf.transpose(counts), LDL_ACNE.hayashi))

    def _get_default_model(self):
        inputs = keras.Input(shape=(224, 224, 3))
        features = keras.applications.ResNet50(include_top=False, weights='imagenet')(inputs)
        poolings = keras.layers.GlobalAveragePooling2D()(features)
        counts = keras.layers.Dense(n_counts + 1, activation='softmax')(poolings)
        grades = keras.layers.Dense(n_grades, activation='softmax')(poolings)
        return keras.Model(inputs=inputs, outputs=[counts, grades])

    @staticmethod
    def loss_function(y, y_pred):
        return tf.reduce_sum(keras.losses.kl_divergence(y, y_pred))

    @tf.function
    def _loss(self, X, y, start, end):
        y_pred, grades_pred = self._call(X)
        counts2grades_pred = LDL_ACNE.counts2grades(y_pred)

        lc = self.loss_function(y, y_pred)
        lg = self.loss_function(self._counts2grades[start:end], grades_pred)
        lc2g = self.loss_function(self._counts2grades[start:end], counts2grades_pred)

        return (1 - self._alpha) * lc + self._alpha / 2 * (lg + lc2g)

    def _before_train(self):
        self._counts2grades = LDL_ACNE.counts2grades(self._y)

    def fit(self, X, y, alpha=.6, **kwargs):
        self._alpha = alpha
        return super().fit(X, y, **kwargs)

    def predict(self, X, batch_size=None, return_grades=False):

        if batch_size is None:
            batch_size = X.shape[0]

        y_pred = np.zeros((X.shape[0], n_counts + 1))
        grades1 = np.zeros((X.shape[0], n_grades))
        for i in range(0, X.shape[0], batch_size):
            y_pred[i:i + batch_size], grades1[i:i + batch_size] = self._call(X[i:i + batch_size])

        if not return_grades:
            return y_pred
        else:
            grades2 = LDL_ACNE.counts2grades(y_pred)
            grades_pred = (grades1 + grades2) / 2.
            return y_pred, grades_pred
