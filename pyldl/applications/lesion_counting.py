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


def preprocessing(labels, sigma=3., n_counts=None):
    if n_counts is None:
        n_counts = np.max(labels)
    matrix = np.tile(list(range(n_counts + 1)), (len(labels), 1))
    matrix = np.exp(-(matrix - labels.reshape(-1, 1))**2 / (2 * sigma**2))
    return matrix / np.sum(matrix, axis=1).reshape(-1, 1)


def visualization(X, grade, count, n_counts,
                  grade_real=None, count_real=None, colors=None, grade_desc=None):
    if colors is None:
        colors = ["#00FF00", "#FFFF00", "#FF5500", "#FF0000"]
    if grade_desc is None:
        grade_desc = ['Mild', 'Moderate', 'Severe', 'Very Severe']
    _, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].axis('off')
    ax[0].imshow(X / 255.)
    count_range = np.array(list(range(n_counts + 1)))
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


class BaseLesionCounting(BaseDeepLDL):

    def score(self, X, target, mode='counting', metrics=None, return_dict=False, **kwargs):
        from pyldl.metrics import score
        if mode == 'counting':
            if metrics is None:
                metrics = ['mean_absolute_error', 'mean_squared_error']
            return score(target, self.predict(X, **kwargs), metrics=metrics, return_dict=return_dict)
        elif mode == 'grading':
            if metrics is None:
                metrics = ["precision", "specificity", "sensitivity", "youden_index", "accuracy"]
            return score(target, self.predict(X, return_grades=True, **kwargs)[1], metrics=metrics, return_dict=return_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class LDL_ACNE(BaseGD, BaseLesionCounting):
    """This approach is proposed in paper :cite:`2019:wu`.
    """

    HAYASHI = [0] * 6 + [1] * 15 + [2] * 30 + [3] * 15

    def __init__(self, n_grades, n_counts, alpha=.6, **kwargs):
        super().__init__(**kwargs)
        self._n_grades = n_grades
        self._n_counts = n_counts
        self._alpha = alpha

    @staticmethod
    @tf.function
    def counts2grades(counts):
        return tf.transpose(tf.math.segment_sum(tf.transpose(counts), LDL_ACNE.HAYASHI))

    def _get_default_model(self):
        inputs = keras.Input(shape=(224, 224, 3))
        features = keras.applications.ResNet50(include_top=False, weights='imagenet')(inputs)
        poolings = keras.layers.GlobalAveragePooling2D()(features)
        counts = keras.layers.Dense(self._n_counts + 1, activation='softmax')(poolings)
        grades = keras.layers.Dense(self._n_grades, activation='softmax')(poolings)
        return keras.Model(inputs=inputs, outputs=[counts, grades])

    @staticmethod
    def loss_function(D, D_pred):
        return tf.reduce_sum(keras.losses.kl_divergence(D, D_pred))

    @tf.function
    def _loss(self, X, D, start, end):
        counts_pred, grades_pred = self._call(X)
        counts2grades_pred = LDL_ACNE.counts2grades(counts_pred)

        lc = self.loss_function(D, counts_pred)
        lg = self.loss_function(self._counts2grades[start:end], grades_pred)
        lc2g = self.loss_function(self._counts2grades[start:end], counts2grades_pred)

        return (1 - self._alpha) * lc + self._alpha / 2 * (lg + lc2g)

    def _before_train(self):
        self._counts2grades = LDL_ACNE.counts2grades(self._D)

    def predict(self, X, batch_size=None, return_grades=False):
        if batch_size is None:
            batch_size = X.shape[0]

        counts_pred = np.zeros((X.shape[0], self._n_counts + 1))
        grades1 = np.zeros((X.shape[0], self._n_grades))
        for i in range(0, X.shape[0], batch_size):
            counts_pred[i:i + batch_size], grades1[i:i + batch_size] = self._call(X[i:i + batch_size])

        if not return_grades:
            return counts_pred
        grades2 = LDL_ACNE.counts2grades(counts_pred)
        grades_pred = (grades1 + grades2) / 2.
        return counts_pred, np.argmax(grades_pred, axis=1)
