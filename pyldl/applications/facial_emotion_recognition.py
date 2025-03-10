import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
warnings.filterwarnings("ignore")

import csv
import subprocess

import glob

import rarfile

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph

import keras
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from pyldl.utils import load_dataset
from pyldl.algorithms.base import BaseGD, BaseDeepLDLClassifier


JAFFE_INDEX = np.delete(np.arange(1, 220), np.array([8, 12, 21, 76, 108, 183]) - 1)


def load_jaffe_single(path, i, size=(256, 256)):
    index = JAFFE_INDEX[i]
    image_path = os.path.join(path, f'*.{index}.tiff')
    files = glob.glob(image_path)
    if not files:
        raise ValueError(f'No image found for index {index} in {path}')
    image = keras.preprocessing.image.load_img(files[0])
    image = tf.image.resize(image, size)
    image = keras.preprocessing.image.img_to_array(image)
    return image


def load_jaffe(path, indices=None, size=(256, 256)):
    if indices is None:
        indices = np.arange(213)
    images = []
    _, y = load_dataset('SJAFFE')
    for i in indices:
        image = load_jaffe_single(path, i, size)
        images.append(image)
    return np.array(images), y[indices]


def load_bu_3dfe(path, size=(256, 256)):
    names = [f"F{i:04d}" for i in range(1, 57)] + [f"M{i:04d}" for i in range(1, 45)]
    for name in names:
        if not os.path.exists(os.path.join(path, name)):
            with rarfile.RarFile(os.path.join(path, f'{name}.rar')) as file:
                for i in file.namelist():
                    if i.endswith('_F2D.bmp'):
                        file.extract(i, path)
    images = []
    for subdir, _, files in sorted(os.walk(path)):
        files = sorted(files)
        for f in files:
            if f.endswith('_F2D.bmp'):
                image = keras.preprocessing.image.load_img(os.path.join(subdir, f))
                image = tf.image.resize(image, size)
                image = keras.preprocessing.image.img_to_array(image)
                images.append(image)
    _, y = load_dataset('SBU_3DFE')
    return np.array(images), y


def extract_ck_plus(input_dir, output_dir, openface_path, basic=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    emotions = ['happiness', 'sadness', 'surprise', 'anger', 'disgust', 'fear']
    if not basic:
        emotions += ['neutral', 'contempt']
    for emotion in emotions:
        emotion_path = os.path.join(input_dir, emotion)
        command = [
            os.path.abspath(os.path.join(openface_path, 'FaceLandmarkImg.exe')),
            '-fdir', os.path.abspath(emotion_path),
            '-out_dir', os.path.abspath(output_dir),
            '-2Dfp',
            '-aus'
        ]
        subprocess.run(command, check=True)


def load_ck_plus(image_dir, feature_dir=None, size=(196, 256), basic=True):
    images = []
    labels = []
    mapping = {'happiness': 0, 'sadness': 1, 'surprise': 2, 'anger': 3, 'disgust': 4, 'fear': 5}
    if feature_dir is not None:
        fps, aus = [], []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if basic and os.path.split(root)[-1] in ['neutral', 'contempt']:
                continue
            if file.lower().endswith('.png'):
                image_path = os.path.join(root, file)
                image = keras.preprocessing.image.load_img(image_path)
                image = tf.image.resize(image, size)
                image = keras.preprocessing.image.img_to_array(image)
                images.append(image)
                labels.append(mapping[os.path.split(root)[-1]])
                if feature_dir is not None:
                    with open(os.path.join(feature_dir, file.replace('.png', '.csv')), 'r') as f:
                        my_reader = csv.reader(f, delimiter=',')
                        features = list(my_reader)[1]
                        fps.append([float(i) for i in features[2:138]])
                        aus.append([float(i) for i in features[138:]])

    images = np.array(images)
    labels = np.array(labels)
    if feature_dir is not None:
        return images, labels, np.array(fps), np.array(aus)
    else:
        return images, labels


def visualization(image, distribution, real, style_real='distribution', labels=None):
    if labels is None:
        labels = ['HA', 'SA', 'SU', 'AN', 'DI', 'FE']
    _, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].axis('off')
    ax[0].imshow(image / 255.)

    label_range = np.array(list(range(len(labels))))
    x = np.linspace(0, len(labels)-1, 100)
    inter_model = interp1d(label_range, distribution, kind='cubic')
    y = inter_model(x)
    ax[1].plot(x, y, c="#DE4444", label='Prediction')

    if style_real == 'distribution':
        inter_model_real = interp1d(label_range, real, kind='cubic')
        y_real = inter_model_real(x)
        ax[1].plot(x, y_real, c="#3367CD", label='Ground Truth')
    elif style_real == 'binary':
        y_min, _ = ax[1].get_ylim()
        ax[1].bar(label_range, real, width=.2, bottom=y_min,
                  color="#3367CD", label='Ground Truth')

    ax[1].legend()
    ax[1].set_xlim((0, len(labels)-1))
    ax[1].set_xticks(label_range, labels)
    ax[1].grid(True, ls='dashed')
    ax[1].get_yaxis().set_visible(False)
    plt.show()


class BaseFacialEmotionRecognition(BaseDeepLDLClassifier):

    def score(self, X, D, metrics=None, return_dict=False):
        if metrics is None:
            metrics = ['accuracy']
        from pyldl.metrics import score
        return score(np.argmax(D, axis=1), np.argmax(self.predict_proba(X), axis=1), metrics, return_dict)


class LDL_ALSG(BaseGD, BaseFacialEmotionRecognition):
    """:class:`LDL-ALSG <pyldl.algorithms.facial_emotion_recognition.LDL_ALSG>` is proposed in :cite:`2020:chen`.
    """

    def _get_default_model(self):
        inputs = keras.Input(shape=self._X.shape[1:])
        features = keras.applications.ResNet50(
            include_top=False, weights='imagenet')(inputs)
        poolings = keras.layers.GlobalAveragePooling2D()(features)
        outputs = keras.layers.Dense(self._n_outputs, activation='softmax')(poolings)
        return keras.Model(inputs=inputs, outputs=outputs)

    def _generate_graphs(self, features, sigma):
        graphs = []
        for i in range(int(np.ceil(self._n_samples / self._batch_size))):
            start = i * self._batch_size
            end = min(start + self._batch_size, self._n_samples)
            graph = kneighbors_graph(features[start:end], n_neighbors=5, include_self=False)
            a = np.exp(-(cdist(
                self._D[start:end], self._D[start:end]
            ) ** 2) / (2 * sigma ** 2))
            graph = a * graph.toarray()
            graphs.append(graph)
        return graphs

    def _before_train(self):
        self._fp_graphs = self._generate_graphs(self._fps, self._fp_sigma)
        self._au_graphs = self._generate_graphs(self._aus, self._au_sigma)

    @staticmethod
    def loss_function(L, D_pred):
        return tf.reduce_sum(keras.losses.categorical_crossentropy(L, D_pred))

    def _aux_loss(self, graph, D_pred):
        indices = tf.where(graph > 0.)
        return tf.reduce_sum(keras.losses.kl_divergence(tf.gather(D_pred, indices[:, 0]),
                                                        tf.gather(D_pred, indices[:, 1])))

    def _loss(self, X, L, start, _):
        D_pred = self._call(X)
        ce = self.loss_function(L, D_pred)
        i = start // self._batch_size
        fp = self._aux_loss(self._fp_graphs[i], D_pred)
        au = self._aux_loss(self._au_graphs[i], D_pred)
        return ce + self._alpha * (fp + au)

    def fit(self, X, L, fps, aus, alpha=5e-4,
            fp_sigma=68., au_sigma=1., batch_size=None, L_val=None, **kwargs):
        self._fps = tf.cast(fps, tf.float32)
        self._aus = tf.cast(aus, tf.float32)
        self._alpha = alpha
        self._fp_sigma = fp_sigma
        self._au_sigma = au_sigma
        self._batch_size = batch_size
        return super().fit(X, L, batch_size=self._batch_size, D_val=L_val, **kwargs)
