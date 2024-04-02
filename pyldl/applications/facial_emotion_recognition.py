import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
warnings.filterwarnings("ignore")

import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from pyldl.utils import load_dataset


jaffe_index = np.delete(np.arange(1, 220), np.array([8, 12, 21, 76, 108, 183]) - 1)


def load_jaffe_single(path, i):
    index = jaffe_index[i]
    image_path = os.path.join(path, f'*.{index}.tiff')
    files = glob.glob(image_path)
    if len(files) == 0:
        raise ValueError(f'No image found for index {index} in {path}')
    image = tf.keras.preprocessing.image.load_img(files[0], target_size=(256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    return image


def load_jaffe(path, indices=np.arange(213)):
    images = []
    _, y = load_dataset('SJAFFE')
    for i in indices:
        image = load_jaffe_single(path, i)
        images.append(image)
    return np.array(images), y[indices]


def visualization(image, distribution, distribution_real,
                  labels=['HA', 'SA', 'SU', 'AN', 'DI', 'FE']):
    _, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].axis('off')
    ax[0].imshow(image / 255.)
    label_range = np.array([i for i in range(len(labels))])
    x = np.linspace(0, len(labels)-1, 100)
    inter_model = interp1d(label_range, distribution, kind='cubic')
    y = inter_model(x)
    inter_model_real = interp1d(label_range, distribution_real, kind='cubic')
    y_real = inter_model_real(x)
    ax[1].set_xlim((0, len(labels)-1))
    ax[1].plot(x, y, c="#DE4444", label='Prediction')
    ax[1].plot(x, y_real, c="#3367CD", label='Ground Truth')
    ax[1].legend()
    ax[1].set_xticks(label_range, labels)
    ax[1].grid(True, ls='dashed')
    ax[1].get_yaxis().set_visible(False)
    plt.show()
