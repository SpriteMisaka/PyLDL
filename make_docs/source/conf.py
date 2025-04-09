# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import inspect

sys.path.insert(0, os.path.abspath('../..'))

project = 'PyLDL'
copyright = '2024, SpriteMisaka'
author = 'SpriteMisaka'
release = '0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_jinja2', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'sphinxcontrib.bibtex']

bibtex_bibfiles = []
for root, dirs, files in os.walk("./"):
    bibtex_bibfiles.extend(
        os.path.join(root, file) for file in files if file.endswith(".bib")
    )
bibtex_default_style = 'unsrtalpha'


from pyldl.algorithms import _ldl__, _le__, _incomldl__, _ldl4c__

import pyldl.algorithms.utils
import pyldl.algorithms.loss_function_engineering
import pyldl.utils
import pyldl.metrics

import pyldl.applications.facial_emotion_recognition as fer
import pyldl.applications.emphasis_selection as es
import pyldl.applications.lesion_counting as lc


def get_module(module, type):
    return [i for i, j in inspect.getmembers(module, type) if j.__module__ == module.__name__]

def get_function(module):
    return get_module(module, inspect.isfunction)

def get_class(module):
    return get_module(module, inspect.isclass)

jinja2_contexts = {
    'ldl_ctx': {
        'all_ldl_algs': _ldl__,
        'all_ldl_loss': get_function(pyldl.algorithms.loss_function_engineering),
    },
    'le_ctx': {'all_le_algs': _le__},
    'incomldl_ctx': {'all_incomldl_algs': _incomldl__},
    'ldl4c_ctx': {'all_ldl4c_algs': _ldl4c__},
    'utils_ctx': {
        'all_alg_utils': get_function(pyldl.algorithms.utils),
        'all_utils': get_function(pyldl.utils),
        'all_metrics': get_function(pyldl.metrics),
    },
    'fer_ctx': {
        'all_fer_cls': get_class(fer),
        'all_fer_func': get_function(fer),
    },
    'es_ctx': {
        'all_es_cls': get_class(es),
        'all_es_func': get_function(es),
    },
    'lc_ctx': {
        'all_lc_cls': get_class(lc),
        'all_lc_func': get_function(lc),
    },
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

autoclass_content = 'both'
autodoc_inherit_docstrings=False

def skip_member(app, what, name, obj, skip, options):
    if not str(getattr(obj, '__module__', None)).startswith('pyldl'):
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_member)
