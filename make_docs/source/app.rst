Applications
------------

.. toctree::
   :maxdepth: 2

Psychology
~~~~~~~~~~

Facial Emotion Recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. jinja:: fer_ctx

   {% for name in all_fer_cls %}
   {% if name.startswith('Base') %}
   `{{                  name                  }}`
   """"""""""""""""""""""""""""""""""""""""""""""""""
   {% else %}
   {{ name.replace('_', '-').lstrip('-') }}
   """"""""""""""""""""""""""""""""""""""""""""""""""
   {% endif %}

   .. autoclass:: pyldl.applications.facial_emotion_recognition.{{ name }}
      :members:
      :private-members:

   {% endfor %}

   {% for name in all_fer_func %}
   `{{                  name                  }}`
   """"""""""""""""""""""""""""""""""""""""""""""""""

   .. autofunction:: pyldl.applications.facial_emotion_recognition.{{ name }}

   {% endfor %}



References
^^^^^^^^^^

.. bibliography:: bib/app/psy/references.bib
   :labelprefix: APP-
   :cited:

Further Reading
^^^^^^^^^^^^^^^

.. bibliography:: bib/app/psy/references.bib
   :labelprefix: APP-
   :notcited:


Medicine
~~~~~~~~

Lesion Counting
^^^^^^^^^^^^^^^

.. jinja:: lc_ctx

   {% for name in all_lc_cls %}
   {% if name.startswith('Base') %}
   `{{                  name                  }}`
   """"""""""""""""""""""""""""""""""""""""""""""""""
   {% else %}
   {{ name.replace('_', '-').lstrip('-') }}
   """"""""""""""""""""""""""""""""""""""""""""""""""
   {% endif %}

   .. autoclass:: pyldl.applications.lesion_counting.{{ name }}
      :members:
      :private-members:

   {% endfor %}

   {% for name in all_lc_func %}
   `{{                  name                  }}`
   """"""""""""""""""""""""""""""""""""""""""""""""""

   .. autofunction:: pyldl.applications.lesion_counting.{{ name }}

   {% endfor %}


References
^^^^^^^^^^

.. bibliography:: bib/app/med/references.bib
   :labelprefix: APP-
   :cited:


Further Reading
^^^^^^^^^^^^^^^

.. bibliography:: bib/app/med/references.bib
   :labelprefix: APP-
   :notcited:


Text/Natural Language
~~~~~~~~~~~~~~~~~~~~~

Emphasis Selection
^^^^^^^^^^^^^^^^^^

.. jinja:: es_ctx

   {% for name in all_es_cls %}
   {% if name.startswith('Base') %}
   `{{                  name                  }}`
   """"""""""""""""""""""""""""""""""""""""""""""""""
   {% else %}
   {{ name.replace('_', '-').lstrip('-') }}
   """"""""""""""""""""""""""""""""""""""""""""""""""
   {% endif %}

   .. autoclass:: pyldl.applications.emphasis_selection.{{ name }}
      :members:
      :private-members:

   {% endfor %}

   {% for name in all_es_func %}
   `{{                  name                  }}`
   """"""""""""""""""""""""""""""""""""""""""""""""""

   .. autofunction:: pyldl.applications.emphasis_selection.{{ name }}

   {% endfor %}

References
^^^^^^^^^^

.. bibliography:: bib/app/nl/references.bib
   :labelprefix: APP-
   :cited:

Further Reading
^^^^^^^^^^^^^^^

.. bibliography:: bib/app/nl/references.bib
   :labelprefix: APP-
   :notcited:

Computer Vision
~~~~~~~~~~~~~~~

Further Reading
^^^^^^^^^^^^^^^

.. bibliography:: bib/app/cv/references.bib
   :labelprefix: APP-
   :notcited:
