LE
--

.. toctree::
   :maxdepth: 2

`BaseLE`
~~~~~~~~~

.. autoclass:: pyldl.algorithms.base.BaseLE
   :members:
   :private-members:

.. jinja:: le_ctx

   {% for class_name in all_le_algs %}

   {{ class_name.replace('_', '-').lstrip('-') }}
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: pyldl.algorithms.{{ class_name }}
      :members:
      :private-members:

   {% endfor %}

References
~~~~~~~~~~

.. bibliography:: bib/le/references.bib
   :labelprefix: LE-
   :cited:


Further Reading
~~~~~~~~~~~~~~~

.. bibliography:: bib/le/references.bib
   :labelprefix: LE-
   :notcited:
