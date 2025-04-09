LDL
---

.. toctree::
   :maxdepth: 2


`BaseLDL`
~~~~~~~~~

.. autoclass:: pyldl.algorithms.base.BaseLDL
   :members:
   :private-members:


.. jinja:: ldl_ctx

   {% for class_name in all_ldl_algs %}
   {% if class_name == 'AdaBoostLDL' %}
   {% else %}
   {% if class_name == 'AA_KNN' %}
   AA
   ~~~~~~~~~~~~~~~

   AA refers to *algorithm adaptation*.

   AA-:math:`k`\NN
   ^^^^^^^^^^^^^^^
   {% elif class_name in ['TLRLDL', 'TKLRLDL', 'SA_BFGS', 'SA_IIS', 'PT_Bayes', 'PT_SVM', 'AA_BP'] %}
   {{ class_name.replace('_', '-').lstrip('-') }}
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   {% else %}
   {{ class_name.replace('_', '-').lstrip('-') }}
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   {% endif %}

   .. autoclass:: pyldl.algorithms.{{ class_name }}
      :members:
      :private-members:

   {% endif %}
   {% endfor %}

   {% for func_name in all_ldl_loss %}
   `{{      func_name      }}`
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   .. autofunction:: pyldl.algorithms.loss_function_engineering.{{ func_name }}
   {% endfor %}

References
~~~~~~~~~~

.. bibliography:: bib/ldl/references.bib
   :labelprefix: LDL-
   :cited:


Further Reading
~~~~~~~~~~~~~~~

.. bibliography:: bib/ldl/references.bib
   :labelprefix: LDL-
   :notcited:
