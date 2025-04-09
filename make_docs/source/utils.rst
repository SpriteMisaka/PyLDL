Utils & Metrics
---------------

.. toctree::
   :maxdepth: 2

.. jinja:: utils_ctx

   {% for func_name in all_alg_utils %}
   {% if func_name == 'sort_loss' %}
   {% else %}
   `{{      func_name      }}`
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   .. autofunction:: pyldl.algorithms.utils.{{ func_name }}
   {% endif %}

   {% endfor %}

   {% for func_name in all_utils %}
   {% if func_name in all_alg_utils %}
   {% else %}
   `{{      func_name      }}`
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   .. autofunction:: pyldl.utils.{{ func_name }}
   {% endif %}

   {% endfor %}

   {% for func_name in all_metrics %}
   {% if func_name in all_alg_utils %}
   {% elif func_name == '_calculate_match_m_top_k' %}
   {% else %}
   `{{      func_name      }}`
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   .. autofunction:: pyldl.metrics.{{ func_name }}
   {% endif %}

   {% endfor %}

References
~~~~~~~~~~

.. bibliography:: bib/utils/references.bib
   :cited:
