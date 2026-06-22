.. Copyright 2026 Google LLC.
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :special-members: __call__

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      {% for item in methods %}
         ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:
      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}
