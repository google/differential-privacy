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

#####################
Mechanism Calibration
#####################

.. currentmodule:: dp_accounting.mechanism_calibration

A `PrivacyAccountant` and a `DpEvent` can be used to compute the privacy
parameters of a mechanism, given the mechanism's parameters. This module
reverses that process: it offers functions to compute the minimal/maximal
parameter of a mechanism that is needed to achieve a given level of privacy.

As a simple example, this snippet calibrates the noise multiplier for a
Gaussian mechanism to achieve `(1, 1e-5)`-DP:

.. code-block:: python

   noise_multiplier = dp_accounting.calibrate_dp_mechanism(
       make_fresh_accountant=dp_accounting.pld.PLDAccountant,
       make_event_from_param=dp_accounting.GaussianDpEvent,
       target_epsilon=1.0,
       target_delta=1e-5,
   )

Functions
---------

.. autofunction:: calibrate_dp_mechanism

Classes
-------

.. autoclass:: BracketInterval
   :members:
   :undoc-members:

.. autoclass:: ExplicitBracketInterval
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LowerEndpointAndGuess
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. autoexception:: NoBracketIntervalFoundError
   :members:

.. autoexception:: NonEmptyAccountantError
   :members:
