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

##################
Privacy Accountant
##################

.. currentmodule:: dp_accounting.privacy_accountant

A `PrivacyAccountant` allows for converting a `DpEvent` or sequence of
`DpEvent`\ s into privacy parameters. One can call `accountant.compose` on each
`DpEvent` we wish to compose, and then use methods like `get_epsilon` and
`get_delta` to get the privacy parameters for the mechanism captured by all
`DpEvent`\ s composed so far.

`dp_accounting` currently supports two types of `PrivacyAccountant`\ s.
:class:`~dp_accounting.pld.PLDAccountant` uses privacy loss distribution (PLD)
accounting, which is generally
tight up to some numerical precision, but can be slow (see the
`supplementary material <https://github.com/google/differential-privacy/blob/main/common_docs/Privacy_Loss_Distributions.pdf>`_
for technical details on PLD accounting). :class:`~dp_accounting.rdp.RdpAccountant` uses
Renyi DP (RDP) accounting, which is faster but not tight in general. Note that not all
`DpEvent`\ s are supported by both accountants, and there may be `DpEvent`\ s
supported by one accountant but not the other.

When initializing a `PrivacyAccountant`, one also specifies the
`NeighboringRelation` which defines how neighboring databases are related.
The default `NeighboringRelation` for both accountants is `ADD_OR_REMOVE_ONE`,
which corresponds to the standard add-or-remove-one definition of DP.

Both accountants have a parameter which allows one to trade off accuracy for
speed. For `RdpAccountant`, this is `orders`, the set of RDP orders which are
tracked and used to compute `epsilon`. For `PLDAccountant`, this is
`value_discretization_interval`, the precision of the discretization of
continuous random variables used in PLD accounting. The default values for these
parameters are fine for most use cases, but one can consider using a denser grid
of `orders` or smaller `value_discretization_interval` if higher accuracy is
needed, or a sparser grid or larger interval if speed is more important.

Classes
-------

.. autoclass:: PrivacyAccountant
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: NeighboringRelation
   :members:
   :undoc-members:

.. autoclass:: dp_accounting.rdp.RdpAccountant
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dp_accounting.pld.PLDAccountant
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. autoexception:: UnsupportedEventError
   :members:
