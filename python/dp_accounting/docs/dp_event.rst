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

########
DpEvent
########

.. currentmodule:: dp_accounting.dp_event

A ``DpEvent`` is a representation of the application of a differentially private
mechanism. Just as a differentially private mechanism can be build from simpler
building blocks, a ``DpEvent`` representing such a mechanism can be built from
simpler ``DpEvent``\ s.

Base Event Class
================

.. autoclass:: DpEvent
   :members:
   :undoc-members:
   :show-inheritance:

Self-Contained Events
=====================

These represent mechanisms that can be fully described by the parameters of the
class, without relying on another ``DpEvent``. These typically correspond to the
"basic building blocks" of differentially private mechanisms, such as the
Gaussian, Laplace, and Exponential mechanisms.

.. autoclass:: GaussianDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DiscreteGaussianDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TruncatedSubsampledGaussianDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LaplaceDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DiscreteLaplaceDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MixtureOfGaussiansDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ExponentialMechanismDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PermuteAndFlipDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RandomizedResponseDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EpsilonDeltaDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ZCDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SingleEpochTreeAggregationDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

Events Wrapping Other Events
============================

These represent mechanisms that are defined in terms of other ``DpEvent``\ s. These
can be used to describe more complex mechanisms in terms of simpler ones. For
example, DP-SGD with Poisson sampling can be represented as a
``SelfComposedDpEvent`` (corresponding to the composition of many iterations) of
a ``PoissonSampledDpEvent`` containing a ``GaussianDpEvent`` (corresponding to
the subsampled Gaussian mechanism used in a single iteration).

.. autoclass:: PoissonSampledDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SampledWithReplacementDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SampledWithoutReplacementDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ComposedDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SelfComposedDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RepeatAndSelectDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

Non-Standard Events
===================

These represent extreme mechanisms, or unsupported mechanisms, provided mostly
for completeness of the API or for testing purposes. Most users will not need to
interact with these ``DpEvent``\ s.

.. autoclass:: NoOpDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NonPrivateDpEvent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: UnsupportedDpEvent
   :members:
   :undoc-members:
   :show-inheritance:
