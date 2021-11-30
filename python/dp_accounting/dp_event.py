# Copyright 2021, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standard DpEvent classes.

A `DpEvent` represents the (hyper)parameters of a differentially
private query, amplification mechanism, or composition, that are necessary
and sufficient for privacy accounting. Various independent implementations of DP
algorithms that are functionally equivalent from an accounting perspective may
correspond to the same `DpEvent`. Similarly, various independent implementations
of accounting algorithms may consume the same `DpEvent`.

All `DpEvents` processed together are assumed to take place on a single dataset
of records. `DpEvents` fall into roughly three categories:
 - `DpEvents` that release an output, and incur a privacy cost,
    e.g., `GaussianDpEvent`.
 - `DpEvents` that select a subset (or subsets) of the dataset, and run nested
    `DpEvents` on those subsets, e.g., `PoissonSampledDpEvent`.
 - `DpEvents` that represent (possibly sequentially) applying (multiple)
   mechanisms to the dataset (or currently active subset). Currently, this is
   only `ComposedDpEvent` and `SelfComposedDpEvent`.

Each `DpEvent` should completely document the mathematical behavior and
assumptions of the mechanism it represents so that the writer of an accountant
class can implement the accounting correctly without knowing any other
implementation details of the algorithm that produced it.

New mechanism types should be given a corresponding `DpEvent` class, although
not all accountants will be required to support them. In general,
`PrivacyAccountant` implementations are not required to be aware of all
`DpEvent` classes, but they should support the following basic events and handle
them appropriately: `NoOpDpEvent`, `NonPrivateDpEvent`, `ComposedDpEvent`, and
`SelfComposedDpEvent`. They should return `supports(event)` is False for
`UnsupportedDpEvent` or any other event type they have not been designed to
handle.

To ensure that a `PrivacyAccountant` does not accidentally start to return
incorrect results, the following should be enforced:
 * `DpEvent` classes and their parameters should never be removed, barring some
   extended, onerous deprecation process.
 * New parameters cannot be added to existing mechanisms unless they are
   optional. That is, old composed `DpEvent` objects that do not include them
   must remain valid.
 * The meaning of existing mechanisms or parameters must not change. That is,
   existing mechanisms should not have their implementations change in ways that
   alter their privacy properties; new `DpEvent` classes should be added
   instead.
 * `PrivacyAccountant` implementations are expected to return `supports(event)`
   is `False` when processing unknown mechanisms.
"""

from typing import List

import attr


class DpEvent(object):
  """Represents application of a private mechanism.

  A `DpEvent` describes a differentially private mechanism sufficiently for
  computing the associated privacy losses, both in isolation and in combination
  with other `DpEvent`s.
  """


@attr.s(frozen=True)
class NoOpDpEvent(DpEvent):
  """Represents appplication of an operation with no privacy impact.

  A `NoOpDpEvent` is generally never required, but it can be useful as a
  placeholder where a `DpEvent` is expected, such as in tests or some live
  accounting pipelines.
  """


@attr.s(frozen=True)
class NonPrivateDpEvent(DpEvent):
  """Represents application of a non-private operation.

  This `DpEvent` should be used when an operation is performed that does not
  satisfy (epsilon, delta)-DP. All `PrivacyAccountant`s should return infinite
  epsilon/delta when encountering a `NonPrivateDpEvent`.
  """


@attr.s(frozen=True)
class UnsupportedDpEvent(DpEvent):
  """Represents application of an as-yet unsupported operation.

  This `DpEvent` should be used when an operation is performed that does not yet
  have any associated DP description, or if the description is temporarily
  inaccessible, for example, during development. All `PrivacyAccountant`s should
  return `supports(event) == False` for `UnsupportedDpEvent`.
  """


@attr.s(frozen=True, slots=True, auto_attribs=True)
class GaussianDpEvent(DpEvent):
  """Represents an application of the Gaussian mechanism.

  For values v_i and noise z ~ N(0, s^2I), this mechanism returns sum_i v_i + z.
  If the norms of the values are bounded ||v_i|| <= C, the noise_multiplier is
  defined as s / C.
  """
  noise_multiplier: float


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SelfComposedDpEvent(DpEvent):
  """Represents repeated application of a mechanism.

  The repeated applications may be adaptive, where the query producing each
  event depends on the results of prior queries.

  This is equivalent to `ComposedDpEvent` that contains a list of length `count`
  of identical copies of `event`.
  """
  event: DpEvent
  count: int


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ComposedDpEvent(DpEvent):
  """Represents application of a series of composed mechanisms.

  The composition may be adaptive, where the query producing each event depends
  on the results of prior queries.
  """
  events: List[DpEvent]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class PoissonSampledDpEvent(DpEvent):
  """Represents an application of Poisson subsampling.

  Each record in the dataset is included in the sample independently with
  probability `sampling_probability`. Then the `DpEvent` `event` is applied
  to the sample of records.
  """
  sampling_probability: float
  event: DpEvent


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SampledWithReplacementDpEvent(DpEvent):
  """Represents sampling a fixed sized batch of records with replacement.

  A sample of `sample_size` (possibly repeated) records is drawn uniformly at
  random from the set of possible samples of a source dataset of size
  `source_dataset_size`. Then the `DpEvent` `event` is applied to the sample of
  records.
  """
  source_dataset_size: int
  sample_size: int
  event: DpEvent


@attr.s(frozen=True, slots=True, auto_attribs=True)
class SampledWithoutReplacementDpEvent(DpEvent):
  """Represents sampling a fixed sized batch of records without replacement.

  A sample of `sample_size` unique records is drawn uniformly at random from the
  set of possible samples of a source dataset of size `source_dataset_size`.
  Then the `DpEvent` `event` is applied to the sample of records.
  """
  source_dataset_size: int
  sample_size: int
  event: DpEvent

