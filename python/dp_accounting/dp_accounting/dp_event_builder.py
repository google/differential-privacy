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
"""Builder and canonicalization utilities for DpEvents."""

import attr

from dp_accounting import dp_event


class DpEventBuilder(object):
  """Constructs a `DpEvent` representing the composition of a series of events.

  Two common use cases of the `DpEventBuilder` are 1) for producing and tracking
  a ledger of `DpEvent`s during sequential accounting using a
  `PrivacyAccountant`, and 2) for building up a description of a composite
  mechanism for subsequent batch accounting.
  """

  def __init__(self):
    # A list of (event, count) pairs.
    self._event_counts = []
    self._composed_event = None

  def compose(self, event: dp_event.DpEvent, count: int = 1):
    """Composes new event into event represented by builder.

    Args:
      event: The new event to compose.
      count: The number of times to compose the event.
    """
    if not isinstance(event, dp_event.DpEvent):
      raise TypeError('`event` must be a subclass of `DpEvent`. '
                      f'Found {type(event)}.')
    if not isinstance(count, int):
      raise TypeError(f'`count` must be an integer. Found {type(count)}.')
    if count < 1:
      raise ValueError(f'`count` must be positive. Found {count}.')

    if isinstance(event, dp_event.NoOpDpEvent):
      return
    elif isinstance(event, dp_event.SelfComposedDpEvent):
      self.compose(event.event, count * event.count)
    else:
      if self._event_counts and self._event_counts[-1][0] == event:
        new_event_count = (event, self._event_counts[-1][1] + count)
        self._event_counts[-1] = new_event_count
      else:
        self._event_counts.append((event, count))
      self._composed_event = None

  def build(self) -> dp_event.DpEvent:
    """Builds and returns the composed DpEvent represented by the builder."""
    if not self._composed_event:
      events = []
      for event, count in self._event_counts:
        if count == 1:
          events.append(event)
        else:
          events.append(dp_event.SelfComposedDpEvent(event, count))
      if not events:
        self._composed_event = dp_event.NoOpDpEvent()
      elif len(events) == 1:
        self._composed_event = events[0]
      else:
        self._composed_event = dp_event.ComposedDpEvent(events)

    return self._composed_event  # pyrefly: ignore[bad-return]


def canonicalize(
    event: dp_event.DpEvent,
    *,
    merge_gaussians: bool = False,
) -> dp_event.DpEvent:
  """Rewrites a DpEvent tree into canonical form.

  Performs the following structural simplifications:
    - Flattens nested ComposedDpEvents and SelfComposedDpEvents, distributing
      counts to leaf events.
    - Unwraps trivial SelfComposedDpEvent(_, 1) and single-element
      ComposedDpEvent.
    - Drops NoOpDpEvents from compositions.
    - Groups identical events within a composition into SelfComposedDpEvents,
      reordering by (type_name, repr) for a deterministic canonical form.

  Recursively canonicalizes DpEvent-typed fields of all event types (e.g. the
  inner event of PoissonSampledDpEvent).

  The output satisfies the following structural invariants:
    - No nested ComposedDpEvent or SelfComposedDpEvent.
    - No NoOpDpEvent inside a composition.
    - No SelfComposedDpEvent with count=1.
    - No ComposedDpEvent with a single element.
    - Children of ComposedDpEvent are distinct (grouped by structural equality)
      and sorted by (type_name, repr).
    - All DpEvent-typed fields are recursively canonical.

  Note: grouping reorders events within a ComposedDpEvent. This is semantically
  valid because the privacy guarantee of composition is order-independent, but
  the result no longer reflects the temporal order of mechanism applications.

  Args:
    event: The DpEvent to canonicalize.
    merge_gaussians: If True, merges GaussianDpEvents composed at the same level
      into a single equivalent GaussianDpEvent. Does not merge across sampling
      boundaries (e.g. inside PoissonSampledDpEvent).

  Returns:
    A semantically equivalent DpEvent in canonical form.
  """
  if isinstance(
      event, (dp_event.SelfComposedDpEvent, dp_event.ComposedDpEvent)
  ):
    counts = {}  # repr(event) → (event, total_count)
    sigma_inv_sq = 0.0
    for base, count in _flatten(event):
      canonical = canonicalize(base, merge_gaussians=merge_gaussians)
      if merge_gaussians and isinstance(canonical, dp_event.GaussianDpEvent):
        sigma_inv_sq += count / canonical.noise_multiplier**2
      elif (key := repr(canonical)) in counts:
        counts[key] = (canonical, counts[key][1] + count)
      else:
        counts[key] = (canonical, count)
    pairs = list(counts.values())
    if sigma_inv_sq > 0:
      pairs.append((dp_event.GaussianDpEvent(sigma_inv_sq**-0.5), 1))
    pairs.sort(key=lambda p: (type(p[0]).__name__, repr(p[0])))
    return _pairs_to_event(pairs)

  # For all other event types, recursively canonicalize DpEvent-typed fields.
  changes = {}
  for field in attr.fields(type(event)):
    value = getattr(event, field.name)
    if isinstance(value, dp_event.DpEvent):
      canonical = canonicalize(value, merge_gaussians=merge_gaussians)
      if canonical is not value:
        changes[field.name] = canonical
  return attr.evolve(event, **changes) if changes else event


def _flatten(event, multiplier=1):
  """Decompose event into (base_event, count) pairs, unwrapping composition."""
  if isinstance(event, dp_event.NoOpDpEvent):
    return
  if isinstance(event, dp_event.ComposedDpEvent):
    for e in event.events:
      yield from _flatten(e, multiplier)
  elif isinstance(event, dp_event.SelfComposedDpEvent):
    yield from _flatten(event.event, multiplier * event.count)
  else:
    yield (event, multiplier)


def _pairs_to_event(pairs):
  """Build a DpEvent from (event, count) pairs."""
  if not pairs:
    return dp_event.NoOpDpEvent()
  events = [
      event if count == 1 else dp_event.SelfComposedDpEvent(event, count)
      for event, count in pairs
  ]
  return events[0] if len(events) == 1 else dp_event.ComposedDpEvent(events)
