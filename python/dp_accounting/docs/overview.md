<!--
Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Overview

`dp-accounting` is a Python library for (differential) privacy accounting.
It defines a number of `DpEvent`s, which can be used to describe a variety of
mechanisms. These events can be passed to a `PrivacyAccountant`, which can then
compute the privacy parameters for the (composition of) events it has seen so
far. For example, this is how we can compute epsilon at `delta=1e-5` for a
single Gaussian mechanism with noise multiplier `2.0`:

```python
accountant = dp_accounting.PLDAccountant()
accountant.compose(dp_accounting.GaussianDpEvent(2.0))
accountant.get_epsilon(1e-5)
```

In addition to `DpEvent` and `PrivacyAccountant`, `dp-accounting` provides
top-level utilities for:

* **Mechanism calibration** (`dp_accounting.calibrate_dp_mechanism`): finding
  the optimal noise or subsampling parameter to hit a target `(epsilon, delta)`
  budget.
* **Event tree canonicalization** (`dp_accounting.canonicalize`): simplifying
  nested `DpEvent` compositions, grouping repeated events, and optionally
  merging composed Gaussians into a single effective Gaussian for faster
  accounting.
* **Exact analytical Gaussian conversions**
  (`dp_accounting.get_sigma_gaussian`, `dp_accounting.get_epsilon_gaussian`):
  closed-form calibration and accounting for a single Gaussian mechanism without
  PLD discretization overhead.

## Installation

```bash
pip install dp-accounting
```
