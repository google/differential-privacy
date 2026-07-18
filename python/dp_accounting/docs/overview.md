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
accountant = dp.accounting.pld.PLDAccountant()
accountant.compose(dp_accounting.GaussianDpEvent(2.0))
accountant.get_epsilon(1e-5)
```

## Installation

```bash
pip install dp-accounting
```
