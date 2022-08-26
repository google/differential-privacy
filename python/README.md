# Differential Privacy Accounting

This directory contains tools for tracking differential privacy budgets,
available as part of the
[Google differential privacy library](https://github.com/google/differential-privacy).

The set of DpEvent classes allow you to describe complex differentially private
mechanisms such as Laplace and Gaussian, subsampling mechanisms, and their
compositions. The PrivacyAccountant classes can ingest DpEvents and return the
ε, δ of the composite mechanism. Privacy Loss Distributions (PLDs) and RDP
accounting are currently supported.

More detailed definitions and references about PLDs can be found
[in our supplementary pdf document](https://github.com/google/differential-privacy/tree/main/common_docs/Privacy_Loss_Distributions.pdf).

Our library only support Python version >= 3.9. We test this library on Linux
with Python version 3.9. If you experience any problems, please file an issue on
GitHub, also for other platforms or Python versions.

## Examples

We provide basic examples on how to use the library in
privacy_loss_distribution_basic_example.py. There are two ways to run this,
either via Bazel or after installing the library using setup.py.

### Run with Bazel

For running the example using Bazel, you need to have
[Bazel installed](https://docs.bazel.build/versions/main/install.html).
Once that is done, run:
```
bazel build dp_accounting:all
bazel run dp_accounting:privacy_loss_distribution_basic_example
```

### Run via setup.py

For the second option, you will need the
[setuptools package](https://pypi.org/project/setuptools/) installed.
To ensure this, you may run
```
pip install --upgrade setuptools
```
Then, to demonstrate our example, run:
```
python setup.py install
python dp_accounting/privacy_loss_distribution_basic_example.py
```
