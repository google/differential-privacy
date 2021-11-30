# Differential Privacy Accounting

This directory contains tools for tracking differential privacy budgets,
available as part of the
[Google differential privacy library](https://github.com/google/differential-privacy).
Currently, it provides an implementation of Privacy Loss Distributions (PLDs)
which can help compute an accurate estimate of the total ε, δ across multiple
executions of differentially private aggregations. Our implementation currently
supports Laplace mechanisms, Gaussian mechanisms and randomized response. More
detailed definitions and references can be found
[in our supplementary pdf document](https://github.com/google/differential-privacy/tree/main/common_docs/Privacy_Loss_Distributions.pdf).

We test this library on Linux with Python version 3.7. If you experience any
problems, please file an issue on GitHub, also for other platforms or Python
versions.

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

If you are using python other than 3.7, you may need to add the flag
--//dp_accounting:python_version=x.x to make Bazel build works. For example,
if your python version is 3.6, you may need to run
```
bazel build dp_accounting:all --//dp_accounting:python_version=3.6
bazel run dp_accounting:privacy_loss_distribution_basic_example --//dp_accounting:python_version=3.6
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
