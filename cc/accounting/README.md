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

## Examples

We provide basic examples on how to use the library in example.cc.

### Run via Bazel

For running the example using Bazel, you need to have
[Bazel installed](https://docs.bazel.build/versions/main/install.html).
Once that is done, run:
```
bazel build :all
bazel run :example
```

### Common Issues
The current version of the library is not supported on Windows.
