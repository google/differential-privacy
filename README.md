# Differential Privacy

This project contains a C++ library of ε-differentially private algorithms,
which can be used to produce aggregate statistics over numeric data sets
containing private or sensitive information. In addition, we provide a
stochastic tester to check the correctness of the algorithms. Currently, we
provide algorithms to compute the following:

  * Count
  * Sum
  * Mean
  * Variance
  * Standard deviation
  * Order statistics (including min, max, and median)

We also provide an implementation of the laplace mechanism that can be used to
perform computations that aren't covered by our pre-built algorithms.

All of these algorithms are suitable for research, experimental or production
use cases.

This project also contains a
[stochastic tester](https://github.com/google/differential-privacy/tree/master/differential_privacy/testing),
used to help catch regressions that could make the differential privacy
property no longer hold.

## How to Build

In order to run the differential private library, you need to install Bazel,
if you don't have it already. [Follow the instructions for your platform on the
Bazel website](https://docs.bazel.build/versions/master/install.html)

You also need to install Git, if you don't have it already.
[Follow the instructions for your platform on the Git website.](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Once you've installed Bazel and Git, open a Terminal and clone the
differential privacy directory into a local folder:

```git clone https://github.com/google/differential-privacy.git```

Navigate into the ```differential-privacy``` folder you just created,
and build the differential privacy library and dependencies using Bazel:

```bazel build differential_privacy/...```

You may need to install additional dependencies when building the PostgreSQL
extension, for example on Ubuntu you will need these packages:

```sudo apt-get install libreadline-dev bison flex```

## How to Use

Full documentation on how to use the library is in the
[cpp/docs](https://github.com/google/differential-privacy/tree/master/differential_privacy/docs)
subdirectory. Here's a minimal example showing how to compute the count of some
data:

```
#include "differential_privacy/algorithms/count.h"

// Epsilon is a configurable parameter. A lower value means more privacy but
// less accuracy.
int64_t count(const vector<double>& values, double epsilon) {
  // Construct the Count object to run on double inputs.
  std::unique_pointer<differential_privacy::Count<double>> count =
     differential_privacy::Count<double>::Builder().SetEpsilon(epsilon)
                                                   .Build()
                                                   .ValueOrDie();

  // Compute the count and get the result.
  differential_privacy::Output result =
     count->Result(values.begin(), values.end());

  // GetValue can be used to extract the value from an Output protobuf. For
  // count, this is always an int64_t value.
  return differential_privacy::GetValue<int64_t>(result);
}

```

We also include the following example code:
- A [tool for releasing epsilon-DP aggregate statistics](https://github.com/google/differential-privacy/tree/master/differential_privacy/example).
- A [PostgreSQL extension](https://github.com/google/differential-privacy/tree/master/differential_privacy/postgres)
that adds epsilon-DP aggregate functions.

## Caveats

All of our code assume that each user contributes only a single row to each
aggregation. You can use the library to build systems that allow multiple
contributions per user - [our paper](https://arxiv.org/abs/1909.01917) describes
one such system. To do so, multiple user contributions should be combined before
they are passed to our algorithms. We chose not to implement this step at the
library level because it's not the logical place for it - it's much easier to
sort contributions by user and combine them together with a distributed
processing framework before they're passed to our algorithms.

## Support

We will continue to publish updates and improvements to the library. We will not
accept pull requests for the immediate future. We will respond to issues filed
in this project. If we intend to stop publishing improvements and responding to
issues we will publish notice here at least 3 months in advance.

## License

[Apache License 2.0](LICENSE)

## Support Disclaimer

This is not an officially supported Google product.
