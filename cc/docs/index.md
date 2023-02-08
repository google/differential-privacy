# Differential Privacy

For a general introduction to differential privacy, check the
[top-level docs](../../differential_privacy.md).

All of our algorithms inherit from the [`Algorithm`](algorithms/algorithm.md)
base class; you can find more details about the API there.

Here's a minimal example showing how to compute the count of some data:

```c++
#include "algorithms/count.h"

// Epsilon is a configurable parameter. A lower value means more privacy but
// less accuracy.
double count_elements(const std::vector<std::string>& values, double epsilon) {
  // Construct the Count object to run on std::string input.
  std::unique_ptr<differential_privacy::Count<std::string>> count =
     differential_privacy::Count<std::string>::Builder()
       .SetEpsilon(epsilon)
       .Build()
       .value();

  // Compute the count and get the result.
  differential_privacy::Output result = count->Result(values.begin(), values.end());

  // GetValue can be used to extract the value from an Output (or Input)
  // protobuf. For count, this is always an int value.
  return differential_privacy::GetValue<int>(result);
}

```

## Caveats

All of our algorithms assume a limited number of input elements per user (1 per
user by default). Clients are responsible for enforcing that limit before input
is passed to our library, and can specify how many inputs each user is allowed.
See the documentation of [`Algorithm`](algorithms/algorithm.md) for more detail.

Our library does not protect against integer overflows, since such protection
could violate DP.

## Error codes

Our libraries use
[Abseil error codes](https://abseil.io/docs/cpp/guides/status-codes) for the
following circumstances:

*   `UNIMPLEMENTED` indicates the functionality is not yet supported by the
    library (e.g. some algorithms do not have confidence intervals implemented).
*   `INVALID_ARGUMENT` indicates that an argument was improperly set. This
    includes parameters that are nonsensical (e.g., `epsilon < 0`, `delta < 0`),
    not provided, or will not produce useful output (e.g., `epsilon = 0`,
    `delta = 1`). In other cases, parameters may be theoretically valid, but
    outside the range that the machine can support (e.g., `sensitivity /
    epsilon` would make the geometric distribution overflow, the difference
    between the upper and lower bound is larger than the maximum representable
    double).
*   `FAILED_PRECONDITION` generally indicates that there is not enough data to
    perform the desired computation. These errors are produced in a
    differentially private manner, and still consume the allotted epsilon and
    delta.
*   `INTERNAL` indicates the input data is malformed, invalid or missing. It
    usually happens in the context of merging summaries. This probably means
    that there is a bug in the system using the library.
