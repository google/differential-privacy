
# Differential Privacy

For a general introduction to differential privacy, check the
[top-level docs](https://github.com/google/differential-privacy/blob/main/differential_privacy.md).

All of our algorithms inherit from the [`Algorithm`](algorithms/algorithm.md)
base class; you can find more details about the API there.

Here's a minimal example showing how to compute the count of some data:

```
#include "algorithms/count.h"

// Epsilon is a configurable parameter. A lower value means more privacy but
// less accuracy.
double count(const vector<double>& vals, double epsilon) {
  // Construct the Count object to run on double inputs.
  std::unique_ptr<differential_privacy::Count<double>> count =
     differential_privacy::Count<double>::Builder().SetEpsilon(epsilon)
                                                   .Build()
                                                   .ValueOrDie();

  // Compute the count and get the result.
  differential_privacy::Output result = count->Result(v.begin(), v.end());

  // GetValue can be used to extract the value from an Output (or Input)
  // protobuf. For count, this is always an int value.
  return differential_privacy::GetValue<int>(result);
}

```

## Caveats

All of our algorithms assume one input element per user. All of a user's
contributions should be reduced to a single input element before being passed to
our algorithms.

## Error codes

Our libraries use
[Abseil error codes](https://github.com/abseil/abseil-cpp/blob/master/absl/status/status.h#L90)
for the following circumstances:

*   `UNIMPLEMENTED` indicates the functionality is not yet supported by the
    library.
*   `INVALID_ARGUMENT` indicates that an argument was improperly set. This
    includes parameters that are nonsensical (e.g., `epsilon < 0`, `delta < 0`),
    not provided, or will not produce useful output (e.g., `epsilon = 0`,
    `delta = 1`). In other cases, parameters may be theoretically valid, but
    outside the range that the machine can support (e.g., `sensitivity / epsilon`
    would make the geometric distribution overflow, the difference between the
    upper and lower bound is larger than the maximum representable double).
*   `FAILED_PRECONDITION` generally indicates that there is not enough data to
    perform the desired computation.
*   `INTERNAL` indicates the input data is malformed, invalid, missing, or other
    problems with merging. This likely indicates that the library is being used
    incorrectly.
