## Differential Privacy library in C++

This is the C++ implementation of the differential privacy library. For general
details and key definitions, see the top-level documentation.
This document describes C++-specific aspects.

## How to Use

Full documentation on how to use the library is in the
[cpp/docs](https://github.com/google/differential-privacy/tree/main/cc/docs)
subdirectory. Here's a minimal example showing how to compute the count of some
data:

```
#include "algorithms/count.h"

// Epsilon is a configurable parameter. A lower value means more privacy but
// less accuracy.
int64_t count(const std::vector<double>& values, double epsilon) {
  // Construct the Count object to run on double inputs.
  std::unique_ptr<differential_privacy::Count<double>> count =
     differential_privacy::Count<double>::Builder().SetEpsilon(epsilon)
                                                   .Build()
                                                   .ValueOrDie();

  // Compute the count and get the result.
  absl::StatusOr<differential_privacy::Output> result =
     count->Result(values.begin(), values.end());
  if (!result.ok()) {
    return 0;
  }

  // GetValue can be used to extract the value from an Output protobuf. For
  // count, this is always an int64_t value.
  return differential_privacy::GetValue<int64_t>(result.ValueOrDie());
}
```

We also include the following example code:
- A [tool for releasing epsilon-DP aggregate statistics](https://github.com/google/differential-privacy/tree/main/examples/cc/).
- A [PostgreSQL extension](https://github.com/google/differential-privacy/tree/main/cc/postgres)
that adds epsilon-DP aggregate functions.


