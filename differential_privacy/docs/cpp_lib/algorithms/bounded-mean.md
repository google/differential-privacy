

<!-- This file is auto-generated. Do not edit. -->

# Bounded Mean

[`BoundedMean`](https://github.com/google/differential-privacy/blob/master/differential_privacy/algorithms/bounded-mean.h)
computes the average of values in a dataset, in a differentially private manner.

## Input & Output

`BoundedMean` supports `int64` and `double` type input sets. When successful,
the returned [`Output`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/protos.md) message will contain one
element containing the differentially private mean. When bounds are inferred,
the `Output` additionally contains a `BoundingReport`. The returned value is
guaranteed to be inclusively within the input bounds.

## Construction

`BoundedMean` is a bounded algorithm. There are no additional parameters.
Information on how to construct a `BoundedMean` is found in the
[bounded algorithm documentation](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-algorithm.md).
Below is a minimal construction example.

```
util::StatusOr<std::unique_ptr<BoundedMean<int64>>> bounded_mean =
   BoundedMean<int64>::Builder.SetEpsilon(1)
                               .SetLower(-10)
                               .SetUpper(10)
                               .Build();
```

## Use

`BoundedMean` is an [`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md)
and supports its full API.

### Result Performance

For `BoundedMean`, calling `Result` is an O(n) operation.
