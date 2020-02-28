

<!-- This file is auto-generated. Do not edit. -->

# Bounded Standard Deviation

[`BoundedStandardDeviation`](https://github.com/google/differential-privacy/blob/master/differential_privacy/algorithms/bounded-standard-deviation.h)
computes the standard deviation of values in a dataset, in a differentially
private manner.

## Input & Output

`BoundedStandardDeviation` supports `int64` and `double` type input sets. When
successful, the returned [`Output`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/protos.md) message will
contain one element containing the differentially private standard deviation.
When bounds are inferred, the `Output` additionally contains a `BoundingReport`.
The returned value is guaranteed to be non-negative, with a maximum possible
value of the size of the bounded interval.

## Construction

`BoundedStandardDeviation` is a bounded algorithm. There are no additional
parameters. Information on how to construct a `BoundedStandardDeviation` is
found in the
[bounded algorithm documentation](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-algorithm.md).
Below is a minimal construction example.

```
util::StatusOr<std::unique_ptr<BoundedStandardDeviation<int64>>> bounded_stdev =
   BoundedStandardDeviation<int64>::Builder.SetEpsilon(1)
                                           .SetLower(-10)
                                           .SetUpper(10)
                                           .Build();
```

## Use

`BoundedStandardDeviation` is an
[`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md) and supports its full
API.

### Result Performance

For `BoundedStandardDeviation`, calling `Result` is an O(n) operation and
requires O(1) additional memory.
