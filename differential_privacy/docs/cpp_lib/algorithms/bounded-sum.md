

<!-- This file is auto-generated. Do not edit. -->

# Bounded Sum

[`BoundedSum`](https://github.com/google/differential-privacy/blob/master/differential_privacy/algorithms/bounded-sum.h)
computes the sum of values in a dataset, in a differentially private manner.

## Input & Output

`BoundedSum` supports `int64` and `double` type input sets. When successful, the
returned [`Output`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/protos.md) message will contain one
element containing the differentially private sum, and a `ConfidenceInterval`
describing the 95% confidence interval of the noise added. When bounds are
inferred, the `Output` additionally contains a `BoundingReport`.

## Construction

`BoundedSum` is a bounded algorithm. There are no additional parameters.
Information on how to construct a `BoundedSum` is found in the
[bounded algorithm documentation](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-algorithm.md).
Below is a minimal construction example.

```
util::StatusOr<std::unique_ptr<BoundedSum<int64>>> bounded_sum =
                 BoundedSum<int64>::Builder.SetEpsilon(1)
                                           .SetLower(-10)
                                           .SetUpper(10)
                                           .Build();
```

## Use

`BoundedSum` is an [`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md) and
supports its full API.

### Result Performance

For `BoundedSum`, calling `Result` is an O(n) operation.
