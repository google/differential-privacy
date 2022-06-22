# Bounded Sum

[`BoundedSum`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/bounded-sum.h)
computes the sum of values in a dataset in a differentially private manner.

## Input & Output

`BoundedSum` supports `int64`s and `double`s as input. When successful, the
returned [`Output`](../protos.md) message will contain one element with
the differentially private sum, and a `ConfidenceInterval` describing the 95%
confidence interval of the noise added. When bounds are inferred, the `Output`
also contains a `BoundingReport`.

The differentially private sum provided by the `Output` is an unbiased estimate
of the raw bounded sum. Consequently, its value may sometimes be higher than the
upper bound or lower than the lower bound.

## Construction

`BoundedSum` is a bounded algorithm. There are no additional parameters.
Information on how to construct a `BoundedSum` is found in the
[bounded algorithm documentation](bounded-algorithm.md). Below is a minimal
construction example.

```
absl::StatusOr<std::unique_ptr<BoundedSum<int64>>> bounded_sum =
                 BoundedSum<int64>::Builder.SetEpsilon(1)
                                           .SetLower(-10)
                                           .SetUpper(10)
                                           .Build();
```

## Use

`BoundedSum` is an [`Algorithm`](algorithm.md) and supports its full API.

### Result Performance

For `BoundedSum`, calling `Result` is an O(n) operation.
