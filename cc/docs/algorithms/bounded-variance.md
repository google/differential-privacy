# Bounded Variance

[`BoundedVariance`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/bounded-variance.h) computes the
variance of values in a dataset, in a differentially private manner.

## Input & Output

`BoundedVariance` supports `int64_t` and `double` types as input. When
successful, the returned [`Output`](../protos.md) message will contain one
element with the differentially private variance. When bounds are inferred, the
`Output` additionally contains a `BoundingReport`. The returned value is
guaranteed to be non-negative, with a maximum possible value of the maximum
variance (length of the bounded interval squared divided by four).

## Construction

`BoundedVariance` is a bounded algorithm. There are no additional parameters.
Information on how to construct a `BoundedVariance` is found in the
[bounded algorithm documentation](bounded-algorithm.md). Below is a minimal
construction example.

```
absl::StatusOr<std::unique_ptr<BoundedVariance<int64_t>>> bounded_var =
            BoundedVariance<int64_t>::Builder.SetEpsilon(1)
                                             .SetLower(-10)
                                             .SetUpper(10)
                                             .Build();
```

## Use

`BoundedVariance` is an [`Algorithm`](algorithm.md) and supports its full API.

### Result Performance

For `BoundedVariance`, calling `Result` is an O(n) operation and requires O(1)
additional memory.
