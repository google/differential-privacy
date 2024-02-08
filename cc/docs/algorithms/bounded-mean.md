# Bounded Mean

[`BoundedMean`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/bounded-mean.h)
computes the average of values in a dataset in a differentially private manner.

## Input & Output

`BoundedMean` supports `int64_t` and `double` types as input. When successful,
the returned [`Output`](../protos.md) message will contain one element
containing the differentially private mean. When bounds are inferred, the
`Output` additionally contains a `BoundingReport`. The returned value is
guaranteed to be <= the upper bound, and >= the lower bound.

## Construction

`BoundedMean` is a bounded algorithm. There are no additional parameters.
Information on how to construct a `BoundedMean` is found in the
[bounded algorithm documentation](bounded-algorithm.md). Below is a minimal
construction example.

```
absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bounded_mean =
   BoundedMean<int64_t>::Builder.SetEpsilon(1)
                                .SetLower(-10)
                                .SetUpper(10)
                                .Build();
```

## Use

`BoundedMean` is an [`Algorithm`](algorithm.md) and supports its full API.

### Result Performance

For `BoundedMean`, calling `Result` is an O(n) operation.
