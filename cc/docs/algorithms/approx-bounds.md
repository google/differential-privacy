
# Approx Bounds

[`ApproxBounds`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/approx-bounds.h) computes an
approximate minimum and maximum of the input set. It is designed to be used to
find the approximate support of a large input set, not to obtain a precise
minimum or maximum. In practice, this algorithm is used to determine the bounds
of input sets inside [bounded algorithms](bounded-algorithm.md).

## Input & Output

`ApproxBounds` supports `int64` and `double` type input sets. When successful,
the returned [`Output`](../protos.md) message will contain two elements. The
first is the differentially private minimum; the second is the maximum. The
`Output` message will contain an error status instead if not enough inputs were
seen to determine the min and max.

## Construction

`ApproxBounds` has all the parameters of any [`Algorithm`](algorithm.md). There
are additional parameters.

```
ApproxBounds<T>::Builder builder =
   ApproxBounds<T>::Builder.SetNumBins(int64 num_bins)
                           .SetScale(double scale)
                           .SetBase(double base)

// Option 1: Set the success probability for finding reasonable min/max.
util::StatusOr<std::unique_ptr<ApproxBounds<T>>> approx_bounds =
                    builder.SetSuccessProbability(double success_probability)
                           .Build();

// Option 2: Set the bin threshold to choose a reasonable min/max.
util::StatusOr<std::unique_ptr<ApproxBounds<T>>> approx_bounds =
                    builder.SetThreshold(double threshold)
                           .Build();
```

All of these additional parameters are optional and have default values. To
manually set appropriate values for these parameters, we must understand a bit
of the workings behind `ApproxBounds`. The algorithm first creates histogram
bins that represent contiguous intervals in the range of data type `T`. The
width of the intervals increases exponentially with base `base`. The `scale` is
the width of the smallest bin. There are `num_bins` bins for positive numbers,
and `num_bins` bins for negative numbers.

Inputs are partitioned into the histogram bins. When fetching the result, a
noisy count of inputs inside each bin is checked to see whether it exceeds the
`threshold`. Out of bins that exceed the `threshold`, the bin representing the
interval containing the largest values maps to the maximum; the one for the
smallest values maps to the minimum. Each `threshold` maps to a
`success_probability`, which is the probability that the bin chosen will not
contain a true count of 0. We only need to specify either `threshold` or
`success_probability`, since each maps to a specific value of the other.

Without a complete understanding, it is best to use the default
values for all of these additional parameters.

## Use

`ApproxBounds` is an [`Algorithm`](algorithm.md), and supports its full API.

### By Other Algorithms

`ApproxBounds` is used by some [bounded algorithms](bounded-algorithm.md) to
automatically infer bounds. While a default `ApproxBounds` algorithm will be
created by these bounded algorithms, a custom one can be passed in.

```
builder.SetApproxBounds(std::unique_ptr<ApproxBounds> approx_bounds)
```

Since `ApproxBounds` is designed to be called by other algorithms, it contains
additional functions in its [API](https://github.com/google/differential-privacy/blob/main/cc/algorithms/approx-bounds.h) to
reveal its underlying structure.

### Result Performance

For `ApproxBounds`, calling `Result` is an O(n) operation.
