# Approx Bounds

[`ApproxBounds`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/approx-bounds.h)
computes an approximate upper and lower bound of the input set. It is designed
to be used to find the approximate support of a large input set, not to obtain a
precise minimum or maximum. This algorithm is mainly used to determine the
bounds of input sets inside [bounded algorithms](bounded-algorithm.md).

## Input & Output

`ApproxBounds` supports `int64_t` and `double` type input sets. When successful,
the returned [`Output`](../protos.md) message will contain two elements. The
first is the differentially private lower bound; the second is the upper. The
`Output` message will contain an error status instead if not enough inputs were
seen to determine the min and max.

## Construction

`ApproxBounds` has all the parameters of any [`Algorithm`](algorithm.md). There
are additional parameters.

```
ApproxBounds<T>::Builder builder =
   ApproxBounds<T>::Builder.SetNumBins(int64_t num_bins)
                           .SetScale(double scale)
                           .SetBase(double base)

// Option 1: Set the success probability for finding reasonable min/max.
absl::StatusOr<std::unique_ptr<ApproxBounds<T>>> approx_bounds =
                    builder.SetSuccessProbability(double success_probability)
                           .Build();

// Option 2: Set the bin threshold to choose a reasonable min/max.
absl::StatusOr<std::unique_ptr<ApproxBounds<T>>> approx_bounds =
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
noisy count of inputs inside each bin is checked to see whether it exceeds a
threshold. Out of bins that exceed the threshold, the bin representing the
interval containing the largest values maps to the maximum; the one for the
smallest values maps to the minimum. The threshold is configured bia a
`success_probability`, which is the probability that we will (correctly) return
no bounds for an empty dataset. Therefore the threshold is set high enough that
there is a less than `1 - success_probability` chance that noise added to any of
the bins will exceed the threshold. If no bin passes the threshold, we will
automatically relax the `success_probobility` slightly, until we either find
bounds, or reach a minimum `success_probability.` We find that this improves
performance on small datasets without substantially increasing the probability
of returning spurious bounds.

It is best to use the default value for `success_probability`.

It is also possible to set the `threshold` directly, though we only recommend
doing so for tests, as it is difficult to predict the results of setting a
particular threshold on real data.

## Use

`ApproxBounds` is an [`Algorithm`](algorithm.md), and supports its full API.

### By Other Algorithms

`ApproxBounds` is used by some [bounded algorithms](bounded-algorithm.md) to
automatically infer bounds. These bounded algorithms will be default create a
default `ApproxBounds` algorithm, but the caller can pass in a custom one if
they so choose.

```
builder.SetApproxBounds(std::unique_ptr<ApproxBounds> approx_bounds)
```

Since `ApproxBounds` is designed to be called by other algorithms, it contains
additional functions in its
[API](https://github.com/google/differential-privacy/blob/main/cc/algorithms/approx-bounds.h)
to get information on its bin bounds, and to combine inputs into partial
aggregations that can be clamped and fully aggregated once bounds are known.

### Result Performance

For `ApproxBounds`, calling `Result` is an O(n) operation.
