# Order Statistics

WARNING: These algorithms are deprecated and may be removed soon. Please use
[Quantiles](quantiles.md) (which is more accurate) instead.

We have a set of algorithms for calculating
[order statistics](https://github.com/google/differential-privacy/blob/main/cc/algorithms/order-statistics.h)
(aka quantiles, percentiles). The following are supported:

*   `Max`
*   `Min`
*   `Median`
*   `Percentile` for percentile `p`.

`Max`, `Min`, and `Median` are convenience wrappers around `Percentile`, which
can be used to calculate any of the other quantities.

## Input & Output

The order statistics algorithms support any numeric type. Their `Output`s
contain an element with a single value. `ConfidenceInterval` and
`BoundingReport` are not provided.

## Construction

The order statistics algorithms are [bounded algorithms](bounded-algorithm.md).
However, when bounds are not manually set, the algorithms do not infer input
bounds by spending privacy budget. Instead, they use the numeric limits of the
input type as bounds. The `Percentile` algorithm requires the additional
parameter `percentile`. For example, when constructing the `Percentile`
algorithm:

```
absl::StatusOr<std::unique_ptr<Percentile<T>>> percentile =
   Percentile<T>::Builder.SetPercentile(double percentile)
                         .Build();
```

*   `T`: The input type, for example `double` or `int64`.
*   `double percentile`: This parameter is required for the `Percentile`
    algorithm and cannot be set for the other order statistics algorithms. It is
    the percentile you wish to find.

## Use

The order statistics algorithms are [`Algorithm`s](algorithm.md) and supports
its full API.

### Result Performance

For order statistics algorithms, calling `Result` has a time complexity of O(n).
Since all inputs are stored in an internal vector, space complexity is O(n).
