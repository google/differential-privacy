# Bounded Algorithms

A bounded algorithm is any algorithm that requires lower and upper input bounds
as parameters. There is no `BoundedAlgorithm` interface as a subclass of
[`Algorithm`](algorithm.md), but `Algorithms` that need bounds do share a common
builder interface. Some Bounded algorithms are constructed using a
[`BoundedAlgorithmBuilder`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/bounded-algorithm.h),
which is a subclass of
[`AlgorithmBuilder`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/algorithm.h).
Others have builders which do not inherit from `BoundedAlgorithmBuilder`, but
share the same interface.

## Construction

Most bounded algorithms can be constructed in two ways. All bounded algorithms
can be constructed by setting lower and upper input bounds directly. Some
bounded algorithms can additionally be constructed without bounds, in which case
they will spend a portion of their privacy budget (epsilon and delta) to
automatically find bounds. Algorithms find bounds using the
[`ApproxBounds`](approx-bounds.md) algorithm; see its page for more information.
If you have knowledge about the range of your input data it is often most
efficient to use it to set bounds explicitly. This will leave more budget for
adding noise. Otherwise, it is often better to infer the bounds than to guess
and overestimate them.

```
BoundedAlgorithmBuilder builder =
  BoundedAlgorithmBuilder().SetEpsilon(double epsilon)

// Option 1: Set bounds directly.
absl::StatusOr<std::unique_ptr<Algorithm<T>>> bounded_algorithm =
                  builder.SetLower(T lower)
                         .SetUpper(T upper)
                         .Build();

// Option 2: Automatically infer bounds.
absl::StatusOr<std::unique_ptr<Algorithm<T>>> bounded_algorithm =
                  builder.Build();
```

*   `T` is the template parameter type (usually `int64_t` or `double`).
*   `T lower`: The lower bound of input to the algorithm. If an input is less
    than `lower`, it will be clamped to `lower`.
*   `T upper`: The upper bound of input to the algorithm. If an input is greater
    than `upper`, it will be clamped to `upper`.

## List of Bounded Algorithms

The following algorithms are bounded, and automatically infer bounds using the
[`ApproxBounds`](approx-bounds.md) algorithm if they are not set manually.

*   [`BoundedSum`](bounded-sum.md)
*   [`BoundedMean`](bounded-mean.md)
*   [`BoundedVariance`](bounded-variance.md)

The following algorithm is bounded, but require the user to specify bounds. If
bounds are not known in advance, you can run `ApproxBounds` separately, and
use the results as bounds.

*   [`Quantiles`](quantiles.md)

## How to Choose Bounds

How do the bounds affect the algorithm? As we have mentioned, values less than
the lower bound or greater than the upper bound are clamped to be equal to the
lower or upper bound, respectively. If the interval between bounds is too
narrow, many input values will be clamped, degrading accuracy.

However, bounds are usually used to determine the amount of noise we add to the
algorithm results to provide privacy. The relationship between the bounds used
and the amount of noise added varies between algorithms. However, as a general
rule, larger bound intervals map to more noise added. So a loose bound interval
around an input set also degrades accuracy. We must set our bounds to balance
between these two sources of inaccuracy.
