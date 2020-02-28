

<!-- This file is auto-generated. Do not edit. -->

# Bounded Algorithms

A bounded algorithm is any algorithm that requires lower and upper input bounds
as parameters. Note that there does not exist a `BoundedAlgorithm` interface as
a subclass of [`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md). Bounded
algorithms are constructed using a
[`BoundedAlgorithmBuilder`](https://github.com/google/differential-privacy/blob/master/differential_privacy/algorithms/bounded-algorithm.h),
which is a subclass of
[`AlgorithmBuilder`](https://github.com/google/differential-privacy/blob/master/differential_privacy/algorithms/algorithm.h).

## Construction

Bounded algorithms can be constructed in two ways. The first way is to set lower
and upper input bounds directly. The second is to omit setting the bounds. If
bounds are omitted, some algorithms will spend a portion of the privacy budget
to automatically infer and set the bounds. How exactly the bounded algorithm
infers the bounds can be configured using the
[`ApproxBounds`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/approx-bounds.md) algorithm; see
its page for more information. We can set lower and upper bounds directly if we
have knowledge about the range of our input data. Otherwise, it is better to
infer the bounds.

```
BoundedAlgorithmBuilder builder =
  BoundedAlgorithmBuilder.SetEpsilon(double epsilon)

// Option 1: Set bounds directly.
util::StatusOr<std::unique_ptr<Algorithm<T>>> bounded_algorithm =
                  builder.SetLower(T lower)
                         .SetUpper(T upper)
                         .Build();

// Option 2: Automatically infer bounds.
util::StatusOr<std::unique_ptr<Algorithm<T>>> bounded_algorithm =
                  builder.Build();
```

*   `T` is the template parameter type (usually `int64` or `double`).
*   `T lower`: The lower bound of input to the algorithm. If an input is less
    than `lower`, it will be clamped to `lower`.
*   `T upper`: The upper bound of input to the algorithm. If an input is greater
    than `upper`, it will be clamped to `upper`.

## List of Bounded Algorithms

The following algorithms are bounded, and automatically infer bounds using the
[`ApproxBounds`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/approx-bounds.md) algorithm if
they are not set manually.

*   [`BoundedSum`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-sum.md)
*   [`BoundedMean`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-mean.md)
*   [`BoundedVariance`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-variance.md)
*   [`BoundedStandardDeviation`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/bounded-standard-deviation.md)

The following algorithms are bounded, but use numeric limits as bounds if they
are not set manually.

*   [`Order Statistics`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/order-statistics.md)

## How to Choose Bounds

How do the bounds affect the algorithm? As we have mentioned, values less than
the lower bound or greater than the upper bound are clamped to be equal to the
lower or upper bound, respectively. If the interval between bounds is too
narrow, many input values will be clamped, degrading accuracy.

However, bounds are also used to determine the amount of noise we add to the
algorithm results to provide privacy. The relationship between the bounds used
and the amount of noise added varies between algorithms. However, as a general
rule, larger bound intervals map to more noise added. So a loose bound interval
around an input set also degrades accuracy. We must set our bounds to balance
between these two sources of inaccuracy.
