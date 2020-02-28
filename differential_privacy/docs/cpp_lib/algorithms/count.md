

<!-- This file is auto-generated. Do not edit. -->

# Count

[`Count`](https://github.com/google/differential-privacy/blob/master/differential_privacy/algorithms/count.h) computes
the number of values in a dataset, in a differentially private manner.

## Input & Output

`Count` supports any input type. Count always returns an
[`Output`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/protos.md) message containing a single element
containing the differentially private count, and a `ConfidenceInterval`
containing the 95% confidence interval of noise added.

## Construction

`Count` takes the usual parameters for
[`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md), with no additional
parameters.

## Use

`Count` is an [`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md) and
supports its full API. Below is a minimal construction example.

```
util::StatusOr<std::unique_ptr<Count<int64>>> count =
              Count<int64>::Builder.SetEpsilon(1)
                                   .Build();
```

### Result Performance

For `Count`, calling `Result` is an O(n) operation. `Count` uses O(1) memory.
