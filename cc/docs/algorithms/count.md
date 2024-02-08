# Count

[`Count`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/count.h)
computes the number of values in a dataset in a differentially private manner.

## Input & Output

`Count` supports any input type. Count always returns an
[`Output`](../protos.md) message containing a single element with the
differentially private count, and a `ConfidenceInterval` with the 95%
confidence interval of noise added.

The differentially private count provided by the `Output` is an unbiased
estimate of the raw count. Consequently, its value may sometimes be negative, in
particular if the raw count is close to 0.

## Construction

`Count` takes the usual parameters for [`Algorithm`](algorithm.md), with no
additional parameters.

## Use

`Count` is an [`Algorithm`](algorithm.md) and supports its full API. Below is a
minimal construction example.

```
absl::StatusOr<std::unique_ptr<Count<int64_t>>> count =
              Count<int64_t>::Builder.SetEpsilon(1)
                                     .Build();
```

### Result Performance

For `Count`, calling `Result` is an O(n) operation. `Count` uses O(1) memory.
