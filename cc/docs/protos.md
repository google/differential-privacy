# Input and Output

Our `Algorithm`s are templated by the input data type, and return output encoded
in an `Output` proto. We also provide some utility functions for converting
scalar data to/from `Output` protos.

## Input

Algorithms are templated on the input type. For example, we can find the
mean of integer values by building the
[`BoundedMean`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/bounded-mean.h)
algorithm like below. Note that the template only affects the input type; the
mean that we return will always be a double.

```
absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bounded_mean =
  BoundedMean<int64>::Builder.SetEpsilon(1)
                             .SetLower(-10)
                             .SetUpper(10)
                             .Build();
```

## [`Output`](https://github.com/google/differential-privacy/blob/main/proto/data.proto)

The `Output` contains the algorithm results, and any available accuracy details.

```
message Output {
  message Element {
    optional ValueType value = 1;
    optional ValueType error = 2;
  }
  repeated Element elements = 1;

  message ErrorReport {
    optional ConfidenceInterval noise_confidence_interval = 1;
    optional BoundingReport bounding_report = 2;
  }
  optional ErrorReport error_report = 3;
}
```

*   The `elements` contain the differentially private results of the algorithm.
    Each `ValueType` holds one value of the type specified by the algorithm.
    Note that `float_value` is a double and `int_value` is a long.
*   The `error_report` contains additional accuracy details. It is only
    populated for some algorithms. If populated, `noise_confidence_interval`
    will return the 95% confidence interval of the noise added. The
    `bounding_report` is populated in the case that a
    [`bounded algorithm`](algorithms/bounded-algorithm.md) automatically
    inferred bounds; see the [`ApproxBounds`](algorithms/approx-bounds.md)
    algorithm page for more information.

## Utility Functions

Converting to/from these types can be cumbersome, so we provide utility
functions for the common cases. We outline some of these functions below. See
the [utility function file](https://github.com/google/differential-privacy/blob/main/cc/proto/util.h)
for the full list.

### `GetValue<T>`

```
template <typename T>
T GetValue(const Output& output);
```

Extracts the first element from the provided `Output` proto. `T` is the
requested return type; the value in the proto must match that type. For example,
if `T` is an integral type, the element must have `int_value` set.

### `MakeOutput`

Creates an `Output` with one element. Which field gets set depends on the type
of the template parameter `T`. For example, `MakeOutput(7)` will set
`int_value`, but `MakeOutput(7.0)` will set `float_value`.

```
template <typename T>
MakeOutput(T value);
```

### `AddToOutput`

Adds an element to the given output. Which field gets set depends on the type of
the template parameter `T`. For example, `AddToOutput(&output, 7.0)` will add an
element to `output` and set the `float_value`.

```
template <typename T>
void AddToOutput(Output* output, T value);
```
