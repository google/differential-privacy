
# Algorithm

[`Algorithm`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/algorithm.h)
is the base class of all
differentially private algorithms. Each algorithm can be constructed using the
builder pattern, which sets its parameters. `Algorithms` are stateful: first you
add data (possibly multiple times), and then you get a result. Algorithms are 
templated on the input type.

## Construction

```
absl::StatusOr<std::unique_ptr<Algorithm>> algorithm =
 AlgorithmBuilder.SetEpsilon(double epsilon)
                 .SetMaxPartitionsContributedTo(int max_partitions)
                 .SetMaxContributionsPerPartition(int max_contributions)
                 .SetLaplaceMechanism(std::unique_ptr<NumericalMechanism::Builder> mechanism_builder)
                 .Build();
```

*   `double epsilon`: The `epsilon` differential privacy parameter. A smaller
    number means more privacy but less accuracy.
*   `int max_partitions`: The number of aggregations, or 'partitions,' that each
    user is allowed to contribute to. Defaults to 1 if unset. The caller must
    guarantee that this limit is enforced on the input. The library cannot
    enforce it because it cannot distinguish between users or aggregations. Note
    that `Algorithm`s that will be merged together are considered part of the
    same partition.
*   `int max_contributions`: The number of pieces of input to this aggregation
    that can belong to a single user. Defaults to 1 if unset. The caller must
    guarantee that this limit is enforced on the input. The library cannot
    enforce it because it does not know which inputs belong to which users. If
    summaries from multiple `Algorithm`s are merged together, the total number
    of inputs from a single user across all merged `Algorithm`s must not exceed
    this limit.
*   `std::unique_ptr<NumericalMechanism::Builder> mechanism_builder`: Used
    to specify the type of numerical mechanism the algorithm will use to add
    noise (e.g. Laplace, Gaussian). In most cases this should not be set (and a
    default LaplaceMechanism will be used), but it can be used to remove or mock
    noise during testing.

### Partitions

Several of the parameters refer to the concept of a partition. We define a
partition as a portion of the data for which a single statistic will be
released. This is best explained through examples: if you're counting the number
of people in each of a set of age buckets, partitions would correspond to age
buckets. Or if you want to count the number of cars broken down by color, each
color would be a partition.

We imagine that you will use one or more `Algorithm`s for the data from each
partition. A single `Algorithm` should not be used for data from more than one
partition. If multiple `Algorithm`s are used for a single partition, we imagine
that serialization (described below) will be used to combine their data into a
single `Algorithm` and produce a single output.

## Use

### Adding data

These functions add data to the `Algorithm`'s internal pool. For most
algorithms, this doesn't consume additional space; the space consumed is
typically constant.

```
void AddEntry(const T& t);
```

Adds a single element `t` to the `Algorithm`'s pool. The type of `t` should be
the `Algorithm`'s templated type `T`.


```
template <typename Iterator>
void AddEntries(Iterator begin, Iterator end);
```

Adds multiple inputs to the algorithm.

*   `Iterator`: any iterator type. All algorithms support any
    [iterator category](http://en.cppreference.com/w/cpp/iterator#Iterator_categories),
    including input iterators.
*   `Iterator begin` and `Iterator end`: The begin and end iterators for your
    data. The `Algorithm` will behave like any STL iterator-based algorithm.

```
void Reset();
```

Clears the algorithm's input pool and allows a new result to be generated.

### Serialization

Since `Algorithm`s hold an internal state to represent all entires that have
been added, we can serialize the algorithm to a `Summary` proto. A `Summary`
proto holds all the information needed to reconstruct an `Algorithm` and its
internal state. We can merge a `Summary` into another `Algorithm` of the same
type that was constructed with identical parameters. Merging `Algorithm`s of
different types or with different parameters doesn't make sense, and will return
an error.

```
Summary Serialize();
absl::Status Merge(const Summary& summary);
```

Serialization and merging can allow these algorithms to be used in a distributed
manner. This could be useful for very large input sets, for example.

### Getting Results

```
Output PartialResult();
```

Get a result based on the current state of the `Algorithm`.

```
template <typename Iterator>
Output Result(Iterator begin, Iterator end)
```

Add the entries from `begin` to `end`, and then get the result.

Note that whichever method of getting a result you use, you can only get a
single result before your `epsilon` and `delta` are exhausted.

Values are returned from `Result` in an [`Output`](../protos.md) proto. For most
algorithms, this is a single `int64_t` or `double` value. Some algorithms
contain additional data about accuracy and algorithm mechanisms. You can use
[`GetValue<Type>`](../protos.md) to get values out of `Output`s easily.
