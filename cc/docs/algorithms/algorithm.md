
# Algorithm

[`Algorithm`](https://github.com/google/differential-privacy/blob/main/cc/algorithms/algorithm.h)
is the base class of all
differentially private algorithms. Each algorithm can be constructed using the
builder pattern, which sets its parameters. `Algorithms` are incremental: it's
possible to insert some data and get a result, then _add more data_ and get a
new result. (It's also possible to add data in chunks but still only get one
result). Algorithms are templated on the input type.

## Privacy budget

Every time you extract a result from an `Algorithm`, you use some "privacy
budget". This can be thought of as a "fraction of your epsilon." Each algorithm
starts with a privacy budget of `1`, and reading uses up that budget.

## Construction

```
util::StatusOr<std::unique_ptr<Algorithm>> algorithm =
 AlgorithmBuilder.SetEpsilon(double epsilon)
                 .SetLaplaceMechanism(std::unique_ptr<LaplaceMechanism::Builder> laplace_mechanism_builder)
                 .Build();
```

*   `double epsilon`: The `epsilon` differential privacy parameter. A smaller
    number means more privacy but less accuracy.
*   `std::unique_ptr<LaplaceMechanism::Builder> laplace_mechanism_builder`: Used
    to specify the type of laplace mechanism the algorithm will use to add
    noise. In most cases they should not be set (and a default LaplaceMechanism
    will be used), but it can be used to remove or mock noise during testing.

## Use

### Adding data

These functions add data to the `Algorithm`'s internal pool. For most
algorithms, this doesn't consume additional space; the space consumed is
typically constant. The exception is the order
statistics algorithms: the space consumed is linear on the number of inputs.

```
void AddEntry(const T& t);
```

Adds a single element `t` to the `Algorithm`'s pool. The type of `t` should be
the `Algorithm`'s templated type `T`.

```
void Reset();
```

Clears the algorithm's input pool, and sets your remaining privacy budget back
to `1.0`.

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

### Serialization

Since `Algorithm`s can hold an internal state as a result of added entries, we
can serialize the algorithm to a `Summary` proto. A `Summary` proto holds all
the information needed to reconstruct an `Algorithm` and its internal state. We
can merge a `Summary` into another `Algorithm` of the same type and that was
constructed with identical parameters. Merging `Algorithm`s of different types
or with different parameters doesn't make sense, and will return an error.

```
Summary Serialize();
util::Status Merge(const Summary& summary);
```

Serialization and merging can be used to run these algorithms in a distributed
manner. This could be useful for very large input sets, for example.

### Getting Results

```
Output PartialResult(double privacy_budget = RemainingPrivacyBudget());
```

Get a result based on the current state of the `Algorithm`. The `privacy_budget`
is the amount of your remaining privacy budget to consume. You must have at
least that much budget remaining. If unspecified, consumes all remaining budget.

```
template <typename Iterator>
Output Result(Iterator begin, Iterator end)
```

Add the entries from `begin` to `end`, and then get the result with the full
remaining privacy budget.

Values are returned from `Result` in an [`Output`](../protos.md) proto. For most
algorithms, this is a single `int64` or `double` value. Some algorithma contain
additional data about accuracy and algorithm mechanisms. You can use
[`GetValue<Type>`](../protos.md) to get values out of `Output`s easily.
