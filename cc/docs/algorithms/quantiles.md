
# Quantiles

We have two different pieces of code to calculate quantiles (aka percentiles, or
order statistics):
[QuantileTree](https://github.com/google/differential-privacy/blob/main/cc/algorithms/quantile-tree.h)
offers a tree-based differentially private algorithm with a distinctive
interface, and
[Quantiles](https://github.com/google/differential-privacy/blob/main/cc/algorithms/quantiles.h)
uses it to implement the `Algorithm` interface. Both offer the same accuracy,
performance, and privacy guarantees because they use the same underlying DP
mechanism. Both can be used to calculate any quantile (though they are least
accurate close to the maximum and minimum), and both can be used to calculate
any number of quantiles with no loss in accuracy (e.g. if you want to calculate
a median, your result will be equally accurate regardless of whether or not you
choose to calculate additional quantiles). The only difference between the two
is in their interface.

Note: If you only want to calculate a single quantile, we recommend using
`Quantiles` and requesting only a single quantile. We do not currently have a
more efficient algorithm for single quantiles. If we add a more efficient
algorithm for single quantiles, we will use it whenever a `Quantiles` is created
to find a single quantile.

## Quantiles

### Input & Output

`Quantiles` support any numeric type. Its `Output`s contain one element for each
requested quantile, in the same order as when the `Quantiles` was built.
`ConfidenceInterval` and `BoundingReport` are not provided.

### Construction

`Quantiles` is an [`Algorithm`](algorithm.md). It does require an upper and
lower bound, but it does not use [BoundedAlgorithmBuilder](bounded-algorithm.md)
and cannot determine the bounds automatically. Instead, the user must manually
set the bounds or the algorithm will fail to build.

Note: You can run [`ApproxBounds`](approx-bounds.md) over your dataset to get
bounds if you do not know them in advance, but you must do so manually in a
separate pass over your data (before you construct your `Quantiles`) and you
must manually pass the bounds from your `ApproxBounds` to your `Quantiles`.

In addition, you must specify a set of quantiles to calculate. Quantiles are
provided as a `vector<double>`. Quantiles will be returned in the same order as
you specify them. If you call `SetQuantiles` multiple times, each call will
overwrite the set of quantiles from the previous call.

Here is the minimal set of arguments for constructing a Quantiles:

```
absl::StatusOr<std::unique_ptr<Quantiles<T>>> quantile =
   Quantiles<T>::Builder()
       .SetLower(upper)
       .SetUpper(lower)
       .SetQuantiles(quantiles)
       .Build();
```

*   `T`: The input type, for example `double` or `int64_t`.
*   `T upper, lower`: The upper and lower bounds on each input element. If any
    input elements are greater than `upper` or less than `lower`, they will be
    replaced with `upper` or `lower` respectively.
*   `std::vector<double> quantiles`: The list of quantiles you wish to
    calculate. Each quantile should be in the range [0, 1] where 0 will find the
    minimum, and 1 the maximum. Can be created inline, e.g. `.SetQuantile({0.25,
    0.5, 0.75})`.

### Use

`Quantiles` is an [`Algorithm`](algorithm.md) and supports its full API, except
for the ability to provide confidence intervals.

## QuantileTree

QuantileTree implements a tree-based differentially private quantile calculation
algorithm. For details, see the
[full algorithm writeup](https://github.com/google/differential-privacy/blob/main/common_docs/Differentially_Private_Quantile_Trees.pdf).

### Construction

A `QuantileTree` is constructed via a `QuantileTree<T>::Builder`. The builder
supports the following methods:

```
absl::StatusOr<std::unique_ptr<QuantileTree<T>>> quantile_tree =
  QuantileTree<T>::Builder()
      .SetLower(lower)
      .SetUpper(upper)
      .SetTreeHeight(tree_height) // optional
      .SetBranchingFactor(branching_factor)
      .Build();
```

*   `T`: The input type, for example `double` or `int64_t`.
*   `T upper, lower`: The upper and lower bounds for each input element. If any
    inputs are greater than `upper` or less than `lower` they will be replaced
    with `upper` or `lower` respectively.
*   `int tree_height, branching_factor`: These parameters specify the height and
    width of the quantile tree. Each node will have `branching_factor` children,
    and the tree will be of height `tree_height`. The quantile tree assigns each
    leaf node to an equal sized portion of the domain [lower, upper], and there
    will be `branching_factor` ^ `tree_height` leaf nodes. These parameters are
    optional, and have reasonable default values.

Note that privacy parameters are not specified when constructing the quantile
tree.

### Input and Output

Like `Quantiles`, `QuantileTree` supports any numeric type as input.

To get results, call the `MakePrivate` method. The `MakePrivate` method takes a
struct that contains all of the privacy parameters (these are the same ones used
when constructing an [`Algorithm`](algorithm.md)):

```
absl::StatusOr<QuantileTree<T>::Privatized> results =
  quantile_tree.MakePrivate({
      .epsilon = epsilon,
      .delta = delta,
      .max_contributions_per_partition = max_contributions,
      .max_partitions_contributed_to = max_partitions,
      .mechanism_builder = absl::MakeUnique<LaplaceMechanism::Builder>()});
```

*   `double epsilon`: The `epsilon` differential privacy parameter. A smaller
    number means more privacy but less accuracy. `epsilon` should be > 0.
*   `double delta`: The `delta` differential privacy parameter. A smaller number
    means more privacy but less accuracy. `delta` should be in (0, 1).
*   `int max_partitions`: The number of aggregations, or 'partitions,' that each
    user is allowed to contribute to. Defaults to 1 if unset. The caller must
    guarantee that this limit is enforced on the input. The library cannot
    enforce it because it cannot distinguish between users or aggregations. Note
    that `Algorithm`s that will be merged together are considered part of the
    same partition.
*   `int max_contributions`: The number of pieces of input to this aggregation
    that can belong to a single user. Defaults to 1 if unset. The caller must
    guarantee that this limit is enforced on the input. The library cannot
    enforce it because it does not now which inputs belong to which users. If
    summaries from multiple `Algorithm`s are merged together, the total number
    of inputs from a single user across all marged `Algorithm`s must not exceed
    this limit.
*   `std::unique_ptr<NumericalMechanism::Builder> mechanism_builder`: Used to
    specify the type of numerical mechanism the algorithm will use to add noise
    (e.g. Laplace, Gaussian). In most cases this should not be set (and a
    default LaplaceMechanism will be used), but it can be used to remove or mock
    noise during testing.

`MakePrivate` returns a `QuantileTree<T>::Privatized` (or an error status if an
error occurred). A `QuantileTree<T>::Privatized` can be used to calculate any
quantile by calling the `GetQuantile` method. Calling `GetQuantile` consumes no
additional privacy budget. A `QuantileTree<T>::Privatized` may contain internal
non-privatized information, but only privatized information can be accessed
through its API. A `QuantileTree<T>::Privatized` answers questions about the
input that had been added to the `QuantileTree<T>` at the time that
`MakePrivate` was called. After a `QuantileTree<T>::Privatized` has been created
it is no longer tied to the `QuantileTree<T>`, and modifying the
`QuantileTree<T>` has no effect on it.

## Result Performance

For both `Quantiles` and `QuantileTree`, calling `Result` has a constant time
complexity (though the value of the constant will depend on the tree
parameters). Space complexity is also constant, and again depends on the tree
parameters.
