# Partition Selection (Truncated Geometric Thresholding)

Many data analysis operations can be expressed as a `GROUP BY` query on an
unbounded set of partitions, followed by a per-partition aggregation. To make
such a query differentially private, adding noise to each aggregation is not
enough: we also need to make sure that the set of partitions released is itself
differentially private.

For the common case where each user contributes a record to exactly one
partition, an optimal $$(\varepsilon,\delta)$$-differentially private mechanism
for publishing or dropping partitions is described in
https://arxiv.org/abs/2006.03684. In the paper, it is also shown that the
optimal mechanism can be closely approximated by adding noise drawn from a
truncated geometric distribution to the raw count of unique users in a
partition, and then thresholding the noisy count. This library includes
implementations of the mechanism in
[C++](https://github.com/google/differential-privacy/blob/main/cc/algorithms/partition-selection.h)
and
[Go](https://github.com/google/differential-privacy/blob/main/go/dpagg/select_partition.go).
A simple Python implementation and visualization is also provided in this
[Google Colab notebook](https://colab.research.google.com/github/google/differential-privacy/blob/main/common_docs/partition_selection_playground.ipynb).

In addition to the raw algorithm,
[Privacy-On-Beam](https://github.com/google/differential-privacy/tree/main/privacy-on-beam)
wraps partition selection as one component of an end-to-end differentially
private data pipeline. Privacy-On-Beam also guarantees differential privacy in
the case when users contribute to multiple partitions by splitting the privacy
budget across these contributions. However, this is not guaranteed to be optimal
in the sense of maximizing the number of partitions reported.
