# Partition Selection with pre-thresholding

Pre-thresholding is a technique in differentially private partition selection
that guarantees that some minimum number of privacy units, n, contribute to
every partition in an anonymized dataset. This ensures a partition with fewer
than n privacy units is never released. For partitions with n or more
contributing privacy units, the probability of releasing the partition increases
with the number of privacy units.

This feature combines the thresholding capability of k-anonymity together with
the guarantees of $$(\varepsilon,\delta)$$-differential privacy. A minimum
threshold may, for instance, be suitable as an extra layer of protection in
cases where the partitions themselves are inherently sensitive.

This library includes an implementation of this technique in
[Java](https://github.com/google/differential-privacy/blob/main/java/main/com/google/privacy/differentialprivacy/PreAggSelectPartition.java)
and in
[C++](https://github.com/google/differential-privacy/blob/main/cc/algorithms/partition-selection.h).
