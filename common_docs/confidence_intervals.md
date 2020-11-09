# Confidence intervals in the Differential Privacy libraries

The mechanisms of the DP library (including the Laplace mechanism, the Gaussian
mechanism, count, bounded sum and bounded mean) provide confidence intervals to
capture the scale of the noise they add to a metric during the anonymization
process.

Given a noised metric **M** and a confidence level **1 - alpha**, the mechanisms
return confidence intervals **[L, R]** containing the raw metric **m** (where
**m** is the value after contribution bounding, but before applying noise) with
a probability of at least **1 - alpha**, i.e., **Pr[L ≤ m ≤ R] ≥ 1 - alpha**.

A particular confidence interval is purely based on **M** and non-personal
parameters of the respective mechanism, such as epsilon, delta, sensitivities
and contribution bounds. In particular, its computation does not access the raw
metric **m** and consequently it also does not consume any privacy budget.

## Contribution bounding

The confidence intervals provided by the library do not account for discrepancy
due to contribution bounding.

For instance, consider a bounded sum over the raw entries [5, 5, 10, 20] with a
lower bound of 0 and an upper bound of 10. The true sum is 40, but the bounded
sum is 30. In this case, a confidence interval will contain **m = 30** with the
respective confidence level. No guarantees whether the interval also contains
the true sum of 40 are given.
