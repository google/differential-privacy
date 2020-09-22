# Confidence intervals in the Differential Privacy libraries

The **Differential Privacy (DP)** library supports confidence intervals to assess
the scale of the noise that has been added to a metric during the anonymization
process and narrow down its original, true value.

Given a noised metric **M** and a confidence level of **1 - alpha**, the DP library
computes a confidence interval **[L, R]** that contains the raw metric **m** (where **m**
is the metric after contribution bounding, but before applying noise)
with a probability of at least **1 - alpha**, i.e. **Pr[L ≤ m ≤ R] ≥ 1 - alpha**.

The computation is performed purely based on the noised metric **M** and on privacy parameters such as *epsilon*, *delta*, *sensitivities* and the *contribution bounds*. As a
result, no privacy budget is consumed for the computation of confidence intervals.

## Contribution bounding
Note that the confidence intervals provided by the library do not take the effects of the contribution bounding into account.

For instance, consider a *Bounded Sum* over where

*raw entries* = [1, 1, 2, 5, 14, 42, 132]

*lower bound* = 10, *upper bound* = 20

The *actual sum* for these entries is equal to 197, but the
*bounded sum* is equal to 94 (1, 2 and 5 are clamped to 10 and 43, 132 are clamped to 20).
So the confidence intervals provided by the library will be based on the *bounded
sum*, i.e. *m* = 94.

## Alpha

The DP libraries are using **alpha** to parameterize the confidence level **1 - alpha**.
The reason is that **alpha** provides more accuracy for confidence levels close to 1,
which is the parametrization we expect.
