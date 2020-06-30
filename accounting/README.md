# Privacy Accounting

This directory contains tools for tracking privacy budgets. Currently, it
provides an implementation of Privacy Loss Distributions (PLDs) which can
help compute an accurate estimate of the total ε, δ across multiple executions
of differentially private queries. Our implementation currently supports Laplace
Mechanism, Gaussian Mechanism and Randomized Response. A supplementary material
with more detailed definitions and references can be found
[here](./docs/Privacy_Loss_Distributions.pdf).
