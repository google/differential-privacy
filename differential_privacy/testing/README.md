# DP Stochastic Tester

This is a framework that attempts to falsify the DP predicate for a given
algorithm *F* over a set of datasets. For a given dataset *D*, we check the
predicate by considering all adjacent datasets in the powerset of *D*,
generating the probability distributions of the over the output of algorithm
*F*. To give us more confidence that an algorithm is DP, the typical usage is to
choose a diverse set of datasets. Note that since this problem is semidecidable,
this method does not prove that an algorithm is DP, but can determine if an
algorithm is not DP.

See testing/stochastic_tester_test.cc for typical usage examples.
