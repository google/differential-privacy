# DP Stochastic Tester

This is a framework that attempts to falsify the DP predicate for a given
algorithm *F* over a set of datasets. For a given dataset *D*, we check the
predicate by considering all adjacent datasets in the powerset of *D*,
generating the probability distributions of the over the output of algorithm
*F*. To give us more confidence that an algorithm is DP, the typical usage is to
choose a diverse set of datasets. Note that since this problem is semidecidable,
this method does not prove that an algorithm is DP, but can determine if an
algorithm is not DP. Details can be found in section 5.3 of
[our paper](https://arxiv.org/abs/1909.01917).

## How to Use

stochastic_tester_test.cc contains typical usage examples. To run the tests, use

```
cd cc
bazel test testing:stochastic_tester_test
```

We also run through a simple example here. First, create a Halton sequence.

```
auto sequence = absl::make_unique<HaltonSequence<double>>(
      /*dimension=*/3, /*sorted_only=*/true, /*scale=*/1, /*offset=*/.5);

```

Here `sequence` is an object that can be used to generate a determinisitic
sequence of uniform random input sets for our algorithms that are spread out
"evenly". The `dimension` is the size of the input. The `scale` and `offset`
above imply that the input points are in the range `[-0.5, 0.5]`. Next, create
the algorithm you want to test.

```
std::unique_ptr<Count<double>> algorithm =
      Count<double>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(.1)
          .Build()
          .ValueOrDie();
```

To avoid flakiness in our test, we use the deterministic
`SeededLaplaceMechanism`. Finally, create an instance of the tester.

```
StochasticTester<double, int64> tester(
    std::move(algorithm), std::move(sequence),
    /*num_datasets=*/500, /*num_samples_per_histogram=*/20000);
```

The `num_datasets` specifies the number of inputs sets we want to check for
whether they violate the DP predicate. For each input set, the
`num_samples_per_histogram` specifies how many runs of the algorithm the tester
will do to generate output distribution histograms. The higher these numbers,
the higher the likelihood to catch any DP violations should they exist. However,
increasing the numbers also makes the test slower. Finally, check the status of
running the tester.

```
bool test_passes = tester.Run();
```

If the result is true, then the tester didn't detect a DP violation. Otherwise,
the tester will log additional output to help your debugging.
