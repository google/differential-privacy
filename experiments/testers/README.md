# Evaluating Testing Tools for Differential Privacy

The purpose of this evaluation is to measure and compare the effectiveness of the two testing tools available to detect potential violations of differential privacy guarantees: the [DP Stochastic Tester](https://github.com/google/differential-privacy/tree/main/cc/testing) and the [DP Statistical Tester](https://github.com/google/differential-privacy/tree/main/java/tests/com/google/privacy/differentialprivacy/statistical).

## Approach

To assess the effectiveness of the Statistical and Stochastic Tester, a series of testing algorithms are constructed and passed to both tools. Each testing algorithm has been built to deliberately violate differential privacy. Given such circumstances, the Stochastic and Statistical Tester should both reject every algorithm as not differentially private. The goal of the evaluation is to measure and compare the number of algorithms that the testing tools correctly reject.

The testing algorithms have been engineered to violate differential privacy by manipulating epsilon. Under typical circumstances, the value of epsilon in a differentially private algorithm determines the amount of noise applied to the data. Smaller epsilons result in more noise, which leads to greater privacy protection. Larger epsilons result in less noise, which leads to less privacy protection. The Stochastic and Statistical Testers use epsilon (along with other parameters) to assess if the amount of noise that has been applied to the data is consistent with the mathematical definition of differential privacy.

However, if the testing tools evaluate a given algorithm using an epsilon that is smaller than the epsilon used to generate noise, then a differential privacy violation occurs. This is because `insufficient_noise` has been applied to the algorithm; in other words, the algorithm is claiming that it provides more privacy protection than it actually does.

For all algorithms with `insufficient_noise`, the `advertised_epsilon`, or the epsilon used to evaluate the algorithm for differential privacy, is always less than the `implemented_epsilon`, or the actual amount of noise applied to the algorithm. This deliberate mismatch ensures that the algorithm will always violate differential privacy, because it is always claiming to provide more privacy protection than the true amount of noise applied to the data.

The relationship between the `advertised_epsilon` and the `implemented_epsilon` can be represented by a third variable, `ratio`, where `ratio = advertised_epsilon / implemented epsilon.` For example, if `advertised_epsilon = 2` and `implemented_epsilon = 5`, then `ratio = 0.4`, meaning that only 40% of privacy protection that is claimed is actually being applied. (Note: We do not recommend using the aforementioned values of epsilon in practice. They were chosen only to clearly illustrate the meaning of `ratio`.)

## Architecture

Because the evaluation compares the performance of the Stochastic Tester (written in C++) to the Statistical Tester (written in Java), the program involves both C++ and Java code. 

Under typical circumstances, the Stochastic Tester evaluates differentially private algorithms constructed from the C++ library, while the Statistical Tester evaluates differentially private algorithms constructed from the Java library. However, to maintain a consistent environment for testing, both tools in this evaluation rely upon the differential privacy algorithms from the C++ library. For the Java-based Statistical Tester, the evaluation constructs the algorithms ahead of time and generates samples using the C++ library, then passes them into the Statistical Tester for assessment. 

The evaluation contains two folders: `cc` and `java`. The `cc` folder must be run first. The C++ files construct algorithms and generate samples from the C++ library and locate them in the `java` folder. Each set of samples has its own folder based on algorithm type (e.g., `countsamples`). Then the Stochastic Tester is run over a series of algorithms with `insufficient_noise`, and results are recorded in the `results` folder.

The `java` folder must be run second. The Java files read in the samples that were created in the C++ differential privacy library. Then the Stochastic Tester is run over a series of algorithms with `insufficient_noise`, and results are recorded in the `results` folder.

## How to Use

The evaluation is run in two parts. Part 1 is written in C++ and Part 2 is written in Java. **Part 1 must be run before Part 2.**

### Part 1: Creating Samples with Insufficient Noise and Testing the Stochastic Tester

The first stage of Part 1 generates samples for algorithms with `insufficient_noise` (e.g., Count, BoundedSum). Text files containing these samples are automatically created in the `java` folder, as they are evaluated by the Statistical Tester in Part 2.

The second stage of Part 1 runs the Stochastic Tester on algorithms with `insufficient_noise` (e.g., Count, BoundedSum) and records the results by algorithm type in the `results` folder. (There is no need to create or read in samples from text files when running the Stochastic Tester, because the Tester creates samples as part of its process.)

Note that the Stochastic Tester test contains several default parameters:
```
num_samples_per_histogram = 1000000
ratio_min = 50.0
ratio_max = 99.0
output_filename = "stochastic_tester_results_[algorithm_type].txt"
```

These parameters are mutable, and users can specify their own on the command line.

To run Part 1 using all default parameters, run:

```
bazel build part1
bazel-bin/part1
```

Users can also specify parameters to override the default values. To run this test with user-supplied parameters, run:

```
bazel build part1
bazel-bin/part1 [count_results_filename] [sum_results_filename] [mean_results_filename] [ratio_min] [ratio_max] [num_samples_per_histogram]
```

### Part 2: Testing the Statistical Tester

Part2 runs the Statistical Tester on algorithms with `insufficient noise` (e.g., Count, BoundedSum).

Note that the Statistical Tester contains the same default parameters with the same default values as the Stochastic Tester. These parameters are mutable, and users can specify their own on the command line. However, they must be equal to the comparable parameters specified in the Stochastic Tester test.

To run Part 2 using all default parameters, run:
```
bazel build part2
bazel-bin/part2
```

To run this test with user-supplied parameters, run:
```
bazel build part2
bazel build part2 [ratio_min] [ratio_max] [number_of_samples_per_histogram] [count_results_filename] [sum_results_filename] [mean_results_filename]
```

### Part 3: Evaluating the Results

Running Part1 and Part2 will result in several output files, located in the `results` folder, which contain outcomes from both Testers over all algorithm types. The files are separated by algorithm type (e.g., Count, BoundedSum, BoundedMean). Each line represents a single test run and contains the following data:

```
test_name: Name of the test performed. All tests should implement the insufficient_noise test. More tests may be implemented in future versions of this framework.
algorithm_type: The name of algorithm constructed (e.g., Count, BoundedSum)
expected: The expected outcome of the test. Since all algorithms violate DP, all outcomes should be `false` or 0.
actual: The actual outcome of the test.
ratio: The ratio value used in the test.
num_datasets: The number of algorithms created with distinct parameters.
num_samples: The number samples used to build each histogram.
time(sec): The number of seconds the test took to run. [Note: This is currently null for the Statistical Tester tests]
```

Take a look at the performances of the Stochastic and Statistical Testers and see which Tester performed best!


