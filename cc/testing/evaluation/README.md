# Evaluating Testing Tools for Differential Privacy

This evaluation framework assesses and compares the effectiveness of two testing tools designed to detect differential privacy violations: the [DP Stochastic Tester](https://github.com/google/differential-privacy/tree/main/cc/testing) and the [DP Statistical Tester](https://github.com/google/differential-privacy/tree/main/java/tests/com/google/privacy/differentialprivacy/statistical). 

## Overview

Under typical circumstances, the epsilon parameter in a differentially private algorithm determines the amount of noise applied to the data. Smaller epsilon values result in more noise, which leads to greater privacy protection. This same epsilon value is also used by both testing tools to evaluate if a given algorithm violates the mathematical premise of differential privacy.

The following framework works by constructing algorithms that *deliberately violate* differential privacy. This allows users to measure and compare each Testers' ability to identify known violations.

We employ an approach we will call the `insufficient_noise` test to construct differential privacy-violating algorithms. For all algorithms with `insufficient_noise`, the `advertised_epsilon`, or the epsilon value used to evaluate the algorithm for differential privacy violations, *is always less than* the `implemented_epsilon`, or the actual amount of noise applied to the algorithm. This deliberate mismatch ensures that the algorithm will *always* violate differential privacy, because it is always claiming to have more privacy protection than the true amount of noise applied to the data.

The relationship between the `advertised_epsilon` and the `implemented_epsilon` can be represented by a third variable, `ratio`, where `ratio = advertised_epsilon / implemented epsilon.` For example, if `advertised_epsilon = 2` and `implemented_epsilon = 5`, then `ratio = 0.4`, meaning that only 40% of privacy protection that is claimed is actually being applied.

## How to Use

The evaluation framework consists of three distinct pieces, which must be run in the proper order.

### Testing the Stochastic Tester

The first stage runs the Stochastic Tester on three algorithms with `insufficient noise` (Count, BoundedSum, and BoundedMean) over a continuous range of ratio values. Given that the algorithms are known to violate differential privacy, the desired outcome for every test is `false`, meaning that the algorithm *does not* satisfy differential privacy.

Note that the test contains several default parameters:

	num_datasets = 15
	num_samples_per_histogram = 1000000
	ratio_min = 80.0
	ratio_max = 99.0
	output_filename = "stochastic_tester_results.txt"

To run this test using all default parameters, simply run:

	bazel build algorithms_with_insufficient_noise_test

Users can also specify parameters in lieu of the default values. To run this test with user-supplied parameters, simply run:

	bazel build algorithms_with_insufficient_noise_test [outputfilename] [ratio_min] [ratio_max] [num_datasets] [num_samples_per_histogram]

This will result in an output text file containing results over all iterations of the test. Each line represents a single test run and contains the following data:

	test_name: Name of the test performed. All tests should implement the insufficient_noise test. More tests may be implemented in future versions of this framework.
	algorithm: The name of algorithm constructed (ie, Count, BoundedSum, or BoundedMean)
	expected: The desired outcome of the test. Since all algorithms violate DP, all outcomes should be false or 0.
	actual: The actual outcome of the test.
	ratio: The ratio value used in the test. 
	num_datasets: The number of datasets used by the Stochastic Tester.
	num_samples`: The number samples used to build each histogram in the Stochastic Tester.
	time(sec): The number of seconds the test took to run.

### Testing the Statistical Tester

The second stage runs the Statistical Tester on three algorithms with `insufficient noise` (Count, BoundedSum, and BoundedMean) over a continuous range of ratio values. Given that the algorithms are known to violate differential privacy, the desired outcome for every test should be `false`, meaning that the algorithm *does not* satisfy differential privacy.

To run the test, first create the samples needed for the Statistical Tester [KR Note: Need to figure out how to run all three files simultaneously in this build call]:

	bazel build ...

Then change into the X directory and run the Statistical Tester test:

	bazel build ...

This will result in an output text file containing results over all iterations of the test. Each line represents a single test run and contains the following data:

	test_name: Name of the test performed. All tests should implement the `insufficient_noise` test. More tests may be implemented in future versions of this framework.
	algorithm: The name of algorithm constructed (ie, Count, BoundedSum, or BoundedMean)
	expected: The expected outcome of the test. Since all algorithms violate DP, all outcomes should be `false` or 0.
	actual: The actual outcome of the test.
	ratio: The ratio value used in the test.
	num_datasets: The number of datasets used by the Stochastic Tester.
	num_samples: The number samples used to build each histogram in the Stochastic Tester.
	time(sec): The number of seconds the test took to run.

### Summarizing and comparing results

The third and final stage of the framework summarizes and compares the performances of the Testers. 

Not quite sure what this is going to look like yet!