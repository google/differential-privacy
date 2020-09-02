# Evaluating Testing Tools for Differential Privacy

The purpose of this evaluation is to measure and compare the effectiveness of the two testing tools available to detect potential diferential privacy violations: the [DP Stochastic Tester](https://github.com/google/differential-privacy/tree/main/cc/testing) and the [DP Statistical Tester](https://github.com/google/differential-privacy/tree/main/java/tests/com/google/privacy/differentialprivacy/statistical). 

## Approach

To assess the effectiveness of the Statistical and Stochastic Tester, a series of testing algorithms are constructed and passed to both tools. Each testing algorithm has been built to deliberately violate differential privacy. Given such circumstances, the Stochastic and Statistical Tester should, in theory, reject every algorithm as not differentially private. This setup enables the Testers' ability to identify known DP violations to be measured and compared.

The testing algorithms have been engineered to violate differential privacy by manipulating epsilon. Under typical circumstances, epsilon in a differentially private algorithm determines the amount of noise applied to the data. Smaller epsilons result in more noise, which leads to greater privacy protection. Larger epsilons result in less noise, which leads to less privacy protection. The testing tools use epsilon (along with other parameters) to assess if the amount of noise that has been applied to the data is consistent with the mathematical definition of differential privacy.

However, if the testing tools evaluate a given algorithm using an epsilon that is smaller than the epsilon used to generate noise, then a differential privacy violation occurs. This is because `insufficient_noise` has been applied to the algorithm; in other words, the algorithm is claiming more privacy protection than it actually provides.

For all algorithms with `insufficient_noise`, the `advertised_epsilon`, or the epsilon used to evaluate the algorithm for differential privacy, *is always less than* the `implemented_epsilon`, or the actual amount of noise applied to the algorithm. This deliberate mismatch ensures that the algorithm will *always* violate differential privacy, because it is always claiming to have more privacy protection than the true amount of noise applied to the data.

The relationship between the `advertised_epsilon` and the `implemented_epsilon` can be represented by a third variable, `ratio`, where `ratio = advertised_epsilon / implemented epsilon.` For example, if `advertised_epsilon = 2` and `implemented_epsilon = 5`, then `ratio = 0.4`, meaning that only 40% of privacy protection that is claimed is actually being applied.

## How to Use

The evaluation is run in two parts. Part 1 is written in C++ and Part 2 is written in Java. Part 1 must be run before Part 2.

### Part 1

#### Testing the Stochastic Tester

The first stage of Part 1 runs the Stochastic Tester on three algorithms with `insufficient noise` (Count, BoundedSum, and BoundedMean) over a continuous range of ratio values. Given that the algorithms are known to violate differential privacy, the desired outcome for every test is `false`, meaning that the algorithm *does not* satisfy differential privacy. 

Note that the Stochastic Tester test contains several default parameters:

	num_samples_per_histogram = 100
	ratio_min = 80.0
	ratio_max = 85.0
	output_filename = "stochastic_tester_results_[algorithm_type].txt"

These parameters are mutable, and users can specify their own on the command line.

Running the Stochastic Tester on the `insufficient_noise` algorithms will result in three output text files that appear in the `results` folder and contain outcomes from all iterations of the test. The files are separated by algorithm type (e.g., Count, BoundedSum, or BoundedMean). Each line represents a single test run and contains the following data:

	test_name: Name of the test performed. All tests should implement the insufficient_noise test. More tests may be implemented in future versions of this framework.
	algorithm: The name of algorithm constructed (ie, Count, BoundedSum, or BoundedMean)
	expected: The desired outcome of the test. Since all algorithms violate DP, all outcomes should be false or 0.
	actual: The actual outcome of the test.
	ratio: The ratio value used in the test. 
	num_datasets: The number of datasets used by the Stochastic Tester.
	num_samples`: The number samples used to build each histogram in the Stochastic Tester.
	time(sec): The number of seconds the test took to run output from the tests appear in the results folder.

#### Creating Samples with Insufficient Noise

The second stage of Part 1 generates samples for three algorithms with `insufficient_noise` (Count, BoundedSum, and BoundedMean), to be used when testing the Statistical Tester. The samples are automatically created in the `statisticaltester` folder. Note that the `ratio_min` and `ratio_max` parameters also apply to the Statistical Tester test, as both testing tools should be evaluated over the same range of ratio values. 

To run Part 1 using all default parameters, simply run:

	bazel build stochastic_tester

Users can also specify parameters in lieu of the default values. To run this test with user-supplied parameters, simply run:

	bazel build stochastic_tester [count_filename] [sum_filename] [mean_filename] [ratio_min] [ratio_max] [num_samples_per_histogram]

### Testing the Statistical Tester

The second stage runs the Statistical Tester on three algorithms with `insufficient noise` (Count, BoundedSum, and BoundedMean) over a continuous range of ratio values. Given that the algorithms are known to violate differential privacy, the desired outcome for every test should be `false`, meaning that the algorithm *does not* satisfy differential privacy.

Note that the Statistical Tester test contains several default parameters:

	num_samples_per_histogram = 100
	ratio_min = 80.0
	ratio_max = 85.0
	output_filename = "statistical_tester_results_[algorithm_type].txt"

These parameters are mutable, and users can specify their own on the command line. However, they must be equal to the comparable parameters specified in the Stochastic Tester test.

Running the Statistical Tester on the `insufficient_noise` samples will result in three output text files that appear in the `results` folder and contain outcomes from all iterations of the test. The files are separated by algorithm type (e.g., Count, BoundedSum, or BoundedMean). Each line represents a single test run and contains the following data:

	test_name: Name of the test performed. All tests should implement the `insufficient_noise` test. More tests may be implemented in future versions of this framework.
	algorithm: The name of algorithm constructed (ie, Count, BoundedSum, or BoundedMean)
	expected: The expected outcome of the test. Since all algorithms violate DP, all outcomes should be `false` or 0.
	actual: The actual outcome of the test.
	ratio: The ratio value used in the test.
	num_datasets: The number of algorithms created with distinct parameters.
	num_samples: The number samples used to build each histogram.
	time(sec): The number of seconds the test took to run. [Note: This is currently null]

To run Part 2 using all default parameters, simply run:

	bazel build statistical_tester

Users can also specify parameters in lieu of the default values. To run this test with user-supplied parameters, simply run:

	bazel build statistical_tester [ratio_min] [ratio_max] [number_of_samples_per_histogram] [count_filename] [sum_filename] [mean_filename]

Once both build commands have been run, take a look at the performances of the Stochastic and Statistical Testers in the `results` folder and see which Tester performed best!