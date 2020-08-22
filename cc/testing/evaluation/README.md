# Evaluating Testing Tools for Differential Privacy

This evaluation framework assesses the relative effectiveness of two testing tools designed to detect differential privacy violations: the Stochastic Tester and the Statistical Tester. The Stochaster Tester is already available for testing in the C++ library. The Statistical Tester is currently only implemented in the Java library, yet its core methodology has been replicated using C++ differential privacy algorithms for the purposes of this evaluation framework.

Given that both testing tools seek to detect differential privacy violations, this framework deliberately constructs algorithms that *violate* differential privacy and proceeds to measure and compare each Testers' ability to identify those violations. The method by which differential privacy is violated, `insufficient_noise`, purposefully manipulates the epsilon value used to construct the algorithm. Under typical circumstances, the epsilon value determines the amount of noise applied to the data. Smaller epsilon values result in more noise, which leads to greater privacy protection. The same epsilon value is then used by testing tools to determine if the mathematical premise of differential privacy is satisfied.

In this framework, the epsilon value used to evaluate the algorithm for differential privacy violations, `advertised_epsilon`, *is always less than* the actual amount of noise applied to the algorithm, `implemented epsilon`. This deliberate mismatch ensures that the algorithm will *always* violate differential privacy, because it is claiming to have more privacy protection (i.e., noise) than the true amount of noise applied to the data. The relationship between the `advertised_epsilon` and the `implemented_epsilon` can be encapsulated by a third variable, `ratio`, where `ratio = advertised_epsilon / implemented epsilon.` For example, if `advertised_epsilon = 2` and `implemented_epsilon = 5`, then `ratio = 0.4`, meaning that only 40% of privacy protection that is claimed is actually being applied.

## How to Use

The evaluation framework consists of three distinct pieces, which must be run in the proper order.

### Testing the Stochastic Tester

The first stage constructs three algorithms with insufficient noise (Count, BoundedSum, and BoundedMean), and passes them into the Stochastic Tester for evaluation over a continuous range of ratio values. Given that the algorithms are known to violate differential privacy, the outcome for every test should be `false`, meaning that the algorithm *does not* satisfy differential privacy.

Note that the test contains several default parameters:

	`num_datasets = 15
	num_samples_per_histogram = 1000000
	ratio_min = 80.0
	ratio_max = 99.0
	output_filename = "stochastic_tester_results.txt"`

To run this test using all default parameters, simply run:

	`bazel build algorithms_with_insufficient_noise_test`

Users can also specify parameters in lieu of the default values. To run this test with user-supplied parameters, simply run:

	`bazel build algorithms_with_insufficient_noise_test [outputfilename] [ratio_min] [ratio_max] [num_datasets] [num_samples_per_histogram]`

This will result in an output text file containing results over all iterations of the test. Each line represents a single test run and contains the following data:

	`test_name`: Name of the test performed. All tests should implement the `insufficient_noise` test. More tests may be implemented in future versions of this framework.
	`algorithm`: The name of algorithm constructed (ie, Count, BoundedSum, or BoundedMean)
	`expected`: The expected outcome of the test. Since all algorithms violate DP, all outcomes should be `false` or 0.
	`actual`: The actual outcome of the test.
	`ratio`: The ratio value used in the test. 
	`num_datasets`: The number of datasets used by the Stochastic Tester.
	`num_samples`: The number samples used to build each histogram in the Stochastic Tester.
	`time(sec)`: The number of seconds the test took to run.

### Testing the Statistical Tester

The second stage constructs three algorithms with insufficient noise (Count, BoundedSum, and BoundedMean), and passes them into the Statistical Tester for evaluation over a continuous range of ratio values. Given that the algorithms are known to violate differential privacy, the outcome for every test should be `false`, meaning that the algorithm *does not* satisfy differential privacy.

To run the test, first create the samples needed for the Statistical Tester:

	`bazel build ...`

Then change into the X directory and run the Statistical Tester test:

	`bazel build ...`

This will result in an output text file containing results over all iterations of the test. Each line represents a single test run and contains the following data:

	`test_name`: Name of the test performed. All tests should implement the `insufficient_noise` test. More tests may be implemented in future versions of this framework.
	`algorithm`: The name of algorithm constructed (ie, Count, BoundedSum, or BoundedMean)
	`expected`: The expected outcome of the test. Since all algorithms violate DP, all outcomes should be `false` or 0.
	`actual`: The actual outcome of the test.
	`ratio`: The ratio value used in the test.
	`num_datasets`: The number of datasets used by the Stochastic Tester.
	`num_samples`: The number samples used to build each histogram in the Stochastic Tester.
	`time(sec)`: The number of seconds the test took to run.

### Summarizing and comparing results

The third and final stage of the framework summarizes and compares the performances of the Testers. 

Not quite sure what this is going to look like yet!