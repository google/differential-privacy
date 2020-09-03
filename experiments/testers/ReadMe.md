# Evaluating Testing Tools for Differential Privacy

The purpose of this evaluation is to measure and compare the effectiveness of the two testing tools available to detect potential diferential privacy violations: the [DP Stochastic Tester](https://github.com/google/differential-privacy/tree/main/cc/testing) and the [DP Statistical Tester](https://github.com/google/differential-privacy/tree/main/java/tests/com/google/privacy/differentialprivacy/statistical).

## Approach

To assess the effectiveness of the Statistical and Stochastic Tester, a series of testing algorithms are constructed and passed to both tools. Each testing algorithm has been built to deliberately violate differential privacy. Given such circumstances, the Stochastic and Statistical Tester should both reject every algorithm as not differentially private. The goal of the evaluation is to measure and compare the number of algorithms that the testing tools correctly reject.

The testing algorithms have been engineered to violate differential privacy by manipulating epsilon. Under typical circumstances, the value of epsilon in a differentially private algorithm determines the amount of noise applied to the data. Smaller epsilons result in more noise, which leads to greater privacy protection. Larger epsilons result in less noise, which leads to less privacy protection. The Stochastic and Statistical Testers use epsilon (along with other parameters) to assess if the amount of noise that has been applied to the data is consistent with the mathematical definition of differential privacy.

However, if the testing tools evaluate a given algorithm using an epsilon that is smaller than the epsilon used to generate noise, then a differential privacy violation occurs. This is because `insufficient_noise` has been applied to the algorithm; in other words, the algorithm is claiming more privacy protection than it actually provides.

For all algorithms with insufficient_noise, the advertised_epsilon, or the epsilon used to evaluate the algorithm for differential privacy, is always less than the implemented_epsilon, or the actual amount of noise applied to the algorithm. This deliberate mismatch ensures that the algorithm will always violate differential privacy, because it is always claiming to have more privacy protection than the true amount of noise applied to the data.

The relationship between the advertised_epsilon and the implemented_epsilon can be represented by a third variable, ratio, where ratio = advertised_epsilon / implemented epsilon. For example, if advertised_epsilon = 2 and implemented_epsilon = 5, then ratio = 0.4, meaning that only 40% of privacy protection that is claimed is actually being applied.

## Architecture

The evaluation involves running both C++ and Java files. This is because the purpose of the evaluation is to compare the Stochastic Tester, which is written in C++, to the Statistical Tester, which is written in Java.

Under typical circumstances, the Stochastic Tester evaluates differentially private algorithms constructed from the C++ library, while the Statistical Tester evaluates differentially private algorithms constructed from the Java library. However, to maintain a consistent environment for testing, both tools in this evaluation rely upon the differential privacy algorithms from the C++ library. For the Java-based Statistical Tester, the evaluation constructs the algorithms ahead of time and generates samples using the C++ library, then passes them into the Statistical Tester for assessment. 

The evaluation contains two folders: `cc` and `java`. The `cc` folder must be run first. The C++ files construct algorithms and generate samples from the C++ library and locate them in the `java` folder. Each set of samples has its own folder based on algorithm type (e.g., `countsamples`). Then the Stochastic Tester is run over a series of algorithms with `insufficient_noise`, and results are recorded in the `results` folder.

The `java` folder must be run second. The Java files read in the samples that were created in the C++ differential privacy library. Then the Stochastic Tester is run over a series of algorithms with `insufficient_noise`, and results are recorded in the `results` folder.

## How to Use