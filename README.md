# Differential Privacy

> **NEW:**
> [Join our DP community in Slack](https://join.slack.com/t/dp-open-source/shared_invite/zt-35hw483tz-nS5YOtGjxCHk3Ek7WiXvlg)!

This repository contains libraries to generate ε- and (ε, δ)-differentially
private (DP) statistics over datasets. It contains the following tools:

*   [Privacy on Beam](privacy-on-beam) is an end-to-end differential privacy
    framework for Go built on top of
    [Apache Beam](https://beam.apache.org/documentation/). It is intended to be
    easy to use, even by non-experts.
*   [PipelineDP4j](pipelinedp4j) is an end-to-end differential privacy framework
    for JVM languages (Java, Kotlin, Scala). It supports different data
    processing frameworks such as
    [Apache Beam](https://beam.apache.org/documentation/) and
    [Apache Spark](https://spark.apache.org/). It is intended to be easy to use,
    even by non-experts.
*   Three "DP building block" libraries, in [C++](cc), [Go](go), and
    [Java](java). These libraries implement basic noise addition primitives and
    differentially private aggregations. Privacy on Beam and PipelineDP4j use
    these libraries.
*   A [stochastic tester](cc/testing), used to help catch regressions that could
    make the differential privacy property no longer hold.
*   A [differential privacy accounting library](python/dp_accounting), used for
    tracking privacy budget.
*   A [command line interface](examples/zetasql) for running differentially
    private SQL queries with [ZetaSQL](https://github.com/google/zetasql).
*   [DP Auditorium](python/dp_auditorium) is a library for auditing differential
    privacy guarantees.

In addition to the tools listed above, it is worth mentioning two related
projects developed by [OpenMined](https://www.openmined.org/) that make use of
our libraries:

*   [PipelineDP](https://pipelinedp.io/) is an end-to-end differential privacy
    framework for Python. It is the Python version of PipelineDP4j and is a
    collaboration between Google and OpenMined. Its source code is located in
    the [OpenMined repository](https://github.com/OpenMined/PipelineDP).
*   [PyDP](https://github.com/OpenMined/PyDP) is a Python wrapper of our C++ DP
    building block library.

The DP building block libraries, Privacy on Beam, PipelineDP4j and PipelineDP
are suitable for research, experimental, or production use cases, while the
other tools are currently experimental and subject to change.

## Getting Started

If you are new to differential privacy, you might want to go through
["A friendly, non-technical introduction to differential privacy"](https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html).
Understanding the basics is helpful, even when using our high-level tools like
Privacy on Beam and PipelineDP4j. If you plan to use more low-level libraries
such as DP building block libraries or other experimental tools, you might need
a more in-depth understanding of differential privacy. You can take a look at
[a comprehensive guide for programmers by Joseph P. Near and Chiké Abuah](https://programming-dp.com/cover.html)
or
[other blog posts at Damien Desfontaines blog](https://desfontain.es/privacy/archives.html).

### Explore the tools

All tools, except the DP Building block libraries, have their documentation in
their respective directories. For example, Privacy on Beam has README in
[the privacy-on-beam directory](privacy-on-beam), same for
[the pipelinedp4j directory](pipelinedp4j).

The high-level documentation for DP Building block libraries follows below
because they share a lot of commonalities. The language-specific documentation
can be found in respective directories.

There is also an ["examples"](examples) directory where you can find examples of
how to use the tools and libraries. The documentation in the tooling directories
refers to these examples.

### How to Build

To build the tools and libraries, follow the instructions in their respective
directories. The build process assumes you have cloned the Git repository. Most
tools and libraries use [Bazel](https://bazel.build/) as a build system, see
instructions below to install it.

#### Bazel

To use Bazel, you need to install Bazelisk, a tool that manages Bazel versions
and installs the correct version of Bazel.
[Follow the instructions for your platform on the Bazelisk GitHub page](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation).

## DP Building Block Libraries

This documentation is common to all DP building block libraries. Currently, the
libraries support the following algorithms:

Algorithm                        | C++       | Go        | Java
:------------------------------- | :-------: | :-------: | :-------:
Laplace mechanism                | Supported | Supported | Supported
Gaussian mechanism               | Supported | Supported | Supported
Count                            | Supported | Supported | Supported
Sum                              | Supported | Supported | Supported
Mean                             | Supported | Supported | Supported
Variance                         | Supported | Supported | Supported
Quantiles                        | Supported | Supported | Supported
Automatic bounds approximation   | Supported | Planned   | Supported
Truncated geometric thresholding | Supported | Supported | Supported
Laplace thresholding             | Supported | Supported | Supported
Gaussian thresholding            | Supported | Supported | Supported
Pre-thresholding                 | Supported | Supported | Supported

Implementations of the Laplace mechanism and the Gaussian mechanism use
[secure noise generation]. These mechanisms can be used to perform computations
that aren't covered by the algorithms implemented in our libraries.

[secure noise generation]: ./common_docs/Secure_Noise_Generation.pdf

### Caveats of the DP building block libraries

Differential privacy requires some bound on maximum number of contributions each
user can make to a single aggregation. The DP building block libraries don't
perform such bounding: their implementation assumes that each user contributes
only a fixed number of rows to each partition. That number can be configured by
the user. The library neither verifies nor enforces this limit; it is the
caller's responsibility to pre-process data to enforce this.

We chose not to implement this step at the DP building block level because it
requires some *global* operation over the data: group by user, and aggregate or
subsample the contributions of each user before passing them on to the DP
building block aggregators. Given scalability constraints, this pre-processing
must be done by a higher-level part of the infrastructure, typically a
distributed data processing framework: for example, Privacy on Beam relies on
Apache Beam for this operation and PipelineDP4j relies on Apache Beam or Apacahe
Spark. Therefore it is recommended to use the end-to-end tooling if possible:
Privacy on Beam for Go, PipelineDP4j for Kotlin/Scala/Java and PipelineDP for
Python.

For more detail about our approach to building scalable end-to-end differential
privacy frameworks, we recommend reading:

1.  [Differential privacy computations in data pipelines reference doc](https://github.com/google/differential-privacy/blob/main/common_docs/Differential_Privacy_Computations_In_Data_Pipelines.pdf),
    which describes how to build such a system using any data pipeline framework
    (e.g. Apache Beam or Apache Spark).
2.  Our
    [paper about differentially private SQL](https://arxiv.org/abs/1909.01917),
    which describes such a system. Even though the interface of Privacy on Beam
    and PipelineDP4j is different, it conceptually uses the same framework as
    the one described in this paper.

### Known issues

Our floating-point implementations are subject to the vulnerabilities described
in [Casacuberta et al. "Widespread Underestimation of Sensitivity in
Differentially Private Libraries and How to Fix
it"](https://arxiv.org/abs/2207.10635) (specifically the rounding, repeated
rounding, and re-ordering attacks). These vulnerabilities are particularly
concerning when an attacker can control some of the contents of a dataset and/or
its order. Our integer implementations are not subject to the vulnerabilities
described in the paper (though note that Java does not have an integer
implementation).

Please refer to our [attack model](common_docs/attack_model.md) to learn more
about how to use our libraries in a safe way.

## Reach out

### Join Our Slack Community

The best way to connect with the Google Differential Privacy team and other DP
enthusiasts is by joining our Slack community. It's the perfect place to ask
questions, get support, discuss new features, and stay up-to-date on all things
related to our open-source DP libraries.

Click
[here](https://join.slack.com/t/dp-open-source/shared_invite/zt-35hw483tz-nS5YOtGjxCHk3Ek7WiXvlg)
to join!

Once you're in, check out these key channels:

*   #introductions: Say hello and tell us a bit about your interest in
    differential privacy!
*   #support: Get assistance with our DP libraries, ask questions, and
    troubleshoot issues.
*   #development: For contributors and developers to discuss code, pull
    requests, and the project roadmap.
*   #general: Official announcements, project news, and broader community
    updates.

<br>
<div align="center">
<a href="https://join.slack.com/t/dp-open-source/shared_invite/zt-35hw483tz-nS5YOtGjxCHk3Ek7WiXvlg" style="display: inline-block; padding: 10px 20px; font-size: 16px; font-weight: bold; text-align: center; text-decoration: none; color: #fff; background-color: #4A154B; border-radius: 5px;">
Join Our Slack </a>
</div>
<br>

### Google Group

If you don't use Slack, you can join our
[public discussion group](https://groups.google.com/g/dp-open-source-users).

### Email

Send us an email at dp-open-source@google.com to discuss your specific use cases
privately and how we can better assist you.

Please avoid sharing any personally identifiable information. If you need to
delete a previous message, please contact us.

## Support

We are actively maintaining and improving the libraries. We welcome
contributions to this project. For pull requests, please review our
[contribution guidelines](CONTRIBUTING.md) and consider joining the #development
channel in
[our Slack workspace](https://join.slack.com/t/dp-open-source/shared_invite/zt-35hw483tz-nS5YOtGjxCHk3Ek7WiXvlg).
We will respond to issues filed in this project. If we plan to discontinue
active maintenance and issue responses, we will provide a notice here at least 3
months in advance.

## License

[Apache License 2.0](LICENSE)

## Support Disclaimer

This is not an officially supported Google product.

## Related projects

-   [OpenDP](https://opendp.org), a community effort around tools for
    statistical analysis of sensitive private data.
-   [JAX Privacy](https://github.com/google-deepmind/jax_privacy), a library to
    train machine learning models with differential privacy.
-   [TensorFlow Privacy](https://github.com/tensorflow/privacy), a TensorFlow
    library that preceded JAX Privacy.
