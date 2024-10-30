# Differential Privacy

> **Note**<br>
> If you are unfamiliar with differential privacy (DP), you might want to go through ["A friendly, non-technical introduction to differential privacy"](https://desfontain.es/privacy/friendly-intro-to-differential-privacy.html).

This repository contains libraries to generate ε- and (ε, δ)-differentially
private statistics over datasets. It contains the following tools.

* [Privacy on Beam](privacy-on-beam) is an end-to-end differential privacy
  framework built on top of [Apache Beam](https://beam.apache.org/documentation/).
  It is intended to be easy to use, even by non-experts.
* [PipelineDP4j](pipelinedp4j) is an end-to-end differential privacy framework
  for JVM languages (Java, Kotlin, Scala). It supports different data
  processing frameworks such as
  [Apache Beam](https://beam.apache.org/documentation/) and
  [Apache Spark](https://spark.apache.org/) (coming soon). It is intended to
  be easy to use, even by non-experts.
* Three "DP building block" libraries, in [C++](cc), [Go](go), and [Java](java).
  These libraries implement basic noise addition primitives and differentially
  private aggregations. Privacy on Beam is implemented using these libraries.
* A [stochastic tester](cc/testing), used to help catch regressions that could
  make the differential privacy property no longer hold.
* A [differential privacy accounting library](python/dp_accounting), used for
  tracking privacy budget.
* A [command line interface](examples/zetasql) for running differentially
  private SQL queries with [ZetaSQL](https://github.com/google/zetasql).
* [DP Auditorium](python/dp_auditorium) is a library for auditing differential
  privacy guarantees.

To get started on generating differentially private data, we recommend you follow
the [Privacy on Beam codelab](https://codelabs.developers.google.com/codelabs/privacy-on-beam/).

Currently, the DP building block libraries support the following algorithms:

| Algorithm                        | C++       | Go        | Java      |
| :------------------------------- | :-------: | :-------: | :-------: |
| Laplace mechanism                | Supported | Supported | Supported |
| Gaussian mechanism               | Supported | Supported | Supported |
| Count                            | Supported | Supported | Supported |
| Sum                              | Supported | Supported | Supported |
| Mean                             | Supported | Supported | Supported |
| Variance                         | Supported | Supported | Supported |
| Standard deviation               | Supported | Supported | Planned   |
| Quantiles                        | Supported | Supported | Supported |
| Automatic bounds approximation   | Supported | Planned   | Supported |
| Truncated geometric thresholding | Supported | Supported | Supported |
| Laplace thresholding             | Supported | Supported | Supported |
| Gaussian thresholding            | Planned   | Supported | Supported |
| Pre-thresholding                 | Supported | Supported | Supported |

Implementations of the Laplace mechanism and the Gaussian mechanism use [secure
noise generation]. These mechanisms can be used to perform computations that
aren't covered by the algorithms implemented in our libraries.

[secure noise generation]: ./common_docs/Secure_Noise_Generation.pdf

The DP building block libraries and Privacy on Beam are suitable for research,
experimental, or production use cases, while the other tools are currently
experimental and subject to change.

## How to Build

In order to run the differential privacy library, you need to install bazelisk,
if you don't have it already. Bazelisk manages [Bazel](https://bazel.build/)
versions and installs correct one.
[Follow the instructions for your platform on the bazelisk github page](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation)

You also need to install Git, if you don't have it already.
[Follow the instructions for your platform on the Git website.](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Once you've installed bazelisk and Git, open a Terminal and clone the
differential privacy directory into a local folder:

```shell
git clone https://github.com/google/differential-privacy.git
```

Navigate into the `differential-privacy` folder you just created,
and build the differential privacy library and dependencies using bazelisk
(note: *...* is a part of the command and not a placeholder):

To build the C++ library, run:
```shell
cd cc
bazelisk build ...
```
To build the Go library, run:
```shell
cd go
bazelisk build ...
```

To build the Java library, run:
```shell
cd java
bazelisk build ...
```

To build the PipelineDP4j library, run:
```shell
cd pipelinedp4j
bazelisk build ...
```

To build Privacy on Beam, run:
```shell
cd privacy-on-beam
bazelisk build ...
```

You may need to install additional dependencies when building the PostgreSQL
extension, for example on Ubuntu you will need these packages:

```shell
sudo apt-get install make libreadline-dev bison flex
```

## Caveats of the DP building block libraries

Differential privacy requires some bound on maximum number of contributions
each user can make to a single aggregation. The DP building block libraries
don't perform such bounding: their implementation assumes that each user
contributes only a fixed number of rows to each partition. That number
can be configured by the user. The library neither verifies nor
enforces this limit; it is the caller's responsibility to pre-process data to
enforce this.

We chose not to implement this step at the DP building block level because it
requires some *global* operation over the data: group by user, and aggregate or
subsample the contributions of each user before passing them on to the DP
building block aggregators. Given scalability constraints, this pre-processing
must be done by a higher-level part of the infrastructure, typically a
distributed processing framework: for example, Privacy on Beam relies on Apache
Beam for this operation.

For more detail about our approach to building scalable end-to-end differential
privacy frameworks, we recommend reading:

1. [Differential privacy computations in data pipelines reference doc](https://github.com/google/differential-privacy/blob/main/common_docs/Differential_Privacy_Computations_In_Data_Pipelines.pdf),
which describes how to build such a system using any data pipeline framework
(e.g. Apache Beam).
2. Our
[paper about differentially private SQL](https://arxiv.org/abs/1909.01917),
which describes such a system. Even though the interface of Privacy on Beam is
different, it conceptually uses the same framework as the one described in this
paper.

## Known issues

Our floating-point implementations are subject to the vulnerabilities described
in [Casacuberta et al. "Widespread Underestimation of
Sensitivity in Differentially Private Libraries and How to Fix it"](https://arxiv.org/abs/2207.10635)
(specifically the rounding, repeated rounding, and re-ordering attacks). These
vulnerabilities are particularly concerning when an attacker can control some of
the contents of a dataset and/or its order. Our integer implementations are not
subject to the vulnerabilities described in the paper (though note that Java
does not have an integer implementation).

Please refer to our [attack model](common_docs/attack_model.md) to learn more
about how to use our libraries in a safe way.

## Support

We will continue to publish updates and improvements to the library. We are
happy to accept contributions to this project. Please follow
[our guidelines](CONTRIBUTING.md) when sending pull requests. We will respond to
issues filed in this project. If we intend to stop publishing improvements and
responding to issues we will publish notice here at least 3 months in advance.

## License

[Apache License 2.0](LICENSE)

## Support Disclaimer

This is not an officially supported Google product.

## Reach out

We are always keen on learning about how you use this library and what use cases
it helps you to solve. We have two communication channels:

  * A [public discussion
    group](https://groups.google.com/g/dp-open-source-users) where we will also
    share our preliminary roadmap, updates, events, etc.

  * A private email alias at dp-open-source@google.com where you can reach us
    directly about your use cases and what more we can do to help.

Please refrain from sending any personal identifiable information. If you wish
to delete a message you've previously sent, please contact us.

## Related projects

- [PyDP](https://github.com/OpenMined/PyDP), a Python wrapper of our C++ DP
  building block library, driven by the [OpenMined](https://www.openmined.org/)
  open-source community.
- [PipelineDP](https://github.com/OpenMined/PipelineDP), an end-to-end
  differential privacy framework (similar to [Privacy on Beam](privacy-on-beam))
  that works with Apache Beam & Apache Spark in Python, co-developed by Google
  and [OpenMined](https://www.openmined.org/).
- [OpenDP](https://opendp.org), a community effort around tools for statistical
  analysis of sensitive private data.
- [TensorFlow Privacy](https://github.com/tensorflow/privacy), a library to
  train machine learning models with differential privacy.
