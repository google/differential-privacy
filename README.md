# Differential Privacy

This repository contains libraries to generate ε- and (ε, δ)-differentially
private statistics over datasets. It contains the following tools.

* [Privacy on Beam](privacy-on-beam) is an end-to-end differential privacy
  framework built on top of [Apache Beam](https://beam.apache.org/documentation/).
  It is intended to be easy to use, even by non-experts.
* Three "DP building block" libraries, in [C++](cc), [Go](go), and [Java](java).
  These libraries implement basic noise addition primitives and differentially
  private aggregations. Privacy on Beam is implemented using these libraries.
* A [stochastic tester](cc/testing), used to help catch regressions that could
  make the differential privacy property no longer hold.
* A [differential privacy accounting library](python/dp_accounting), used for
  tracking privacy budget.

To get started on generating differentially private data, we recomend you follow
the [Privacy on Beam codelab](https://codelabs.developers.google.com/codelabs/privacy-on-beam/).

Currently, the DP building block libraries support the following algorithms:

| Algorithm          | C++           | Go        |Java      |
| -------------      |:-------------:|:---------:|:--------:|
| Count              | Supported     | Supported |Supported |
| Sum                | Supported     | Supported |Supported |
| Mean               | Supported     | Supported |Supported |
| Variance           | Supported     | Planned   |Planned   |
| Standard deviation | Supported     | Planned   |Planned   |
| Order statistics (incl. min, max, and median) | Supported   | Planned | Planned |
| Automatic bounds approximation | Supported   | Planned | Planned |

They also contain [safe implementations](common_docs/Secure_Noise_Generation.pdf)
of Laplace and Gaussian mechanisms, which can be used to perform computations
that aren't covered by the algorithms implemented in our libraries.

The DP building block libraries are suitable for research, experimental or
production use cases, while the other tools are currently experimental, and
subject to change.

## How to Build

In order to run the differential private library, you need to install Bazel in
version 3.7.2, if you don't have it already.
[Follow the instructions for your platform on the Bazel website](https://docs.bazel.build/versions/master/install.html)

You also need to install Git, if you don't have it already.
[Follow the instructions for your platform on the Git website.](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Once you've installed Bazel and Git, open a Terminal and clone the
differential privacy directory into a local folder:

```git clone https://github.com/google/differential-privacy.git```

Navigate into the ```differential-privacy``` folder you just created,
and build the differential privacy library and dependencies using Bazel
(note: *...* is a part of the command and not a placeholder):

To build the C++ library, run:
```
cd cc
bazel build ...
```
To build the Go library, run:
```
cd go
bazel build ...
```

To build the Java library, run:
```
cd java
bazel build ...
```

You may need to install additional dependencies when building the PostgreSQL
extension, for example on Ubuntu you will need these packages:

```sudo apt-get install make libreadline-dev bison flex```

## Caveats of the DP building block libraries

Differential privacy requires some bound on maximum number of contributions
each user can make to a single aggregation. The DP building block libraries
don't perform such bounding: their implementation assumes that each user
contributes only a single row to each partition. It neither verifies nor
enforces this; it is the caller's responsibility to pre-process data to enforce
this bound.

We chose not to implement this step at the DP building block level because it
requires some *global* operation over the data: group by user, and aggregate or
subsample the contributions of each user before passing them on to the DP
building block aggregators. Given scalability constraints, this pre-processing
must be done by a higher-level part of the infrastructure, typically a
distributed processing framework: for example, Privacy on Beam relies on Apache
Beam for this operation.

For more detail about our approach to building scalable end-to-end differential
privacy frameworks, we recommend reading our
[paper about differentially private SQL](https://arxiv.org/abs/1909.01917),
which describes such a system. Even though the interface of Privacy on Beam is
different, it conceptually uses the same framework as the one described in this
paper.

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
it helps you to solve.  We have two communication channels:

  * A [public discussion
    group](https://groups.google.com/g/dp-open-source-users) where we will also
    share our preliminary roadmap, updates, events, etc.

  * A private email alias at dp-open-source@google.com where you can reach us
    directly about your use cases and what more we can do to help.

Please refrain from sending any personal identifiable information. If you wish
to delete a message you've previously sent, please contact us.

