# PipelineDP4j

PipelineDP4j is an end-to-end differential privacy solution for JVM that
supports various frameworks for distributed data processing such as
[Apache Beam](https://beam.apache.org/documentation/) and
[Apache Spark](https://spark.apache.org/) (coming soon). It is intended to be
usable by all developers, regardless of their differential privacy expertise.

Internally, PipelineDP4j relies on the lower-level building blocks from the
differential privacy library and combines them into an "out-of-the-box" solution
that takes care of all the steps that are essential to differential privacy,
including noise addition,
[partition selection](https://arxiv.org/abs/2006.03684), and contribution
bounding. Thus, rather than using the lower-level differential privacy library,
it is recommended to use PipelineDP4j, as it can reduce implementation mistakes.

You can use PipelineDP4j in Java, Kotlin or Scala.

## How to Use

WARNING: Current API version (0.0.1) is experimental and will be changed in 2024
without backward-compatibility. The experimental API won't be supported and
maintained after that.

### Example

<!-- TODO: create codelab and rewrite this section. -->
<!-- TODO: generate kDoc of API using Dokka and GitHub pages. -->

Familiarize yourself with an
[example](https://github.com/google/differential-privacy/tree/main/examples/pipelinedp4j).
It shows how to compute differentially private statistics on a real dataset
using the library. Also, the documentation explains how to
[run the library on Google Cloud](https://github.com/google/differential-privacy/tree/main/examples/pipelinedp4j#running-on-google-cloud-platform).

The public API of the library is located in the
[API package](https://github.com/google/differential-privacy/tree/main/pipelinedp4j/main/com/google/privacy/differentialprivacy/pipelinedp4j/api).
You can look at it if you need something beyond the example.

### Use the library from Maven repository

The easiest way to start using the library in your project is to use the
dependency from Maven repository. You can find it
[here](https://mvnrepository.com/artifact/com.google.privacy.differentialprivacy.pipelinedp4j/pipelinedp4j).
After adding this dependency into your project you can write the same code as in
the example above and it will compile.

Please, don't use `0.0.1` version in production code as it is experimental and
its maintenance will be stopped in 2024 with release of the new version.
