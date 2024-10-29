<!-- TODO: revise all pipelinedp4j related readmes (this one,
google internal and root differential privacy package readme). -->

# PipelineDP4j

PipelineDP4j is an end-to-end differential privacy solution for JVM that supports various frameworks for distributed data processing such as [Apache Spark](https://spark.apache.org/) and
[Apache Beam](https://beam.apache.org/documentation/).
It is intended to be usable by all developers, regardless of their differential
privacy expertise.

Internally, PipelineDP4j relies on the lower-level building blocks from the
differential privacy library and combines them into an "out-of-the-box" solution
that takes care of all the steps that are essential to differential privacy,
including noise addition, [partition selection](https://arxiv.org/abs/2006.03684),
and contribution bounding. Thus, rather than using the lower-level differential
privacy library, it is recommended to use PipelineDP4j, as it can reduce
implementation mistakes.

PipelineDP4j can be used on any JVM using any JVM compatible language like Kotlin, Scala or Java.

## How to Use

<!-- TODO: create codelab and check links. -->
Our [codelab](https://codelabs.developers.google.com/codelabs/pipelinedp4j/)
about computing private statistics with PipelineDP4j
demonstrates how to use the library. Source code for the codelab is available in
the [codelab/](codelab)
directory.

<!-- TODO: insert link. -->
Full documentation of the API is available as [kdoc]().

## Using with Bazel
<!-- TODO: describe how to build. -->