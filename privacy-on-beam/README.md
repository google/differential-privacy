# Privacy on Beam

Privacy on Beam is an end-to-end differential privacy solution built on
[Apache Beam](https://beam.apache.org/documentation/).
It is intended to be usable by all developers, regardless of their differential
privacy expertise.

Internally, Privacy on Beam relies on the lower-level building blocks from the
differential privacy library and combines them into an "out-of-the-box" solution
that takes care of all the steps that are essential to differential privacy,
including noise addition, [partition selection](https://arxiv.org/abs/2006.03684),
and contribution bounding. Thus, rather than using the lower-level differential
privacy library, it is recommended to use Privacy on Beam, as it can reduce
implementation mistakes.

Privacy on Beam is only available in Go at the moment.

Note that this work is still experimental, as well as the
[Go SDK for Beam](https://beam.apache.org/documentation/sdks/go/), and is
subject to change.

## How to Use

Our [codelab](https://codelabs.developers.google.com/codelabs/privacy-on-beam/)
about computing private statistics with Privacy on Beam
demonstrates how to use the library. Source code for the codelab is available in
the [codelab/](https://github.com/google/differential-privacy/tree/master/privacy-on-beam/codelab)
directory.

Full documentation of the API is available as [godoc](https://godoc.org/github.com/google/differential-privacy/privacy-on-beam/pbeam).
