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
the [codelab/](https://github.com/google/differential-privacy/tree/main/privacy-on-beam/codelab)
directory.

Full documentation of the API is available as [godoc](https://godoc.org/github.com/google/differential-privacy/privacy-on-beam/pbeam).

## Using with Bazel

In order to include Privacy on Beam in your Bazel project, you need to add the
following to your `WORKSPACE` file (use the latest commit id or the id of the
commit you want to depend on):

```
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "6f111c57fd50baf5b8ee9d63024874dd2a014b069426156c55adbf6d3d22cb7b",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.25.0/rules_go-v0.25.0.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.25.0/rules_go-v0.25.0.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.15.5")

http_archive(
    name = "bazel_gazelle",
    sha256 = "b85f48fa105c4403326e9525ad2b2cc437babaa6e15a3fc0b1dbab0ab064bc7c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies", "go_repository")

gazelle_dependencies()

git_repository(
    name = "com_github_google_differential_privacy",
    remote = "https://github.com/google/differential-privacy.git",
    commit = "de8460c9791de4c89a9dbb906b11a8f62e045f7b",
)

# Load dependencies for Google DP Library base workspace.
load("@com_github_google_differential_privacy//:differential_privacy_deps.bzl", "differential_privacy_deps")
differential_privacy_deps()

# Protobuf transitive dependencies.
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

git_repository(
    name = "com_google_go_differential_privacy",
    remote = "https://github.com/google/differential-privacy.git",
    # Workaround from https://github.com/bazelbuild/bazel/issues/10062#issuecomment-642144553
    patch_cmds = ["mv (broken link) ."],
    commit = "de8460c9791de4c89a9dbb906b11a8f62e045f7b",
)

load("@com_google_go_differential_privacy//:go_differential_privacy_deps.bzl", "go_differential_privacy_deps")
go_differential_privacy_deps()

git_repository(
    name = "com_google_privacy_on_beam",
    remote = "https://github.com/google/differential-privacy.git",
    strip_prefix = "privacy-on-beam/",
    commit = "de8460c9791de4c89a9dbb906b11a8f62e045f7b",
)

load("@com_google_privacy_on_beam//:privacy_on_beam_deps.bzl", "privacy_on_beam_deps")
privacy_on_beam_deps()
```

Then, you can depend on `@com_google_privacy_on_beam` in your BUILD files.
