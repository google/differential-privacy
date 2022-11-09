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

## How to Use

Our [codelab](https://codelabs.developers.google.com/codelabs/privacy-on-beam/)
about computing private statistics with Privacy on Beam
demonstrates how to use the library. Source code for the codelab is available in
the [codelab/](codelab)
directory.

Full documentation of the API is available as [godoc](https://godoc.org/github.com/google/differential-privacy/privacy-on-beam/v2/pbeam).

## Using with the "go" Command

For building Privacy on Beam with the ["go" command](https://golang.org/cmd/go/),
you can run the following:
```shell
go build -mod=mod ./...
```
This will build all the packages. `-mod=mod` is necessary for installing all the
dependencies automatically. Otherwise, you'll be asked to install each
dependency manually.

Similarly, you can run all the tests with:
```shell
go test -mod=mod ./...
```

If you wish to run the codelab, you can do so by:
```shell
cd codelab/main
go run -mod=mod . -example=count -input_file=day_data.csv -output_stats_file=stats.csv -output_chart_file=chart.png
```

Change `example` to run other examples. See the
[codelab documentation](https://codelabs.developers.google.com/codelabs/privacy-on-beam/)
for more information.

Both for `go run` and `go test`, if you already built the code with `go build`,
you can omit `-mod=mod`.

## Using with Bazel

In order to include Privacy on Beam in your Bazel project, you need to add the
following to your `WORKSPACE` file (change `dp_lib_version` to the version you
want to depend on, or alternatively you can depend on a specific commit; but
keep in mind that you have to update `dp_lib_tar_sha256` as well):

```
dp_lib_version = "1.1.2" # Change to the version you want to use.
dp_lib_tar_sha256 = "b7476712485053ed039b05bbe7481a43c8760457e6ebd7afb320d51fdf744fb5" # Change to the sha256 of the .tar.gz of the version you want to use.
dp_lib_url = "https://github.com/google/differential-privacy/archive/refs/tags/v" + dp_lib_version + ".tar.gz"

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_github_google_differential_privacy",
    sha256 = dp_lib_tar_sha256,
    urls = [
        dp_lib_url,
    ],
    strip_prefix = "differential-privacy-" + dp_lib_version,
)

http_archive(
    name = "com_google_go_differential_privacy",
    sha256 = dp_lib_tar_sha256,
    urls = [
        dp_lib_url,
    ],
    strip_prefix = "differential-privacy-" + dp_lib_version + "/go",
)

http_archive(
    name = "com_google_privacy_on_beam",
    sha256 = dp_lib_tar_sha256,
    urls = [
        dp_lib_url,
    ],
    strip_prefix = "differential-privacy-" + dp_lib_version + "/privacy-on-beam",
)

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "099a9fb96a376ccbbb7d291ed4ecbdfd42f6bc822ab77ae6f1b5cb9e914e94fa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "448e37e0dbf61d6fa8f00aaa12d191745e14f07c31cabfa731f0c8e8a4f41b97",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.28.0/bazel-gazelle-v0.28.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.28.0/bazel-gazelle-v0.28.0.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.18.3")

# Load dependencies for Privacy on Beam.
load("@com_google_privacy_on_beam//:privacy_on_beam_deps.bzl", "privacy_on_beam_deps")
privacy_on_beam_deps()

# Load dependencies for the Go DP Library.
load("@com_google_go_differential_privacy//:go_differential_privacy_deps.bzl", "go_differential_privacy_deps")
go_differential_privacy_deps()

# Load dependencies for Google DP Library base workspace.
load("@com_github_google_differential_privacy//:differential_privacy_deps.bzl", "differential_privacy_deps")
differential_privacy_deps()

# Protobuf transitive dependencies.
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Gazelle dependencies must be added last.
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()
```

Then, you can depend on `@com_google_privacy_on_beam` in your BUILD files.
