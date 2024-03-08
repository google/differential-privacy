# Differential Privacy in Go

This is the Go implementation of the differential privacy library. For general
details and key definitions, see the top-level documentation.
This document describes Go-specific aspects.

## How to Use

Usage of the Go Differential Privacy library is demonstrated in the
[codelab](../examples/go/).

Full documentation of the API is available as [godoc](https://godoc.org/github.com/google/differential-privacy/go/v3/dpagg).

## Using with the "go" Command

For building the differential privacy library with the ["go" command](https://golang.org/cmd/go/),
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

If you already built the code with `go build`, you can omit `-mod=mod`.

If you wish to run the
[codelab](../examples/go/),
you can do so by (from the root of the repository):

```shell
cd examples/go/main
go run -mod=mod . -scenario=CountVisitsPerHour -input_file=../data/day_data.csv -non_private_output_file=out1.csv -private_output_file=out2.csv
```

Change `scenario` and `input_file` to run other scenarios. Check out the
[README](../examples/go/README.md)
for more information.

## Using with Bazel

In order to include the Go DP Library in your Bazel project, we recommend you
use [Gazelle](https://github.com/bazelbuild/bazel-gazelle):

1. Add the following dependencies to your `WORKSPACE` file (feel free to use the
   latest version of the dependencies by updating the version numbers in the
   URLs & sha256 checksums):
   ```
   load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

   http_archive(
      name = "io_bazel_rules_go",
      sha256 = "80a98277ad1311dacd837f9b16db62887702e9f1d1c4c9f796d0121a46c8e184",
      urls = [
         "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.46.0/rules_go-v0.46.0.zip",
         "https://github.com/bazelbuild/rules_go/releases/download/v0.46.0/rules_go-v0.46.0.zip",
      ],
   )

   http_archive(
      name = "bazel_gazelle",
      sha256 = "32938bda16e6700063035479063d9d24c60eda8d79fd4739563f50d331cb3209",
      urls = [
         "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.35.0/bazel-gazelle-v0.35.0.tar.gz",
         "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.35.0/bazel-gazelle-v0.35.0.tar.gz",
      ],
   )

   http_archive(
       name = "com_google_protobuf",
       sha256 = "4a7e87e4166c358c63342dddcde6312faee06ea9d5bb4e2fa87d3478076f6639",
       url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.5.tar.gz",
       strip_prefix = "protobuf-21.5",
   )

   load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

   go_rules_dependencies()

   go_register_toolchains(version = "1.22.0")

   # Protobuf transitive dependencies.
   load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
   protobuf_deps()

   # Gazelle dependencies must be added last.
   load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

   gazelle_dependencies()
   ```

1. Add the following code to your root `BUILD` or `BUILD.bazel` file:
   ```
   load("@bazel_gazelle//:def.bzl", "gazelle")
   # gazelle:prefix github.com/example/project
   gazelle(name = "gazelle")
   ```

1. Run `bazel run //:gazelle -- update-repos -from_file=go.mod`, which will add
   dependencies to your `WORKSPACE` file (you need a valid go.mod file with your
   dependencies, e.g. `github.com/google/differential-privacy/go/v3`).

1. Run `bazel run //:gazelle -- -go_naming_convention_external=go_default_library`
   to automatically generate or update your `BUILD` files and build targets.
   Alternatively, you can manually add
   `@com_github_google_differential_privacy_go_v3` as a dependency
   to targets in your `BUILD` files.
