"""Setup code to allow loading dependencies from maven."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dp_java_deps_prework():
    """ Does the pre-work necessary to use the maven_install rule.

        This must be called before the rest of the dependencies are loaded.
    """
    http_archive(
        name = "rules_jvm_external",
        sha256 = "995ea6b5f41e14e1a17088b727dcff342b2c6534104e73d6f06f1ae0422c2308",
        url = "https://github.com/bazelbuild/rules_jvm_external/archive/4.1.tar.gz",
        strip_prefix = "rules_jvm_external-4.1",
    )
    http_archive(
        name = "bazel_common",
        url = "https://github.com/google/bazel-common/archive/3d0e5005cfcbee836e31695d4ab91b5328ccc506.tar.gz",
        sha256 = "8dd4dd688b42148f2a87652901a4eb2c85c64834be7a6890ebfc8ef1f67eeeaa",
        strip_prefix = "bazel-common-3d0e5005cfcbee836e31695d4ab91b5328ccc506",
    )
