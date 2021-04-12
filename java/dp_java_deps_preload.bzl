"""Setup code to allow loading dependencies from maven."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dp_java_deps_prework():
    """ Does the pre-work necessary to use the maven_install rule.

        This must be called before the rest of the dependencies are loaded.
    """
    http_archive(
        name = "rules_jvm_external",
        sha256 = "31d226a6b3f5362b59d261abf9601116094ea4ae2aa9f28789b6c105e4cada68",
        url = "https://github.com/bazelbuild/rules_jvm_external/archive/4.0.tar.gz",
        strip_prefix = "rules_jvm_external-4.0",
    )
