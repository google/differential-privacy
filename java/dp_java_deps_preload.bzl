"""Setup code to allow loading dependencies from maven."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dp_java_deps_prework():
    """ Does the pre-work necessary to use the maven_install rule.

        This must be called before the rest of the dependencies are loaded.
    """
    RULES_JVM_EXTERNAL_TAG = "4.4.2"
    RULES_JVM_EXTERNAL_SHA = "735602f50813eb2ea93ca3f5e43b1959bd80b213b836a07a62a29d757670b77b"
    BAZEL_COMMON_TAG = "3d0e5005cfcbee836e31695d4ab91b5328ccc506"
    BAZEL_COMMON_SHA = "8dd4dd688b42148f2a87652901a4eb2c85c64834be7a6890ebfc8ef1f67eeeaa"
    http_archive(
        name = "rules_jvm_external",
        strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
        sha256 = RULES_JVM_EXTERNAL_SHA,
        url = "https://github.com/bazelbuild/rules_jvm_external/archive/refs/tags/%s.zip" % RULES_JVM_EXTERNAL_TAG,
    )
    http_archive(
        name = "bazel_common",
        url = "https://github.com/google/bazel-common/archive/%s.tar.gz" % BAZEL_COMMON_TAG,
        sha256 = BAZEL_COMMON_SHA,
        strip_prefix = "bazel-common-%s" % BAZEL_COMMON_TAG,
    )
