"""Setup code to allow loading dependencies from maven."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dp_java_deps_prework():
    """ Does the pre-work necessary to use the maven_install rule.

        This must be called before the rest of the dependencies are loaded.
    """
    RULES_JVM_EXTERNAL_TAG = "6.0"
    RULES_JVM_EXTERNAL_SHA = "85fd6bad58ac76cc3a27c8e051e4255ff9ccd8c92ba879670d195622e7c0a9b7"
    BAZEL_COMMON_TAG = "a9e1d8efd54cbf27249695b23775b75ca65bb59d"
    BAZEL_COMMON_SHA = "17ea98149586dff60aa741c67fbd9a010fbb1507df90e741c50403bf5228bea3"
    http_archive(
        name = "rules_jvm_external",
        strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
        sha256 = RULES_JVM_EXTERNAL_SHA,
        url = "https://github.com/bazelbuild/rules_jvm_external/releases/download/%s/rules_jvm_external-%s.tar.gz" % (RULES_JVM_EXTERNAL_TAG, RULES_JVM_EXTERNAL_TAG),
    )
    http_archive(
        name = "bazel_common",
        url = "https://github.com/google/bazel-common/archive/%s.tar.gz" % BAZEL_COMMON_TAG,
        sha256 = BAZEL_COMMON_SHA,
        strip_prefix = "bazel-common-%s" % BAZEL_COMMON_TAG,
    )
