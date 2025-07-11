load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

cc_library(
    name = "inverse_gaussian_cdf",
    srcs = ["inverse_gaussian_cdf.cc"],
    hdrs = ["inverse_gaussian_cdf.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "inverse_gaussian_cdf_test",
    srcs = ["inverse_gaussian_cdf_test.cc"],
    deps = [
        ":inverse_gaussian_cdf",
        "@com_google_googletest//:gtest_main",
    ],
)
