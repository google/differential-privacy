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
        "//base/testing:status_matchers",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_googletest//:gtest_main",
    ],
)
