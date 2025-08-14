include(FetchContent)

# Abseil
FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG         20250512.1
)
FetchContent_MakeAvailable(abseil-cpp)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0  # April 30 2025
)

FetchContent_MakeAvailable(googletest)

find_package(OpenSSL REQUIRED)

# Include directories for the project
# CMAKE_CURRENT_BINARY_DIR/.. for including proto/
include_directories(${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/..")

# --- Core Library ---
add_library(dp_algorithms
    algorithms/rand.cc
    algorithms/distributions.cc
    algorithms/util.cc
    algorithms/numerical-mechanisms.cc
    algorithms/gaussian-dp-calculator.cc
    algorithms/internal/count-tree.cc
    algorithms/internal/gaussian-stddev-calculator.cc
    algorithms/internal/bounded-mean-ci.cc
    algorithms/internal/clamped-calculation-without-bounds.h
    ../third_party/cephes/inverse_gaussian_cdf.cc
    # header only files (like algoritm.h) must not be added
)

target_link_libraries(dp_algorithms
    PRIVATE
        absl::base
        absl::status
        absl::strings
        OpenSSL::SSL
        OpenSSL::Crypto
        protobuf::libprotobuf
        dp_protos
)
# Public include directory for the library
target_include_directories(dp_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})


# --- Tests ---
enable_testing()

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/*_test.cc")

set(SKIPPED_TESTS
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/distributions_benchmark_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/approx-bounds-provider_test.cc" # weird link error because of matchers
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/approx-bounds-as-bounds-provider_test.cc" # weird link error because of matchers
)


foreach(TEST_SOURCE ${TEST_SOURCES})
    if(${TEST_SOURCE} IN_LIST SKIPPED_TESTS)
        message(STATUS "Skipping test: ${TEST_SOURCE}")
        continue()
    endif()
    # Extract the test name (e.g., "my_feature_test" from "my_feature_test.cpp")
    # This creates a unique name for each test executable and CTest entry.
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE) # NAME_WE gets name without extension

    # Create an executable for each test source file
    # This will compile my_feature_test.cpp into my_feature_test executable
    add_executable(${TEST_NAME} ${TEST_SOURCE} base/testing/status_matchers.cc)

    # Link necessary libraries (e.g., testing frameworks like Google Test, your project's libraries)
    target_link_libraries(${TEST_NAME} PRIVATE
         dp_algorithms
            GTest::gtest
            GTest::gmock
            GTest::gtest_main
            absl::base
            absl::flags
            absl::log
            absl::log_internal_check_op
            absl::random_random
            absl::status
            absl::statusor
            absl::strings
            OpenSSL::SSL
            OpenSSL::Crypto
            protobuf::libprotobuf
            dp_protos
    )

    # Add the executable as a CTest test
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
