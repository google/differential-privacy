// Unit tests for the patched CalculateDeltaForGaussianStddev.
// Add to: cc/algorithms/internal/gaussian-stddev-calculator_test.cc

#include <cmath>
#include <limits>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "algorithms/internal/gaussian-stddev-calculator.h"

namespace differential_privacy {
namespace internal {
namespace {

// ── Finding 1: NaN elimination ───────────────────────────────────────────────

TEST(GaussianStddevCalculatorTest, NoNanAtLargeEpsilon) {
  for (double epsilon : {709.78, 710.0, 800.0, 1000.0}) {
    const double delta = CalculateDeltaForGaussianStddev(
        epsilon, /*l2_sensitivity=*/1.0, /*stddev=*/1.0);
    EXPECT_FALSE(std::isnan(delta)) << "NaN at epsilon=" << epsilon;
    EXPECT_GE(delta, 0.0);
    EXPECT_LE(delta, 1.0);
  }
}

TEST(GaussianStddevCalculatorTest, LargeEpsilonReturnsNearZero) {
  for (double epsilon : {500.0, 600.0, 709.0, 1000.0}) {
    const double delta = CalculateDeltaForGaussianStddev(
        epsilon, /*l2_sensitivity=*/1.0, /*stddev=*/1.0);
    EXPECT_NEAR(delta, 0.0, 1e-10) << "eps=" << epsilon;
  }
}

// ── Invalid inputs: must return 1.0 (conservative maximum delta) ─────────────

TEST(GaussianStddevCalculatorTest, NegativeStddevReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(1.0, 1.0, -1.0), 1.0);
}

TEST(GaussianStddevCalculatorTest, ZeroStddevReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(1.0, 1.0, 0.0), 1.0);
}

TEST(GaussianStddevCalculatorTest, NanStddevReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(
          1.0, 1.0, std::numeric_limits<double>::quiet_NaN()),
      1.0);
}

TEST(GaussianStddevCalculatorTest, InfStddevReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(
          1.0, 1.0, std::numeric_limits<double>::infinity()),
      1.0);
}

TEST(GaussianStddevCalculatorTest, InfEpsilonReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(
          std::numeric_limits<double>::infinity(), 1.0, 1.0),
      1.0);
}

TEST(GaussianStddevCalculatorTest, NegativeEpsilonReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(-1.0, 1.0, 1.0), 1.0);
}

TEST(GaussianStddevCalculatorTest, NanL2SensitivityReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(
          1.0, std::numeric_limits<double>::quiet_NaN(), 1.0),
      1.0);
}

TEST(GaussianStddevCalculatorTest, InfL2SensitivityReturnsOne) {
  EXPECT_DOUBLE_EQ(
      CalculateDeltaForGaussianStddev(
          1.0, std::numeric_limits<double>::infinity(), 1.0),
      1.0);
}

// ── Correctness: matches Theorem 8 reference values ─────────────────────────

// a = 1/(2*1) = 0.5, b = 1*1/1 = 1.0
// delta = Phi(-0.5) - e^1 * Phi(-1.5) ≈ 0.12693
TEST(GaussianStddevCalculatorTest, CorrectnessStandardParams) {
  const double delta = CalculateDeltaForGaussianStddev(
      /*epsilon=*/1.0, /*l2_sensitivity=*/1.0, /*stddev=*/1.0);
  EXPECT_NEAR(delta, 0.1269367, 1e-6);
}

TEST(GaussianStddevCalculatorTest, CorrectnessLargeSigma) {
  const double delta = CalculateDeltaForGaussianStddev(
      /*epsilon=*/1.0, /*l2_sensitivity=*/1.0, /*stddev=*/10.0);
  EXPECT_GT(delta, 0.0);
  EXPECT_LT(delta, 1e-3);
  EXPECT_FALSE(std::isnan(delta));
}

// ── Integration: binary search still finds correct sigma ─────────────────────

TEST(GaussianStddevCalculatorTest, BinarySearchStandardTarget) {
  const double sigma = CalculateGaussianStddev(
      /*epsilon=*/1.0, /*delta=*/1e-6, /*l2_sensitivity=*/1.0);
  EXPECT_GT(sigma, 0.0);
  EXPECT_FALSE(std::isnan(sigma));
  const double delta_actual = CalculateDeltaForGaussianStddev(1.0, 1.0, sigma);
  EXPECT_LE(delta_actual, 1e-6);  // Must satisfy the target.
}

TEST(GaussianStddevCalculatorTest, BinarySearchTightTarget) {
  const double sigma = CalculateGaussianStddev(
      /*epsilon=*/0.5, /*delta=*/1e-10, /*l2_sensitivity=*/1.0);
  EXPECT_GT(sigma, 0.0);
  EXPECT_FALSE(std::isnan(sigma));
  const double delta_actual = CalculateDeltaForGaussianStddev(0.5, 1.0, sigma);
  EXPECT_LE(delta_actual, 1e-10);
}

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
