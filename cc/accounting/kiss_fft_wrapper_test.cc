// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "accounting/kiss_fft_wrapper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"

namespace differential_privacy {
namespace accounting {
namespace {
using ::testing::ElementsAre;
using ::testing::Pointwise;

constexpr double kMaxError = 1e-5;

MATCHER_P(IsNear, tolerance, "is near") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) <= tolerance;
}

TEST(KissFftWrapper, FftSize) {
  auto wrapper = KissFftWrapper(5);
  EXPECT_EQ(wrapper.EfficientRealSize(), 6);
}

TEST(KissFftWrapper, ForwardTransform) {
  std::vector<double> x = {2, 0, 4, 0};
  auto wrapper = KissFftWrapper(x.size());
  std::vector<double> input(wrapper.EfficientRealSize(), 0.0);
  absl::c_copy(x, input.begin());
  std::vector<std::complex<double>> output(wrapper.ComplexSize(), 0.0);

  wrapper.ForwardTransform(input.data(), output.data());

  std::vector<std::complex<double>> expected = {{6, 0}, {-2, 0}, {6, 0}};
  ASSERT_THAT(output, Pointwise(IsNear(kMaxError), expected));
}

TEST(KissFftWrapper, InverseTransform) {
  std::vector<std::complex<double>> x = {{6, 0}, {-2, 0}, {6, 0}};
  auto wrapper = KissFftWrapper(x.size());
  std::vector<double> output(wrapper.EfficientRealSize(), 0.0);

  wrapper.InverseTransform(x.data(), output.data());

  ASSERT_THAT(output, ElementsAre(8, 0, 16, 0));
}
}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
