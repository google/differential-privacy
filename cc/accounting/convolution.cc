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

#include "accounting/convolution.h"

#include <cmath>
#include <complex>
#include <optional>
#include <ostream>
#include <vector>

#include "accounting/common/common.h"
#include "accounting/kiss_fft_wrapper.h"

namespace differential_privacy {
namespace accounting {

using ::kiss_fft_cpx;
using ::kiss_fftr_cfg;
using ::std::complex;

UnpackedProbabilityMassFunction UnpackProbabilityMassFunction(
    const ProbabilityMassFunction& input) {
  if (input.empty()) {
    return UnpackedProbabilityMassFunction();
  }
  auto minmax = absl::c_minmax_element(
      input,
      [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
  int min_key = minmax.first->first;
  int max_key = minmax.second->first;

  UnpackedProbabilityMassFunction unpacked_map;
  unpacked_map.min_key = min_key;
  for (int key = min_key; key <= max_key; key++) {
    auto it = input.find(key);
    unpacked_map.items.push_back(it == input.end() ? 0 : it->second);
  }
  return unpacked_map;
}

ProbabilityMassFunction CreateProbabilityMassFunction(
    const UnpackedProbabilityMassFunction& input, double tail_mass_truncation) {
  int lower_truncation_index = -1;
  double lower_truncation_mass = 0;
  do {
    lower_truncation_mass += input.items.at(++lower_truncation_index);
  } while (lower_truncation_mass <= tail_mass_truncation / 2 &&
           lower_truncation_index + 1 < input.items.size());

  int upper_truncation_index = input.items.size();
  double upper_truncation_mass = 0;
  do {
    upper_truncation_mass += input.items.at(--upper_truncation_index);
  } while (upper_truncation_mass <= tail_mass_truncation / 2 &&
           upper_truncation_index > 0);

  ProbabilityMassFunction output;
  for (int index = lower_truncation_index; index <= upper_truncation_index;
       index++) {
    auto value = input.items.at(index);
    if (value > 0) {
      output[index + input.min_key] = value;
    }
  }
  return output;
}

ProbabilityMassFunction Convolve(const ProbabilityMassFunction& x,
                                 const ProbabilityMassFunction& y,
                                 double tail_mass_truncation) {
  UnpackedProbabilityMassFunction x_map = UnpackProbabilityMassFunction(x);
  UnpackedProbabilityMassFunction y_map = UnpackProbabilityMassFunction(y);

  const int size_x = x_map.items.size();
  const int size_y = y_map.items.size();
  const int output_size = size_x + size_y - 1;
  KissFftWrapper wrapper(output_size);
  const int real_size = wrapper.EfficientRealSize();
  const int complex_size = wrapper.ComplexSize();
  std::vector<double> x_input(real_size, 0.0);
  absl::c_copy(x_map.items, x_input.begin());
  std::vector<double> y_input(real_size, 0.0);
  absl::c_copy(y_map.items, y_input.begin());

  std::vector<complex<double>> x_transformed(complex_size, 0.0);
  std::vector<complex<double>> y_transformed(complex_size, 0.0);

  wrapper.ForwardTransform(x_input.data(), x_transformed.data());
  wrapper.ForwardTransform(y_input.data(), y_transformed.data());

  std::vector<complex<double>> convolution_transformed;
  convolution_transformed.reserve(x_transformed.size());
  for (int i = 0; i < x_transformed.size(); ++i) {
    convolution_transformed.push_back(
        std::complex<double>(x_transformed[i] * y_transformed[i]));
  }

  std::vector<double> result_vector(real_size, 0.0);
  wrapper.InverseTransform(convolution_transformed.data(),
                           result_vector.data());
  for (int i = 0; i < result_vector.size(); ++i) {
    result_vector[i] /= real_size;
  }
  UnpackedProbabilityMassFunction result_map;
  result_map.min_key = x_map.min_key + y_map.min_key;
  result_map.items = std::vector<double>(result_vector.begin(),
                                         result_vector.begin() + output_size);
  return CreateProbabilityMassFunction(result_map, tail_mass_truncation);
}

ConvolutionTruncationBounds ComputeConvolutionTruncationBounds(
    const UnpackedProbabilityMassFunction& x, int num_times,
    double tail_mass_truncation, std::optional<std::vector<double>> orders) {
  const int size = x.items.size();
  int64_t upper_bound = static_cast<int64_t>(size - 1) * num_times;
  int64_t lower_bound = 0;

  if (tail_mass_truncation == 0) {
    return ConvolutionTruncationBounds{lower_bound, upper_bound};
  }

  std::vector<double> mgf_orders;
  if (orders.has_value()) {
    mgf_orders = *std::move(orders);
  } else {
    // Set default orders so whose absolute values are not too large; otherwise,
    // we may run into numerical issues.
    for (int i = -20; i <= 20; ++i) {
      if (i != 0) {
        mgf_orders.push_back(static_cast<double>(i) / size);
      }
    }
  }

  for (const double order : mgf_orders) {
    // Compute the value of the moment generating function for the given order.
    double mgf_value = 0;
    for (int64_t i = 0; i < size; ++i) {
      mgf_value += std::exp(order * i) * x.items[i];
    }
    // Use Chernoff bound to update the upper/lower bound. See equation (5) in
    // the supplementary material.
    double bound =
        (num_times * std::log(mgf_value) + std::log(2 / tail_mass_truncation)) /
        order;
    if (order > 0) {
      upper_bound =
          std::min(upper_bound, static_cast<int64_t>(std::ceil(bound)));
    } else {
      lower_bound =
          std::max(lower_bound, static_cast<int64_t>(std::floor(bound)));
    }
  }

  return ConvolutionTruncationBounds{lower_bound, upper_bound};
}

ProbabilityMassFunction Convolve(const ProbabilityMassFunction& x,
                                 int num_times, double tail_mass_truncation) {
  UnpackedProbabilityMassFunction x_map = UnpackProbabilityMassFunction(x);

  const ConvolutionTruncationBounds truncation_bounds =
      ComputeConvolutionTruncationBounds(x_map, num_times,
                                         tail_mass_truncation);
  const int output_size =
      truncation_bounds.upper_bound - truncation_bounds.lower_bound + 1;
  KissFftWrapper wrapper(output_size);
  const int real_size = wrapper.EfficientRealSize();
  const int complex_size = wrapper.ComplexSize();

  std::vector<double> x_input(real_size, 0.0);
  absl::c_copy(x_map.items, x_input.begin());

  std::vector<complex<double>> x_transformed(complex_size, 0.0);
  wrapper.ForwardTransform(x_input.data(), x_transformed.data());

  std::vector<complex<double>> convolution_transformed;
  convolution_transformed.reserve(x_transformed.size());
  for (int i = 0; i < x_transformed.size(); ++i) {
    convolution_transformed.push_back(std::pow(x_transformed[i], num_times));
  }

  std::vector<double> result_vector(real_size, 0.0);
  wrapper.InverseTransform(convolution_transformed.data(),
                           result_vector.data());

  UnpackedProbabilityMassFunction result_map;
  result_map.min_key =
      x_map.min_key * num_times + truncation_bounds.lower_bound;

  // Discrete Fourier Transform wraps around module complex_size. Extract the
  // output values in the range of interest.
  for (int i = 0; i < output_size; ++i) {
    result_map.items.push_back(
        result_vector[(i + truncation_bounds.lower_bound) % real_size] /
        real_size);
  }

  return CreateProbabilityMassFunction(result_map);
}
}  // namespace accounting
}  // namespace differential_privacy
