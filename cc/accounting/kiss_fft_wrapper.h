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
#ifndef DIFFERENTIAL_PRIVACY_ACCOUNTING_CPP_KISS_FFT_WRAPPER_H_
#define DIFFERENTIAL_PRIVACY_ACCOUNTING_CPP_KISS_FFT_WRAPPER_H_

#include <cmath>
#include <complex>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "kissfft/kiss_fft.h"
#include "kissfft/kiss_fftr.h"

namespace differential_privacy {
namespace accounting {

// Simple wrapper around KISS FFT library for real(double)->complex
// transformations. Efficient FFT size is determined from the size of the input
// and is available as EfficientRealSize member. The caller should pad input up
// to that size.
class KissFftWrapper {
 public:
  // Instantiates wrapper with the size of double vector to be transformed.
  explicit KissFftWrapper(int real_size)
      : real_size_(GetEfficientSize(real_size)),
        complex_size_(real_size_ / 2 + 1),
        forward_config_(ABSL_DIE_IF_NULL(::kiss_fftr_alloc(
            real_size_, 0, nullptr, nullptr))),
        inverse_config_(ABSL_DIE_IF_NULL(::kiss_fftr_alloc(
            real_size_, 1, nullptr, nullptr))) {}
  virtual ~KissFftWrapper() {
    ::kiss_fftr_free(forward_config_);
    ::kiss_fftr_free(inverse_config_);
  }

  // Returns efficient FFT size.
  int EfficientRealSize() const { return real_size_; }

  // Returns complex size.
  int ComplexSize() const { return complex_size_; }

  // Forward real-to-complex FFT.
  // Caller should allocate buffer the size of EfficientRealSize().
  void ForwardTransform(const double* input,
                        std::complex<double>* output) const {
    CHECK(input);
    CHECK(output);
    kiss_fftr(forward_config_, reinterpret_cast<const kiss_fft_scalar*>(input),
              reinterpret_cast<::kiss_fft_cpx*>(output));
  }

  // Inverse complex-to-real FFT.
  // Caller should allocate buffer the size of EfficientRealSize().
  void InverseTransform(const std::complex<double>* input,
                        double* output) const {
    CHECK(input);
    CHECK(output);
    kiss_fftri(inverse_config_,
               reinterpret_cast<const ::kiss_fft_cpx*>(input),
               reinterpret_cast<kiss_fft_scalar*>(output));
  }

  KissFftWrapper(const KissFftWrapper&) = delete;
  KissFftWrapper& operator=(const KissFftWrapper&) = delete;

 private:
  // Gets next efficient even size.
  int GetEfficientSize(int n) const {
    return (::kiss_fft_next_fast_size(((n) + 1) >> 1) << 1);
  }
  const int real_size_;
  const int complex_size_;
  const ::kiss_fftr_cfg forward_config_;
  const ::kiss_fftr_cfg inverse_config_;
};
}  // namespace accounting
}  // namespace differential_privacy
#endif  // DIFFERENTIAL_PRIVACY_ACCOUNTING_CPP_KISS_FFT_WRAPPER_H_
