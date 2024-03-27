//
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "algorithms/rand.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/synchronization/mutex.h"
#include "openssl/rand.h"

namespace differential_privacy {
namespace {
// We usually expect DBL_MANT_DIG to be 53.
static_assert(DBL_MANT_DIG < 64,
              "Double mantissa must have less than 64 bits.");
static_assert(sizeof(double) == sizeof(uint64_t) &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2,
              "double representation is not IEEE 754 binary64.");
const constexpr int kMantDigits = DBL_MANT_DIG - 1;
const constexpr uint64_t kMantissaMask = (uint64_t{1} << kMantDigits) - 1ULL;

ABSL_CONST_INIT absl::Mutex global_mutex(absl::kConstInit);
}  // namespace

double UniformDouble() {
  uint64_t uint_64_number = SecureURBG::GetInstance()();
  // A random integer of Uniform[0, 2^kMantDigits).
  uint64_t i = uint_64_number & kMantissaMask;

  // Instead of throwing the leading 12 bits away, we use them to create
  // geometric random number.
  uint64_t j = uint_64_number >> kMantDigits;

  // exponent is the number of leading zeros in the first 11 bits plus one.
  uint64_t exponent = absl::countl_zero(j) - kMantDigits + 1;

  // Extra geometric sampling is needed only when the leading 11 bits are all 0.
  if (j == 0) {
    exponent += Geometric() - 1;
  }

  j = (uint64_t{1023} - exponent) << kMantDigits;
  if (ABSL_PREDICT_FALSE(exponent >= 1023)) {
    // Denormalized value. Extremely improbable.
    j = 0;
  }
  // Addition instead of bitwise or since the carry overflow increments the
  // floating point exponent, which is exactly what we want.
  i += j;
  double r;
  std::memcpy(&r, &i, sizeof(r));
  return r;
}

uint64_t Geometric() {
  uint64_t result = 1;
  uint64_t r = 0;
  while (r == 0 && result < 1023) {
    r = SecureURBG::GetInstance()();
    result += absl::countl_zero(r);
  }
  return result;
}

SecureURBG& SecureURBG::GetInstance() {

  static auto* kInstance = new SecureURBG;
  return *kInstance;
}

SecureURBG::result_type SecureURBG::operator()() {
  absl::WriterMutexLock lock(&mutex_);
  if (current_index_ + sizeof(result_type) > kBufferSize) {
    RefreshBuffer();
  }
  int old_index = current_index_;
  current_index_ += sizeof(result_type);
  result_type result;
  std::memcpy(&result, buffer_ + old_index, sizeof(result_type));
  return result;
}

void SecureURBG::RefreshBuffer() {
  int one_on_success = 0;
  {
    absl::MutexLock lock(&global_mutex);
    one_on_success = RAND_bytes(buffer_, kBufferSize);
  }
  CHECK(one_on_success == 1)
      << "Error during buffer refresh: OpenSSL's RAND_byte is expected to "
         "return 1 on success, but returned "
      << one_on_success;
  current_index_ = 0;
}
}  // namespace differential_privacy
