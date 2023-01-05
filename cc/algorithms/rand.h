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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_RAND_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_RAND_H_

#include <cstdint>
#include <limits>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace differential_privacy {

// Generates a double-valued random number of Uniform[0, 1). This has the same
// distribution as generating a uniform real r in (0, 1) and then returning the
// largest double value less than or equal to r.
double UniformDouble();

// geometric returns a number randomly picked from a geometric distribution of
// parameter 0.5. Will not exceed 1025.
uint64_t Geometric();

class SecureURBG {
 public:
  using result_type = uint64_t;
  static SecureURBG& GetInstance();

  static constexpr result_type(min)() {
    return (std::numeric_limits<result_type>::min)();
  }
  static constexpr result_type(max)() {
    return (std::numeric_limits<result_type>::max)();
  }
  result_type operator()() ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  SecureURBG() { buffer_ = new uint8_t[kBufferSize]; }
  ~SecureURBG() { delete[] buffer_; }
  // Refresh the cache with new random bytes.
  void RefreshBuffer() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static constexpr int kBufferSize = 65536;
  // The current index in the cache.
  int current_index_ ABSL_GUARDED_BY(mutex_) = kBufferSize;
  uint8_t* buffer_ ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;
};
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_RAND_H_
