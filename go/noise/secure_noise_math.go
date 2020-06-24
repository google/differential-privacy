//
// Copyright 2020 Google LLC
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

package noise

import (
	"math"
)

// ceilPowerOfTwo returns the smallest power of 2 larger or equal to x. The
// value of x must be a finite positive number not greater than 2^1023. The
// result of this method is guaranteed to be an exact power of 2.
func ceilPowerOfTwo(x float64) float64 {
	if x <= 0.0 || math.IsInf(x, 0) || math.IsNaN(x) {
		return math.NaN()
	}

	// The following bit masks are based on the bit layout of float64 values,
	// which according to the IEEE 754 standard is defined as "1*s 11*e 52*m"
	// where "s" is the sign bit, "e" are the exponent bits, and "m" are the
	// mantissa bits.
	var exponentMask uint64 = 0x7ff0000000000000
	var mantissaMask uint64 = 0x000fffffffffffff

	bits := math.Float64bits(x)
	mantissaBits := bits & mantissaMask

	// Since x is a finite positive number, x is a power of 2 if and only if
	// it has a mantissa of 0.
	if mantissaBits == 0x0000000000000000 {
		return x
	}

	exponentBits := bits & exponentMask
	maxExponentBits := math.Float64bits(math.MaxFloat64) & exponentMask

	if exponentBits >= maxExponentBits {
		// Input is too large.
		return math.NaN()
	}

	// Increasing the exponent by 1 to get the next power of 2. This is done by
	// adding 0x0010000000000000 to the exponent bits, which will keep a mantissa
	// of all 0s.
	return math.Float64frombits(exponentBits + 0x0010000000000000)
}

// roundToMultipleOfPowerOfTwo returns a multiple of granularity that is
// closest to x. The value of granularity needs to be an exact power of 2,
// otherwise the result might not be exact.
func roundToMultipleOfPowerOfTwo(x, granularity float64) float64 {
	return math.Round(x/granularity) * granularity
}
