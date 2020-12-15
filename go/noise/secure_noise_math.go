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

// roundToMultiple returns a multiple of granularity that is closest to x.
// The result is exact.
func roundToMultiple(x, granularity int64) int64 {
	result := (x / granularity) * granularity
	if x >= 0 {
		if x-result >= (result+granularity)-x {
			// round up
			return result + granularity
		}
		// round down
		return result
	}
	if x-(result-granularity) >= result-x {
		// round up
		return result
	}
	// round down
	return result - granularity
}

// nextLargerFloat64 computes the smallest float64 value that is larger than or equal to the provided int64 value.
//
// Mapping from int64 to float64 for large int64 values (> 2^53) is inaccurate since they cannot be
// represented as a float64. Implicit/explicit conversion from int64 to float64 either rounds up or
// down the int64 value to the nearest representable float64. This function ensures that int64 n <=
// float64 nextLargerFloat64(n)
func nextLargerFloat64(n int64) float64 {
	// Large int64 values n may lie between two representable float64 values a and b,
	// i.e., a < n < b, (note that in this case a and b are guaranteed to be integers).
	// If the standard conversion to float64 rounds the int64 value down, e.g. float64(n) = a,
	// the difference a - n will be negative, indicating that the result needs to be incremented
	// to the next float64 value b.
	result := float64(n)
	diff := int64(result) - n
	if diff < 0 {
		return math.Nextafter(result, math.Inf(1))
	}
	return result
}

// nextSmallerFloat64 computes the largest float64 value that is smaller than or equal to the provided int64 value.
//
// Mapping from int64 to float64 for large int64 values (> 2^53) is inaccurate since they cannot be
// represented as a float64. Implicit/explicit conversion from int64 to float64 either rounds up or
// down the int64 value to the nearest representable float64. This function ensures that int64 n >=
// float64 nextSmallerFloat64(n)
func nextSmallerFloat64(n int64) float64 {
	// Large int64 values n may lie between two representable float64 values a and b,
	// i.e., a < n < b, (note that in this case a and b are guaranteed to be integers).
	// If the standard conversion to float64 rounds the int64 value up, e.g. int64(n) = b,
	// the difference b - n will be positive, indicating that the result needs to be decremented
	// to the previous float64 value a.
	result := float64(n)
	diff := int64(result) - n
	if diff > 0 {
		return math.Nextafter(result, math.Inf(-1))
	}
	return result
}
