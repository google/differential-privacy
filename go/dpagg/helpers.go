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

package dpagg

import (
	"fmt"
	"math"
)

// LargestRepresentableDelta is the largest delta we could support in 64 bit precision, approximately equal to one.
var LargestRepresentableDelta = 1 - math.Pow(2, -53)

// ClampFloat64 clamps e within lower and upper, such that lower is returned
// if e < lower, and upper is returned if e > upper. Otherwise, e is returned.
func ClampFloat64(e, lower, upper float64) (float64, error) {
	if lower > upper {
		return 0, fmt.Errorf("lower must be less than or equal to upper, got lower = %v, upper = %v", lower, upper)
	}

	if e > upper {
		return upper, nil
	}
	if e < lower {
		return lower, nil
	}
	return e, nil
}

// ClampInt64 clamps e within lower and upper.
// Returns lower if e < lower.
// Returns upper if e > upper.
func ClampInt64(e, lower, upper int64) (int64, error) {
	if lower > upper {
		return 0, fmt.Errorf("lower must be less than or equal to upper, got lower = %v, upper = %v", lower, upper)
	}
	if e > upper {
		return upper, nil
	}
	if e < lower {
		return lower, nil
	}
	return e, nil
}
