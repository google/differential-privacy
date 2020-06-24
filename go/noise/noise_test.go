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
	"testing"
)

var (
	ln2 = math.Log(2)
	ln3 = math.Log(3)

	lap   = Laplace()
	gauss = Gaussian()
)

func nearEqual(a, b, maxError float64) bool {
	return math.Abs(a-b) < maxError
}

var benchResultFloat64 float64

func BenchmarkLaplaceFloat64(b *testing.B) {
	var r float64
	for i := 0; i < b.N; i++ {
		r = lap.AddNoiseFloat64(42, 1, 1, ln3, 0)
	}
	benchResultFloat64 = r
}
