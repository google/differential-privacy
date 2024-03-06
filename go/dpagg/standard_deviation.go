//
// Copyright 2021 Google LLC
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

	"github.com/google/differential-privacy/go/v3/noise"
)

// BoundedStandardDeviation calculates a differentially private standard deviation
// of a collection of float64 values.
//
// The output will be clamped between 0 and (upper - lower).
//
// The implementation simply computes the bounded variance and takes the square
// root, which is differentially private by the post-processing theorem. It
// relies on the fact that the bounded variance algorithm guarantees that the
// output is non-negative.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for float64 values. This
// aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type BoundedStandardDeviation struct {
	// State variables
	Variance BoundedVariance
	state    aggregationState
}

func bstdvEquallyInitialized(bstdv1, bstdv2 *BoundedStandardDeviation) bool {
	return bstdv1.state == bstdv2.state &&
		bvEquallyInitialized(&bstdv1.Variance, &bstdv2.Variance)
}

// BoundedStandardDeviationOptions contains the options necessary to initialize a BoundedStandardDeviation.
type BoundedStandardDeviationOptions struct {
	Epsilon                      float64 // Privacy parameter ε. Required.
	Delta                        float64 // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed     int64   // How many distinct partitions may a single user contribute to? Required.
	MaxContributionsPerPartition int64   // How many times may a single user contribute to a single partition? Required.
	// Lower and Upper bounds for clamping. Required; must be such that Lower < Upper.
	Lower, Upper float64
	Noise        noise.Noise // Type of noise used in BoundedStandardDeviation. Defaults to Laplace noise.
}

// NewBoundedStandardDeviation returns a new BoundedStandardDeviation.
func NewBoundedStandardDeviation(opt *BoundedStandardDeviationOptions) (*BoundedStandardDeviation, error) {
	variance, err := NewBoundedVariance(&BoundedVarianceOptions{
		Epsilon:                      opt.Epsilon,
		Delta:                        opt.Delta,
		MaxPartitionsContributed:     opt.MaxPartitionsContributed,
		Lower:                        opt.Lower,
		Upper:                        opt.Upper,
		Noise:                        opt.Noise,
		MaxContributionsPerPartition: opt.MaxContributionsPerPartition,
	})
	if err != nil {
		return nil, fmt.Errorf("couldn't initialize BoundedVariance for NewBoundedStandardDeviation: %w", err)
	}

	return &BoundedStandardDeviation{
		Variance: *variance,
		state:    defaultState,
	}, nil
}

// Add an entry to a BoundedStandardDeviation. It skips NaN entries and doesn't count
// them in the final result because introducing even a single NaN entry will result in
// a NaN standard deviation regardless of other entries, which would break the
// indistinguishability property required for differential privacy.
func (bstdv *BoundedStandardDeviation) Add(e float64) error {
	if bstdv.state != defaultState {
		return fmt.Errorf("BoundedStandardDeviation cannot be amended: %v", bstdv.state.errorMessage())
	}
	return bstdv.Variance.Add(e)
}

// Result returns a differentially private estimate of the standard deviation of bounded
// elements added so far. The method can be called only once.
//
// Note that the returned value is not an unbiased estimate of the raw bounded standard
// deviation.
func (bstdv *BoundedStandardDeviation) Result() (float64, error) {
	if bstdv.state != defaultState {
		return 0, fmt.Errorf("BoundedStandardDeviation's noised result cannot be computed: " + bstdv.state.errorMessage())
	}
	bstdv.state = resultReturned
	variance, err := bstdv.Variance.Result()
	if err != nil {
		return 0, err
	}
	return math.Sqrt(variance), nil
}

// Merge merges bstdv2 into bstdv (i.e., adds to bstdv all entries that were added to
// bstdv2). bstdv2 is consumed by this operation: bstdv2 may not be used after it is
// merged into bstdv.
func (bstdv *BoundedStandardDeviation) Merge(bstdv2 *BoundedStandardDeviation) error {
	if err := checkMergeBoundedStandardDeviation(bstdv, bstdv2); err != nil {
		return err
	}
	bstdv.Variance.Merge(&bstdv2.Variance)
	bstdv2.state = merged
	return nil
}

func checkMergeBoundedStandardDeviation(bstdv1, bstdv2 *BoundedStandardDeviation) error {
	if bstdv1.state != defaultState {
		return fmt.Errorf("checkMergeBoundedStandardDeviation: bv1 cannot be merged with another BoundedStandardDeviation instance: %v", bstdv1.state.errorMessage())
	}
	if bstdv2.state != defaultState {
		return fmt.Errorf("checkMergeBoundedStandardDeviation: bv2 cannot be merged with another BoundedStandardDeviation instance: %v", bstdv2.state.errorMessage())
	}

	if !bstdvEquallyInitialized(bstdv1, bstdv2) {
		return fmt.Errorf("checkMergeBoundedStandardDeviation: bstdv1 and bstdv2 are not compatible")
	}

	return nil
}

// GobEncode encodes BoundedStandardDeviation.
func (bstdv *BoundedStandardDeviation) GobEncode() ([]byte, error) {
	if bstdv.state != defaultState && bstdv.state != serialized {
		return nil, fmt.Errorf("StandardDeviation object cannot be serialized: " + bstdv.state.errorMessage())
	}
	enc := encodableBoundedStandardDeviation{
		EncodableVariance: &bstdv.Variance,
	}
	bstdv.state = serialized
	return encode(enc)
}

// GobDecode decodes BoundedStandardDeviation.
func (bstdv *BoundedStandardDeviation) GobDecode(data []byte) error {
	var enc encodableBoundedStandardDeviation
	err := decode(&enc, data)
	if err != nil {
		return fmt.Errorf("couldn't decode BoundedStandardDeviation from bytes")
	}
	*bstdv = BoundedStandardDeviation{
		Variance: *enc.EncodableVariance,
	}
	return nil
}

// encodableBoundedStandardDeviation can be encoded by the gob package.
type encodableBoundedStandardDeviation struct {
	EncodableVariance *BoundedVariance
}
