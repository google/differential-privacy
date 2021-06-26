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

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/noise"
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
	MaxPartitionsContributed     int64   // How many distinct partitions may a single user contribute to? Defaults to 1.
	MaxContributionsPerPartition int64   // How many times may a single user contribute to a single partition? Required.
	// Lower and Upper bounds for clamping. Default to 0; must be such that Lower < Upper.
	Lower, Upper float64
	Noise        noise.Noise // Type of noise used in BoundedStandardDeviation. Defaults to Laplace noise.
}

// NewBoundedStandardDeviation returns a new BoundedStandardDeviation.
func NewBoundedStandardDeviation(opt *BoundedStandardDeviationOptions) *BoundedStandardDeviation {
	variance := NewBoundedVariance(&BoundedVarianceOptions{
		Epsilon:                      opt.Epsilon,
		Delta:                        opt.Delta,
		MaxPartitionsContributed:     opt.MaxPartitionsContributed,
		Lower:                        opt.Lower,
		Upper:                        opt.Upper,
		Noise:                        opt.Noise,
		MaxContributionsPerPartition: opt.MaxContributionsPerPartition,
	})

	return &BoundedStandardDeviation{
		Variance: *variance,
		state:    defaultState,
	}
}

// Add an entry to a BoundedStandardDeviation. It skips NaN entries and doesn't count
// them in the final result because introducing even a single NaN entry will result in
// a NaN standard deviation regardless of other entries, which would break the
// indistinguishability property required for differential privacy.
func (bstdv *BoundedStandardDeviation) Add(e float64) {
	if bstdv.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("BoundedStandardDeviation cannot be amended. Reason: %v", bstdv.state.errorMessage())
	}
	bstdv.Variance.Add(e)
}

// Result returns a differentially private estimate of the standard deviation of bounded
// elements added so far. The method can be called only once.
//
// Note that the returned value is not an unbiased estimate of the raw bounded standard
// deviation.
func (bstdv *BoundedStandardDeviation) Result() float64 {
	if bstdv.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("StandardDeviation's noised result cannot be computed. Reason: " + bstdv.state.errorMessage())
	}
	bstdv.state = resultReturned
	return math.Sqrt(bstdv.Variance.Result())
}

// Merge merges bstdv2 into bstdv (i.e., adds to bstdv all entries that were added to
// bstdv2). bstdv2 is consumed by this operation: bstdv2 may not be used after it is
// merged into bstdv.
func (bstdv *BoundedStandardDeviation) Merge(bstdv2 *BoundedStandardDeviation) {
	if err := checkMergeBoundedStandardDeviation(bstdv, bstdv2); err != nil {
		// TODO: do not exit the program from within library code
		log.Exit(err)
	}
	bstdv.Variance.Merge(&bstdv2.Variance)
	bstdv2.state = merged
}

func checkMergeBoundedStandardDeviation(bstdv1, bstdv2 *BoundedStandardDeviation) error {
	if bstdv1.state != defaultState {
		return fmt.Errorf("checkMergeBoundedStandardDeviation: bv1 cannot be merged with another BoundedStandardDeviation instance. Reason: %v", bstdv1.state.errorMessage())
	}
	if bstdv2.state != defaultState {
		return fmt.Errorf("checkMergeBoundedStandardDeviation: bv2 cannot be merged with another BoundedStandardDeviation instance. Reason: %v", bstdv2.state.errorMessage())
	}

	if !bstdvEquallyInitialized(bstdv1, bstdv2) {
		return fmt.Errorf("checkMergeBoundedStandardDeviation: bstdv1 and bstdv2 are not compatible")
	}

	return nil
}

// GobEncode encodes BoundedStandardDeviation.
func (bstdv *BoundedStandardDeviation) GobEncode() ([]byte, error) {
	if bstdv.state != defaultState {
		return nil, fmt.Errorf("StandardDeviation object cannot be serialized. Reason: " + bstdv.state.errorMessage())
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
		log.Fatalf("GobDecode: couldn't decode BoundedStandardDeviation from bytes")
		return err
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
