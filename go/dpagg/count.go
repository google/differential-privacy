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

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/noise"
)

// Count calculates a differentially private count of a collection of values
// using the Laplace or Gaussian mechanism
//
// It supports privacy units that contribute to multiple partitions (via the
// MaxPartitionsContributed parameter) by scaling the added noise appropriately.
// However, it does not support multiple contributions to a single partition from
// the same privacy unit. For that use case, BoundedSumInt64 should be used instead.
//
// The provided differentially private count is an unbiased estimate of the raw
// count meaning that its expected value is equal to the raw count.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for int64 values.
// This aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type Count struct {
	// Parameters
	epsilon         float64
	delta           float64
	l0Sensitivity   int64
	lInfSensitivity int64
	noise           noise.Noise
	noiseKind       noise.Kind // necessary for serializing noise.Noise information

	// State variables
	count          int64
	state          state 
	noisedCount    int64
}

func countEquallyInitialized(c1, c2 *Count) bool {
	return c1.epsilon == c2.epsilon &&
		c1.delta == c2.delta &&
		c1.l0Sensitivity == c2.l0Sensitivity &&
		c1.lInfSensitivity == c2.lInfSensitivity &&
		c1.noiseKind == c2.noiseKind
}

// CountOptions contains the options necessary to initialize a Count.
type CountOptions struct {
	Epsilon                  float64     // Privacy parameter ε. Required.
	Delta                    float64     // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed int64       // How many distinct partitions may a single privacy unit contribute to? Defaults to 1.
	Noise                    noise.Noise // Type of noise used. Defaults to Laplace noise.
	// How many times may a single privacy unit contribute to a single partition?
	// Defaults to 1. This is only needed for other aggregation functions using Count;
	// which is why the option is not exported.
	maxContributionsPerPartition int64
}

// NewCount returns a new Count, initialized at 0.
func NewCount(opt *CountOptions) *Count {
	if opt == nil {
		opt = &CountOptions{}
	}
	// Set defaults.
	l0 := opt.MaxPartitionsContributed
	if l0 == 0 {
		l0 = 1
	}

	lInf := opt.maxContributionsPerPartition
	if lInf == 0 {
		lInf = 1
	}

	n := opt.Noise
	if n == nil {
		n = noise.Laplace()
	}
	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some dummy value.
	eps, del := opt.Epsilon, opt.Delta
	n.AddNoiseInt64(0, l0, lInf, eps, del)

	return &Count{
		epsilon:         eps,
		delta:           del,
		l0Sensitivity:   l0,
		lInfSensitivity: lInf,
		noise:           n,
		noiseKind:       noise.ToKind(n),
		count:           0,
		state:           Default,
	}
}

// Increment increments the count by one.
func (c *Count) Increment() {
	c.IncrementBy(1)
}

// IncrementBy increments the count by the given value.
// Note that this shouldn't be used to count multiple contributions to a
// single partition from the same privacy unit.
func (c *Count) IncrementBy(count int64) {
	if c.state != Default {
		log.Fatalf("Count cannot be amended. Reason: %v", c.state.errorMessage("Count"))
	}
	c.count += count
}

// Merge merges c2 into c (i.e., adds to c all entries that were added to c2).
// c2 is consumed by this operation: it may not be used after it is merged
// into c.
func (c *Count) Merge(c2 *Count) {
	if e := checkMergeCount(c, c2); e != nil {
		log.Exit(e)
	}
	c.count += c2.count
	c2.state = Merged
}

func checkMergeCount(c1, c2 *Count) error {
	if c1.state != Default {
		return fmt.Errorf("checkMergeCount: c1 cannot be merged with another Count instance. Reason: %v", c1.state.errorMessage("Count"))
	}
	if c2.state != Default {
		return fmt.Errorf("checkMergeCount: c2 cannot be merged with another Count instance. Reason: %v", c2.state.errorMessage("Count"))
	}

	if !countEquallyInitialized(c1, c2) {
		return fmt.Errorf("checkMergeCount: c1 and c2 are not compatible")
	}

	return nil
}

// Result returns a differentially private estimate of the current count. The
// method can be called only once.
//
// The returned value is an unbiased estimate of the raw count.
//
// The returned value may sometimes be negative. This can be corrected by setting
// negative results to 0. Note that such post processing introduces bias to the
// result.
func (c *Count) Result() int64 {
	if c.state != Default {
		log.Fatalf("The noised result cannot be computed. Reason: " + c.state.errorMessage("Count"))
	}
	c.state = ResultReturned
	c.noisedCount = c.noise.AddNoiseInt64(c.count, c.l0Sensitivity, c.lInfSensitivity, c.epsilon, c.delta)
	return c.noisedCount
}

// ThresholdedResult is similar to Result() but applies thresholding to the
// result. So, if the result is less than the threshold specified by the noise
// mechanism, it returns nil. Otherwise, it returns the result.
func (c *Count) ThresholdedResult(thresholdDelta float64) *int64 {
	threshold := c.noise.Threshold(c.l0Sensitivity, float64(c.lInfSensitivity), c.epsilon, c.delta, thresholdDelta)
	result := c.Result()
	if result < int64(threshold) {
		return nil
	}
	return &result
}

// ComputeConfidenceInterval computes a confidence interval with integer bounds that
// contains the true count with a probability greater than or equal to 1 - alpha using
// the noised count computed by Result(). The computation is based exclusively on the
// noised count returned by Result(). Thus no privacy budget is consumed by this operation.
//
// Result() needs to be called before ComputeConfidenceInterval, otherwise this will return
// an error.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (c *Count) ComputeConfidenceInterval(alpha float64) (noise.ConfidenceInterval, error) {
	if c.state != ResultReturned {
		return noise.ConfidenceInterval{}, fmt.Errorf("Result() must be called before calling ComputeConfidenceInterval()")
	}
	confInt, err := c.noise.ComputeConfidenceIntervalInt64(c.noisedCount, c.l0Sensitivity, c.lInfSensitivity, c.epsilon, c.delta, alpha)
	if err != nil {
		return noise.ConfidenceInterval{}, err
	}
	// True count cannot be negative.
	confInt.LowerBound, confInt.UpperBound = math.Max(0, confInt.LowerBound), math.Max(0, confInt.UpperBound)
	return confInt, nil
}

// encodableCount can be encoded by the gob package.
type encodableCount struct {
	Epsilon         float64
	Delta           float64
	L0Sensitivity   int64
	LInfSensitivity int64
	NoiseKind       noise.Kind
	Count           int64
}

// GobEncode encodes Count.
func (c *Count) GobEncode() ([]byte, error) {
	if c.state != Default {
		return nil, fmt.Errorf("Count object cannot be serialized. Reason: " + c.state.errorMessage("Count"))
	}
	enc := encodableCount{
		Epsilon:         c.epsilon,
		Delta:           c.delta,
		L0Sensitivity:   c.l0Sensitivity,
		LInfSensitivity: c.lInfSensitivity,
		NoiseKind:       noise.ToKind(c.noise),
		Count:           c.count,
	}
	c.state = Serialized
	return encode(enc)
}

// GobDecode decodes Count.
func (c *Count) GobDecode(data []byte) error {
	var enc encodableCount
	err := decode(&enc, data)
	if err != nil {
		log.Fatalf("GobDecode: couldn't decode Count from bytes")
		return err
	}
	*c = Count{
		epsilon:         enc.Epsilon,
		delta:           enc.Delta,
		l0Sensitivity:   enc.L0Sensitivity,
		lInfSensitivity: enc.LInfSensitivity,
		noiseKind:       enc.NoiseKind,
		noise:           noise.ToNoise(enc.NoiseKind),
		count:           enc.Count,
	}
	return nil
}