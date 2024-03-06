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

	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/go/v3/rand"
)

// PreAggSelectPartition is used to compute an (ε,δ)-differentially private decision
// of whether to materialize a partition.
//
// Many differential privacy mechanisms work by performing an aggregation and
// adding noise. They achieve (ε_m,δ_m)-differential privacy under the
// assumption that partitions are chosen in advance. In other words, they assume that
// even if no data is associated with a partition, noise is added to the empty
// aggregation, and the noisy result is materialized. However, when only
// partitions containing data are materialized, such mechanisms fail to protect
// privacy for partitions containing data from a single privacy ID (e.g.,
// user). To fix this, partitions with small numbers of privacy IDs must
// sometimes be dropped in order to maintain privacy. This process of partition
// selection incurs an additional (ε,δ) differential privacy budget resulting
// in a total differential privacy budget of (ε+ε_m,δ+δ_m) being used for the
// aggregation with partition selection.
//
// Depending on the l0sensitivity, the PreAggSelectPartition uses one of two
// differentially private partition selection algorithms.
//
// When l0sensitivity ≤ 3, the partition selection process is made (ε,δ)
// differentially private by applying the definition of differential privacy to
// the count of privacy IDs. Supposing l0Sensitivity bounds the number of partitions
// a privacy ID may contribute to, we define:
//
//	pε := ε/l0Sensitivity
//	pδ := δ/l0Sensitivity
//
// to be the per-partition differential privacy losses incurred by the partition
// selection process. Letting n denote the number of privacy IDs in a partition,
// the probability of selecting a partition is given by the following recurrence
// relation:
//
//	keepPartitionProbability(n) = min(
//	        keepPartitionProbability(n-1) * exp(pε) + pδ,              (1)
//	        1 - exp(-pε) * (1-keepPartitionProbability(n-1)-pδ),       (2)
//	        1                                                   (3)
//	    )
//
// with base case keepPartitionProbability(0) = 0. This formula is optimal in terms of
// maximizing the likelihood of selecting a partition under (ε,δ)-differential
// privacy, with the caveat that the input values for pε and pδ are lower bound
// approximations. For efficiency, we use a closed-form solution to this
// recurrence relation. See [Differentially private partition selection paper]
// https://arxiv.org/pdf/2006.03684.pdf for details on the underlying mathematics.
//
// When l0sensitivity > 3, the partition selection process is made (ε,δ)
// differentially private by using the ThresholdedResult() of the Count primitive
// with Gaussian noise. Count computes a (ε,δ/2) differentially private count of
// the privacy IDs in a partition by adding Gaussian noise. Then, it computes
// a threshold T for which the probability that a (ε,δ/2) differentially private
// count of a single privacy ID can exceed T is δ/2. It keeps the partition iff
// differentially private count exceeds the threshold.
//
// The reason two different algorithms for deciding whether to keep a partition
// is used is because the first algorithm ("magic partition selection") is optimal
// when l0sensitivity ≤ 3 but is outperformed by Gaussian-based thresholding when
// l0sensitivity > 3.
//
// PreAggSelectPartition is a utility for maintaining the count of IDs in a single
// partition and then determining whether the partition should be
// materialized. Use Increment() to increment the count of IDs and ShouldKeepPartition() to decide
// if the partition should be materialized.
//
// PreAggSelectPartition also supports doing additional pre-thresholding on top of the
// differentially private partition selection via the PreThreshold in PreAggSelectPartitionOptions.
//
// See https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
// for more information.
type PreAggSelectPartition struct {
	// parameters
	epsilon       float64
	delta         float64
	l0Sensitivity int64
	preThreshold  int64

	// State variables
	// idCount is the count of unique privacy IDs in the partition.
	idCount int64
	state   aggregationState
}

func preAggSelectPartitionEquallyInitialized(s1, s2 *PreAggSelectPartition) bool {
	return s1.epsilon == s2.epsilon &&
		s1.delta == s2.delta &&
		s1.l0Sensitivity == s2.l0Sensitivity &&
		s1.state == s2.state &&
		s1.preThreshold == s2.preThreshold
}

// PreAggSelectPartitionOptions is used to set the privacy parameters when
// constructing a PreAggSelectPartition.
type PreAggSelectPartitionOptions struct {
	// Epsilon and Delta specify the (ε,δ)-differential privacy budget used for
	// partition selection. Required.
	Epsilon float64
	Delta   float64
	// An additional thresholding that is performed in combination with private
	// partition selection to ensure that each partition has at least a
	// PreThreshold number of unique contributions.
	//
	// See https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
	// for more information.
	// Optional.
	PreThreshold int64
	// MaxPartitionsContributed is the number of distinct partitions a single
	// privacy unit can contribute to. Required.
	MaxPartitionsContributed int64
}

// NewPreAggSelectPartition constructs a new PreAggSelectPartition from opt.
func NewPreAggSelectPartition(opt *PreAggSelectPartitionOptions) (*PreAggSelectPartition, error) {
	if opt == nil {
		opt = &PreAggSelectPartitionOptions{} // Prevents panicking due to a nil pointer dereference.
	}

	if err := checks.CheckPreThreshold(opt.PreThreshold); err != nil {
		return nil, fmt.Errorf("NewPreAggSelectPartition: %w", err)
	}
	// Set PreThreshold to default 1 if not specified.
	if opt.PreThreshold < 1 {
		opt.PreThreshold = 1
	}

	s := PreAggSelectPartition{
		epsilon:       opt.Epsilon,
		delta:         opt.Delta,
		preThreshold:  opt.PreThreshold,
		l0Sensitivity: opt.MaxPartitionsContributed,
	}

	if err := checks.CheckDeltaStrict(s.delta); err != nil {
		return nil, fmt.Errorf("NewPreAggSelectPartition: %v", err)
	}
	// ε=0 is theoretically acceptable, but in practice it's probably an error,
	// so we do not accept it as argument.
	if err := checks.CheckEpsilonStrict(s.epsilon); err != nil {
		return nil, fmt.Errorf("NewPreAggSelectPartition: %v", err)
	}
	if err := checks.CheckL0Sensitivity(s.l0Sensitivity); err != nil {
		return nil, fmt.Errorf("NewPreAggSelectPartition: %v", err)
	}
	return &s, nil
}

// Increment increments the ids count by one.
// The caller must ensure this method is called at most once per privacy ID.
func (s *PreAggSelectPartition) Increment() error {
	return s.IncrementBy(1)
}

// IncrementBy increments the ids count by the given value.
// Note that this shouldn't be used to count multiple contributions to a
// single partition from the same privacy unit.
//
// It could, for example, be used to increment the ids count by k privacy
// units at once.
//
// Note that decrementing counts by inputting a negative value is allowed,
// for example if you want to remove some users you have previously added.
func (s *PreAggSelectPartition) IncrementBy(count int64) error {
	if s.state != defaultState {
		return fmt.Errorf("PreAggSelectPartition cannot be amended: %v", s.state.errorMessage())
	}
	s.idCount += count
	return nil
}

// Merge merges s2 into s (i.e., add the idCount of s2 to s). This implicitly
// assumes that s and s2 act on distinct privacy IDs. s2 is consumed by this
// operation: s2 may not be used after it is merged into s.
//
// Preconditions: s and s2 must have the same privacy parameters. In addition,
// ShouldKeepPartition() may not be called yet for either s or s2.
func (s *PreAggSelectPartition) Merge(s2 *PreAggSelectPartition) error {
	if err := checkMergePreAggSelectPartition(s, s2); err != nil {
		return err
	}

	s.idCount += s2.idCount
	s2.state = merged
	return nil
}

func checkMergePreAggSelectPartition(s1, s2 *PreAggSelectPartition) error {
	if s1.state != defaultState {
		return fmt.Errorf("checkMergePreAggSelectPartition: s1 cannot be merged with another PreAggSelectPartition instance: %v", s1.state.errorMessage())
	}
	if s2.state != defaultState {
		return fmt.Errorf("checkMergePreAggSelectPartition: s2 cannot be merged with another PreAggSelectPartition instance: %v", s2.state.errorMessage())
	}

	if !preAggSelectPartitionEquallyInitialized(s1, s2) {
		return fmt.Errorf("checkMergePreAggSelectPartition: s1 and s2 are not compatible")
	}

	return nil
}

// ShouldKeepPartition returns whether the partition should be materialized.
func (s *PreAggSelectPartition) ShouldKeepPartition() (bool, error) {
	if s.state != defaultState {
		return false, fmt.Errorf("PreAggSelectPartition's ShouldKeepPartition cannot be computed: %v", s.state.errorMessage())
	}
	s.state = resultReturned

	// Pre-thresholding guarantees that at least this number of unique contributions are in the
	// partition.
	if s.idCount < s.preThreshold {
		return false, nil
	}
	// PreThreshold is set to 1 as the default, subtract it here so it has no effect.
	// This subtraction also ensures that idCount will always be > 0 if preThreshold = idsCount.
	s.idCount = s.idCount - (s.preThreshold - 1)

	if s.l0Sensitivity > 3 { // Gaussian thresholding outperforms in this case.
		c, err := NewCount(&CountOptions{
			Epsilon:                  s.epsilon,
			Delta:                    s.delta / 2,
			MaxPartitionsContributed: s.l0Sensitivity,
			Noise:                    noise.Gaussian()})
		if err != nil {
			return false, fmt.Errorf("couldn't initialize count for PreAggSelectPartition: %v", err)
		}
		err = c.IncrementBy(s.idCount)
		if err != nil {
			return false, fmt.Errorf("couldn't increment count for PreAggSelectPartition: %v", err)
		}
		result, err := c.ThresholdedResult(s.delta / 2)
		if err != nil {
			return false, fmt.Errorf("couldn't compute thresholded result for PreAggSelectPartition: %v", err)
		}
		return result != nil, nil
	}
	prob, err := keepPartitionProbability(s.idCount, s.l0Sensitivity, s.epsilon, s.delta)
	if err != nil {
		return false, fmt.Errorf("couldn't compute keepPartitionProbability for PreAggSelectPartition: %v", err)
	}
	return rand.Uniform() < prob, nil
}

// sumExpPowers returns the evaluation of
//
//	exp(minPower * ε) + exp((minPower+1) * ε) + ... + exp((numPowers+minPower-1) * ε)
//
// sumExpPowers requires ε >= 0. sumExpPowers may return +∞, but does not return
// NaN.
func sumExpPowers(epsilon float64, minPower, numPowers int64) (float64, error) {
	if err := checks.CheckEpsilon(epsilon); err != nil {
		return 0, fmt.Errorf("sumExpPowers: %v", err)
	}
	if numPowers <= 0 {
		return 0, fmt.Errorf("sumExpPowers: numPowers (%d) must be > 0", numPowers)
	}

	// expEpsilonPow returns exp(a * epsilon).
	expEpsilonPow := func(a int64) float64 {
		return math.Exp(float64(a) * epsilon)
	}

	if math.IsInf(expEpsilonPow(minPower), 1) {
		return math.Inf(1), nil
	}
	// In the case ε=0, sumExpPowers is simply numPowers. We use exp(-ε) = 1 to
	// identify this case because our closed form solutions would otherwise
	// result in division by 0 under finite precision arithmetic.
	if math.Exp(-epsilon) == 1 {
		return float64(numPowers), nil
	}

	// For the general case, we use a closed form solution to a geometric
	// series. See https://en.wikipedia.org/wiki/Geometric_series#Sum.
	//
	// The typical closed form formula is:
	//   exp(minPower*ε) * (exp(numPowers*ε) - 1) / (exp(ε) - 1)
	//
	// In our setting, it is OK to return +∞ but not OK to return NaN. We use the
	// following mathematically equivalent formulas to avoid returning NaN when
	// using our finite precision arithmetic:
	//   exp((minPower-1)*ε) * (exp(numPowers*ε) - 1) / (1 - exp(-ε))               (1)
	//   (exp((numPowers+minPower-1)*ε) - exp((minPower-1)*ε)) / (1 - exp(-ε))      (2)
	//
	// We use (1) when minPower >= 1. In that case, (exp(numPowers*ε) - 1) is
	// the only potentially infinite term. The other two terms satisfy
	// exp((minPower-1) * ε) >= 1 and 0 < (1 - exp(-ε)) <= 1. The
	// multiplication and division operations involving these other two
	// terms increase the result. Thus, the result is never NaN, and +∞ is
	// returned only when necessary.
	//
	// We use (2) when minPower < 1. In that case, 0 <= exp((minPower-1)*ε) < 1,
	// so the numerator
	//   (exp((numPowers+minPower-1)*ε) - exp((minPower-1)*ε))
	// is never NaN, and is only +∞ when necessary. The denominator in (2)
	// satisfies 0 < (1-exp(-ε)) <= 1. The result of (2) is never NaN and
	// only achieves +∞ when necessary overall.
	if minPower >= 1 {
		return expEpsilonPow(minPower-1) * (expEpsilonPow(numPowers) - 1) / (1 - expEpsilonPow(-1)), nil
	}
	return (expEpsilonPow(numPowers+minPower-1) - expEpsilonPow(minPower-1)) / (1 - expEpsilonPow(-1)), nil
}

// keepPartitionProbability calculates the value of keepPartitionProbability
// from PreAggSelectPartition's godoc comment.
func keepPartitionProbability(idCount, l0Sensitivity int64, epsilon, delta float64) (float64, error) {
	if idCount <= 0 {
		return 0, nil
	}
	// Per-partition (ε,δ) privacy loss.
	pEpsilon := epsilon / float64(l0Sensitivity)
	pDelta := delta / float64(l0Sensitivity)

	// In keepPartitionProbability's recurrence formula (see Theorem 1 in the
	// [Differentially private partition selection paper]), argument selection in
	// the min operation has 3 distinct regions: min selects (1) on the lowest
	// region of the domain, (2) on the second region of the domain, and (3)
	// (i.e., the value 1) on the highest region of the domain. We denote by nCr
	// the crossover point in the domain from (1) to (2).
	nCr := int64(1 + math.Floor((1/pEpsilon)*math.Log(
		(1+math.Exp(-pEpsilon)*(2*pDelta-1))/(pDelta*(1+math.Exp(-pEpsilon))))))

	if idCount <= nCr {
		// Closed form solution of keepPartitionProbability(n) on [0, nCr].
		sum, err := sumExpPowers(pEpsilon, 0, idCount)
		if err != nil {
			return 0, err
		}
		return pDelta * sum, nil
	}

	sum, err := sumExpPowers(pEpsilon, 0, nCr)
	if err != nil {
		return 0, err
	}
	selectPartitionPrNCr := pDelta * sum
	// Compute form solution of keepPartitionProbability(n) on the domain (nCr, ∞).
	m := idCount - nCr
	sum, err = sumExpPowers(pEpsilon, -m, m)
	if err != nil {
		return 0, err
	}
	return math.Min(
		1+math.Exp(-float64(m)*pEpsilon)*(selectPartitionPrNCr-1)+sum*pDelta,
		1), nil
}

// GetHardThreshold returns a threshold k, where if there are at least
// k privacy units in a partition, we are guaranteed to keep that partition.
//
// This is the conceptual equivalent of the post-aggregation threshold of the
// noise.Noise interface with the difference that here there is 0 probability
// of not keeping the partition if it has at least k privacy units, whereas
// with the post-aggregation threshold there is a non-zero probability
// (however small).
func (s *PreAggSelectPartition) GetHardThreshold() (int, error) {
	for i := int64(1); ; i++ {
		prob, err := keepPartitionProbability(i, s.l0Sensitivity, s.epsilon, s.delta)
		if err != nil {
			return 0, err
		}
		if prob >= 1 { // keepPartitionProbability converges to 1.
			return int(i), nil
		}
	}
}

// encodablePreAggSelectPartition can be encoded by the gob package.
type encodablePreAggSelectPartition struct {
	Epsilon       float64
	Delta         float64
	L0Sensitivity int64
	IDCount       int64
	State         aggregationState
	PreThreshold  int64
}

// GobEncode encodes PreAggSelectPartition.
func (s *PreAggSelectPartition) GobEncode() ([]byte, error) {
	if s.state != defaultState && s.state != serialized {
		return nil, fmt.Errorf("PreAggSelectPartition object cannot be serialized: " + s.state.errorMessage())
	}
	enc := encodablePreAggSelectPartition{
		Epsilon:       s.epsilon,
		Delta:         s.delta,
		L0Sensitivity: s.l0Sensitivity,
		IDCount:       s.idCount,
		State:         s.state,
		PreThreshold:  s.preThreshold,
	}
	s.state = serialized
	return encode(enc)
}

// GobDecode decodes PreAggSelectPartition.
func (s *PreAggSelectPartition) GobDecode(data []byte) error {
	var enc encodablePreAggSelectPartition
	err := decode(&enc, data)
	if err != nil {
		return fmt.Errorf("couldn't decode PreAggSelectPartition from bytes")
	}
	*s = PreAggSelectPartition{
		epsilon:       enc.Epsilon,
		delta:         enc.Delta,
		l0Sensitivity: enc.L0Sensitivity,
		idCount:       enc.IDCount,
		state:         enc.State,
		preThreshold:  enc.PreThreshold,
	}
	return nil
}
