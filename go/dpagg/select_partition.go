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
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/rand"
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
// The partition selection process is made (ε,δ) differentially private by
// applying the definition of differential privacy to the count of privacy
// IDs. Supposing l0Sensitivity bounds the number of partitions a privacy ID may
// contribute to, we define:
//     pε := ε/l0Sensitivity
//     pδ := δ/l0Sensitivity
// to be the per-partition differential privacy losses incurred by the partition
// selection process. Letting n denote the number of privacy IDs in a partition,
// the probability of selecting a partition is given by the following recurrence
// relation:
//   selectPartitionPr(n) = min(
//           selectPartitionPr(n-1) * exp(pε) + pδ,              (1)
//           1 - exp(-pε) * (1-selectPartitionPr(n-1)-pδ),       (2)
//           1                                                   (3)
//       )
// with base case selectPartitionPr(0) = 0. This formula is optimal in terms of
// maximizing the likelihood of selecting a partition under (ε,δ)-differential
// privacy, with the caveat that the input values for pε and pδ are lower bound
// approximations. For efficiency, we use a closed-form solution to this
// recurrence relation. See https://arxiv.org/pdf/2006.03684.pdf for details
// on the underlying mathematics.
//
// PreAggSelectPartition is a utility for maintaining the count of IDs in a single
// partition and then determining whether the partition should be
// materialized. Use Add() to increment the count of IDs and Result() to decide
// if the partition should be materialized.
type PreAggSelectPartition struct {
	// parameters
	epsilon       float64
	delta         float64
	l0Sensitivity int64

	// State variables
	// idCount is the count of unique privacy IDs in the partition.
	idCount int64
	// whether a result has already been returned / consumed for this PreAggSelectPartition
	resultReturned bool
}

func (s *PreAggSelectPartition) String() string {
	return fmt.Sprintf("&PreAggSelectPartition(epsilon %f, delta %e, l0Sensitivity %d, resultReturned %t)",
		s.epsilon, s.delta, s.l0Sensitivity, s.resultReturned)
}

// PreAggSelectPartitionOptions is used to set the privacy parameters when
// constructing a PreAggSelectPartition.
type PreAggSelectPartitionOptions struct {
	// Epsilon and Delta specify the (ε,δ)-differential privacy budget used for
	// partition selection. Required.
	Epsilon float64
	Delta   float64
	// MaxPartitionsContributed is the number of distinct partitions a single user can
	// contribute to.
	// Defaults to 1.
	MaxPartitionsContributed int64
}

// NewPreAggSelectPartition constructs a new PreAggSelectPartition from opt.
func NewPreAggSelectPartition(opt *PreAggSelectPartitionOptions) *PreAggSelectPartition {
	s := PreAggSelectPartition{
		epsilon:       opt.Epsilon,
		delta:         opt.Delta,
		l0Sensitivity: opt.MaxPartitionsContributed,
	}
	// Override the 0-default, but do not override any explicitly set (i.e., negative) values
	// for l0Sensitivity.
	if s.l0Sensitivity == 0 {
		s.l0Sensitivity = 1
	}

	if err := checks.CheckDeltaStrict("dpagg.NewPreAggSelectPartition", s.delta); err != nil {
		log.Fatalf("%s: CheckDeltaStrict failed with %v", &s, err)
	}
	if err := checks.CheckEpsilon("dpagg.NewPreAggSelectPartition", s.epsilon); err != nil {
		log.Fatalf("%s: CheckEpsilon failed with %v", &s, err)
	}
	if err := checks.CheckL0Sensitivity("dpagg.NewPreAggSelectPartition", s.l0Sensitivity); err != nil {
		log.Fatalf("%s: CheckL0Sensitivity failed with %v", &s, err)
	}
	return &s
}

// Add increments the count of privacy IDs.
func (s *PreAggSelectPartition) Add() {
	if s.resultReturned {
		log.Exitf("This PreAggSelectPartition has already returned a Result. It can only be used once.")
	}
	s.idCount++
}

// Merge merges s2 into s (i.e., add the idCount of s2 to s). This implicitly
// assumes that s and s2 act on distinct privacy IDs. s2 is consumed by this
// operation: s2 may not be used after it is merged into s.
//
// Preconditions: s and s2 must have the same privacy parameters. In addition,
// Result() may not be called yet for either s or s2.
func (s *PreAggSelectPartition) Merge(s2 *PreAggSelectPartition) {
	if err := checkMergePreAggSelectPartition(*s, *s2); err != nil {
		log.Exit(err)
	}

	s.idCount += s2.idCount
	s2.resultReturned = true
}

func checkMergePreAggSelectPartition(s PreAggSelectPartition, s2 PreAggSelectPartition) error {
	resultReturnedMsg := "checkMerge: %s already returned the result, cannot be merged with another PreAggSelectPartition instance"
	if s.resultReturned {
		return fmt.Errorf(resultReturnedMsg, "s")
	}
	if s2.resultReturned {
		return fmt.Errorf(resultReturnedMsg, "s2")
	}

	s.idCount, s2.idCount = 0, 0
	if !reflect.DeepEqual(s, s2) {
		return fmt.Errorf("s and s2 are not compatible")
	}
	return nil
}

// Result returns whether the partition should be materialized.
func (s *PreAggSelectPartition) Result() bool {
	if s.resultReturned {
		log.Exitf("This PreAggSelectPartition has already returned a Result. It can only be used once.")
	}
	s.resultReturned = true
	return rand.Uniform() < selectPartitionPr(s.idCount, s.l0Sensitivity, s.epsilon, s.delta)
}

// sumExpPowers returns the evaluation of
//   exp(minPower * ε) + exp((minPower+1) * ε) + ... + exp((numPowers+minPower-1) * ε)
//
// sumExpPowers requires ε >= 0. sumExpPowers may return +∞, but does not return
// NaN.
func sumExpPowers(epsilon float64, minPower, numPowers int64) float64 {
	if err := checks.CheckEpsilon("sumExpPowers", epsilon); err != nil {
		log.Fatalf("CheckEpsilon failed with %v", err)
	}

	// expEpsilonPow returns exp(a * epsilon).
	expEpsilonPow := func(a int64) float64 {
		return math.Exp(float64(a) * epsilon)
	}

	// Special case handling.
	if numPowers <= 0 {
		return 0
	}
	// In the case ε=0, sumExpPowers is simply numPowers. We use exp(-ε) = 1 to
	// identify this case because our closed form solutions would otherwise
	// result in division by 0 under finite precision arithmetic.
	if math.Exp(-epsilon) == 1 {
		return float64(numPowers)
	}
	if math.IsInf(expEpsilonPow(minPower), 1) {
		return math.Inf(1)
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
		return expEpsilonPow(minPower-1) * (expEpsilonPow(numPowers) - 1) / (1 - expEpsilonPow(-1))
	}
	return (expEpsilonPow(numPowers+minPower-1) - expEpsilonPow(minPower-1)) / (1 - expEpsilonPow(-1))
}

// selectPartitionPr calculates the value of selectPartitionPr from PreAggSelectPartition's
// godoc comment.
func selectPartitionPr(idCount, l0Sensitivity int64, epsilon, delta float64) float64 {
	// Per-partition (ε,δ) privacy loss.
	pEpsilon := epsilon / float64(l0Sensitivity)
	pDelta := delta / float64(l0Sensitivity)

	// Special cases
	if idCount == 0 {
		return 0
	}
	if epsilon == 0 {
		return math.Min(float64(idCount)*pDelta, 1)
	}

	// In selectPartitionPr's recurrence formula, argument selection in the min
	// operation has 3 distinct regions: min selects (1) on the lowest region of
	// the domain, (2) on the second region of the domain, and (3) (i.e., the
	// value 1) on the highest region of the domain. We denote by nCr the
	// crossover proint in the domain from (1) to (2).
	nCr := int64(1 + math.Floor((1/pEpsilon)*math.Log(
		(1+math.Exp(-pEpsilon)*(2*pDelta-1))/(pDelta*(1+math.Exp(-pEpsilon))))))

	if idCount <= nCr {
		// Closed form solution of selectPartitionPr(n) on [0, nCr].
		return pDelta * sumExpPowers(pEpsilon, 0, idCount)
	}

	selectPartitionPrNCr := pDelta * sumExpPowers(pEpsilon, 0, nCr)
	// Compute form solution of selectPartitionPr(n) on the domain (nCr, ∞).
	m := idCount - nCr
	return math.Min(
		1+math.Exp(-float64(m)*pEpsilon)*(selectPartitionPrNCr-1)+sumExpPowers(pEpsilon, -m, m)*pDelta,
		1)
}

// GetHardThreshold returns a threshold k, where if there are more than
// k users in a partition, we are guaranteed to keep that partition. This
// is the conceptual equivalent of the post-aggregation threshold of the
// noise.Noise interface.
func (s *PreAggSelectPartition) GetHardThreshold() int {
	for i := int64(1); ; i++ {
		if selectPartitionPr(i, s.l0Sensitivity, s.epsilon, s.delta) == 1 { // selectPartitionPr converges to 1.
			return int(i)
		}
	}
}

// encodablePreAggSelectPartition can be encoded by the gob package.
type encodablePreAggSelectPartition struct {
	Epsilon        float64
	Delta          float64
	L0Sensitivity  int64
	IDCount        int64
	ResultReturned bool
}

// GobEncode encodes PreAggSelectPartition.
func (s *PreAggSelectPartition) GobEncode() ([]byte, error) {
	enc := encodablePreAggSelectPartition{
		Epsilon:        s.epsilon,
		Delta:          s.delta,
		L0Sensitivity:  s.l0Sensitivity,
		IDCount:        s.idCount,
		ResultReturned: s.resultReturned,
	}
	s.resultReturned = true
	return encode(enc)
}

// GobDecode decodes PreAggSelectPartition.
func (s *PreAggSelectPartition) GobDecode(data []byte) error {
	var enc encodablePreAggSelectPartition
	err := decode(&enc, data)
	*s = PreAggSelectPartition{
		epsilon:        enc.Epsilon,
		delta:          enc.Delta,
		l0Sensitivity:  enc.L0Sensitivity,
		idCount:        enc.IDCount,
		resultReturned: enc.ResultReturned,
	}
	return err
}
