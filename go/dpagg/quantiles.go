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

	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/noise"
)

// Constants used for QuantileTrees.
const (
	numericalTolerance     = 1e-6
	DefaultTreeHeight      = 4
	DefaultBranchingFactor = 16
	rootIndex              = 0
	// Fraction a node needs to contribute to the total count of itself and its siblings to be
	// considered during the search for a particular quantile. The idea of alpha is to filter out
	// noisy empty nodes. This is a post processing parameter with no privacy implications.
	alpha = 0.0075
)

// BoundedQuantiles calculates a differentially private quantiles of a collection
// of float64 values using a quantile tree mechanism.
// See https://github.com/google/differential-privacy/blob/main/common_docs/Differentially_Private_Quantile_Trees.pdf.
//
// It supports privacy units that contribute to multiple partitions (via the
// MaxPartitionsContributed parameter) as well as contribute to the same partition
// multiple times (via the MaxContributionsPerPartition parameter), by scaling the
// added noise appropriately.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for float64
// values. This aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type BoundedQuantiles struct {
	// Parameters
	epsilon         float64
	delta           float64
	lower           float64
	upper           float64
	treeHeight      int
	branchingFactor int
	l0Sensitivity   int64
	lInfSensitivity float64
	Noise           noise.Noise
	noiseKind       noise.Kind // necessary for serializing noise.Noise information

	// State variables
	tree              map[int]int64
	noisedTree        map[int]float64
	numLeaves         int
	leftmostLeafIndex int
	state             aggregationState
}

// BoundedQuantilesOptions contains the options necessary to initialize a BoundedQuantiles.
type BoundedQuantilesOptions struct {
	Epsilon                      float64 // Privacy parameter ε. Required.
	Delta                        float64 // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed     int64   // How many distinct partitions may a single privacy unit contribute to? Required.
	MaxContributionsPerPartition int64   // How many times may a single user contribute to a single partition? Required.
	// Lower and Upper bounds for clamping. Required; must be such that Lower < Upper.
	Lower, Upper float64
	Noise        noise.Noise // Type of noise used in BoundedSum. Defaults to Laplace noise.
	// It is not recommended to set TreeHeight and BranchingFactor since they require
	// implementation-specific insight to modify and they only apply to QuantileTree
	// algorithm, which might become obsolote if another algorithm is used.
	TreeHeight      int // Height of the QuantileTree. Defaults to defaultTreeHeight.
	BranchingFactor int // Number of children of every non-leaf node. Defaults to defaultBranchingFactor.
}

// NewBoundedQuantiles returns a new BoundedQuantiles.
func NewBoundedQuantiles(opt *BoundedQuantilesOptions) (*BoundedQuantiles, error) {
	if opt == nil {
		opt = &BoundedQuantilesOptions{} // Prevents panicking due to a nil pointer dereference.
	}

	maxContributionsPerPartition := opt.MaxContributionsPerPartition
	if err := checks.CheckMaxContributionsPerPartition(maxContributionsPerPartition); err != nil {
		return nil, fmt.Errorf("NewBoundedQuantiles: %w", err)
	}

	maxPartitionsContributed := opt.MaxPartitionsContributed
	if maxPartitionsContributed == 0 {
		return nil, fmt.Errorf("NewBoundedQuantiles: MaxPartitionsContributed must be set")
	}

	n := opt.Noise
	if n == nil {
		n = noise.Laplace()
	}
	// Check bounds.
	lower, upper := opt.Lower, opt.Upper
	if lower == 0 && upper == 0 {
		return nil, fmt.Errorf("NewBoundedQuantiles: Lower and Upper must be set (automatic bounds determination is not implemented yet). Lower and Upper cannot be both 0")
	}
	if err := checks.CheckBoundsFloat64(lower, upper); err != nil {
		return nil, fmt.Errorf("NewBoundedQuantiles: %w", err)
	}
	if err := checks.CheckBoundsNotEqual(lower, upper); err != nil {
		return nil, fmt.Errorf("NewBoundedQuantiles: %w", err)
	}

	// Check tree height and branching factor, set defaults if not specified, and use them to compute numLeaves and leftmostLeafIndex.
	treeHeight := opt.TreeHeight
	if treeHeight == 0 {
		treeHeight = DefaultTreeHeight
	}
	if err := checks.CheckTreeHeight(treeHeight); err != nil {
		return nil, fmt.Errorf("NewBoundedQuantiles: %v", err)
	}
	branchingFactor := opt.BranchingFactor
	if branchingFactor == 0 {
		branchingFactor = DefaultBranchingFactor
	}
	if err := checks.CheckBranchingFactor(branchingFactor); err != nil {
		return nil, fmt.Errorf("NewBoundedQuantiles: %v", err)
	}
	numNodes := getNumNodes(treeHeight, branchingFactor)
	numLeaves := getNumLeaves(treeHeight, branchingFactor)
	// The following assumes that nodes are indexed in a breadth first fashion from left to right.
	leftmostLeafIndex := numNodes - numLeaves
	eps, del := opt.Epsilon, opt.Delta

	// The l_1 sensitivty of a privacy unit's contribution is
	//    treeHeight * maxPartitionsContributed * maxContributionsPerPartition
	// while the l_2 sensitivty is
	//    sqrt(treeHeight * maxPartitionsContributed) * maxContributionsPerPartition
	// (the latter is realized if the privacy unit increments the exact same counters for each of
	// their contributions to a particular partition). Setting the l_0 and l_inf sensitivity as
	// follows yields the respective l_1 and l_2 values.
	l0Sensitivity := int64(treeHeight) * maxPartitionsContributed
	lInfSensitivity := float64(maxContributionsPerPartition)

	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some placeholder value.
	_, err := n.AddNoiseFloat64(0, l0Sensitivity, lInfSensitivity, eps, del)
	if err != nil {
		return nil, fmt.Errorf("NewBoundedQuantiles: %w", err)
	}

	return &BoundedQuantiles{
		epsilon:           eps,
		delta:             del,
		lower:             lower,
		upper:             upper,
		treeHeight:        treeHeight,
		branchingFactor:   branchingFactor,
		l0Sensitivity:     l0Sensitivity,
		lInfSensitivity:   lInfSensitivity,
		Noise:             n,
		noiseKind:         noise.ToKind(n),
		tree:              make(map[int]int64),
		noisedTree:        make(map[int]float64),
		numLeaves:         numLeaves,
		leftmostLeafIndex: leftmostLeafIndex,
		state:             defaultState,
	}, nil
}

// Add adds an entry to BoundedQuantiles. It skips (ignores) NaN values because their
// contribution to the final result is not well defined.
func (bq *BoundedQuantiles) Add(e float64) error {
	if bq.state != defaultState {
		return fmt.Errorf("BoundedQuantiles cannot be amended: %v", bq.state.errorMessage())
	}
	if !math.IsNaN(e) {
		// Increment all counts on the path from the leaf node where the value is inserted up to the
		// first level (root not included).
		clamped, err := ClampFloat64(e, bq.lower, bq.upper)
		if err != nil {
			return fmt.Errorf("couldn't clamp input value %f, err %w", e, err)
		}
		index := bq.getIndex(clamped)
		for index != rootIndex {
			count := bq.tree[index]
			bq.tree[index] = count + 1
			index = bq.getParent(index)
		}
	}
	return nil
}

// Result calculates and returns a differentially private quantile of the values added.
// The specified rank must be between 0.0 and 1.0.
//
// This function can be called multiple times to compute different quantiles. Privacy budget is
// paid only once, on its first invocation. Calling this method repeatedly for the same rank will
// return the same result. The results of repeated calls are guaranteed to be monotonically
// increasing in the sense that r_1 < r_2 implies that Result(r_1) <= Result(r_2).
//
// Note that the returned values is not an unbiased estimate of the raw bounded quantile.
func (bq *BoundedQuantiles) Result(rank float64) (float64, error) {
	if bq.state != defaultState && bq.state != resultReturned {
		return 0, fmt.Errorf("BoundedQuantiles' noised result cannot be computed: %v", bq.state.errorMessage())
	}
	bq.state = resultReturned

	if rank < 0.0 || rank > 1.0 {
		return 0, fmt.Errorf("rank %f must be >= 0 and <= 1", rank)
	}
	rank = adjustRank(rank)

	index := rootIndex
	// Search for the index of the leaf node containg the specified quantile, starting at the root.
	for index < bq.leftmostLeafIndex {
		leftmostChildIndex := bq.getLeftmostChild(index)
		rightmostChildIndex := bq.getRightmostChild(index)

		totalCount := 0.0
		for i := leftmostChildIndex; i <= rightmostChildIndex; i++ {
			noisedCount, err := bq.getNoisedCount(i)
			if err != nil {
				return 0, fmt.Errorf("couldn't get noised count for node %d: %w", i, err)
			}
			totalCount += math.Max(0.0, noisedCount)
		}

		correctedTotalCount := 0.0
		for i := leftmostChildIndex; i <= rightmostChildIndex; i++ {
			// Treat child nodes contributing less than an alpha fraction to the total count as empty
			// subtrees.
			noisedCount, err := bq.getNoisedCount(i)
			if err != nil {
				return 0, fmt.Errorf("couldn't get noised count for node %d: %w", i, err)
			}
			if noisedCount >= totalCount*alpha {
				correctedTotalCount += noisedCount
			}
		}
		if correctedTotalCount == 0.0 {
			// Either all counts are 0.0 or no child node contributes more than an alpha fraction to the
			// total count (the latter can only happen when alpha > 1 / branching factor, which is not
			// the case for the default branching factor). This means that all child nodes are
			// considered empty and there is no need to proceed further down the tree.
			break
		}

		// Determine the child node whose subtree contains the quantile.
		partialCount := 0.0
		for i := leftmostChildIndex; true; i++ {
			count, err := bq.getNoisedCount(i)
			if err != nil {
				return 0, fmt.Errorf("couldn't get noised count for node %d: %w", i, err)
			}
			// Skip child nodes contributing less than alpha to the total count.
			if count >= totalCount*alpha {
				partialCount += count
				// Check if the quantile is in the current child's subtree.
				if partialCount/correctedTotalCount >= rank-numericalTolerance {
					rank = (rank - (partialCount-count)/correctedTotalCount) / (count / correctedTotalCount)
					// Clamping rank to a value between 0.0 and 1.0. Note that rank can become greater than
					// 1 because of the numerical tolerance. Values less than 0.0 should not occur. The
					// respective clamping is set in place to be on the safe side.
					rank = math.Min(math.Max(0.0, rank), 1.0)
					index = i
					break
				}
			}
		}
	}
	// Linearly interpolate between the smallest and largest value associated with the node of the
	// current index.
	return (1-rank)*bq.getLeftValue(index) + rank*bq.getRightValue(index), nil
}

// getIndex returns the index of the leaf node associated with the provided value, assuming that
// the leaf nodes partition the range betwen lower and upper into intervals of equal size.
func (bq *BoundedQuantiles) getIndex(value float64) int {
	indexFromLeftmostLeaf := int((value - bq.lower) / (bq.upper - bq.lower) * float64(bq.numLeaves))
	if value == bq.upper {
		indexFromLeftmostLeaf = bq.numLeaves - 1
	}
	return bq.leftmostLeafIndex + indexFromLeftmostLeaf
}

// getLeftValue returns the smallest value mapped to the subtree of the provided index, assuming that
// the leaf nodes partition the range betwen lower and upper into intervals of equal size.
func (bq *BoundedQuantiles) getLeftValue(index int) float64 {
	// Traverse the tree towards the leaves starting at the provided index always taking the leftmost branch.
	for index < bq.leftmostLeafIndex {
		index = bq.getLeftmostChild(index)
	}
	return (bq.upper-bq.lower)*(float64((index-bq.leftmostLeafIndex))/float64(bq.numLeaves)) + bq.lower
}

// getRightValue returns the greatest value mapped to the subtree of the provided index, assuming that
// the leaf nodes partition the range betwen lower and upper into intervals of equal size.
func (bq *BoundedQuantiles) getRightValue(index int) float64 {
	// Traverse the tree towards the leaves starting at the provided index always taking the rightmost branch.
	for index < bq.leftmostLeafIndex {
		index = bq.getRightmostChild(index)
	}
	// The returned value bounds the range of values for which getIndex returns the specified index.
	// This bound is not itself contained in that range, i.e., getIndex will return the next index
	// when called for the bound.
	return (bq.upper-bq.lower)*(float64((index-bq.leftmostLeafIndex+1))/float64(bq.numLeaves)) + bq.lower
}

func (bq *BoundedQuantiles) getLeftmostChild(index int) int {
	return index*bq.branchingFactor + 1
}

func (bq *BoundedQuantiles) getRightmostChild(index int) int {
	return (index + 1) * bq.branchingFactor
}

func (bq *BoundedQuantiles) getParent(index int) int {
	return (index - 1) / bq.branchingFactor
}

func (bq *BoundedQuantiles) getNoisedCount(index int) (float64, error) {
	if noisedCount, ok := bq.noisedTree[index]; ok {
		return noisedCount, nil
	}
	rawCount := bq.tree[index]
	noisedCount, err := bq.Noise.AddNoiseFloat64(float64(rawCount), bq.l0Sensitivity, bq.lInfSensitivity, bq.epsilon, bq.delta)
	if err != nil {
		return 0, err
	}
	bq.noisedTree[index] = noisedCount
	return noisedCount, nil
}

// Clamps the rank to a value between 0.005 and 0.995. The purpose of this adjustment is to mitigate
// the inaccuracy of the quantile tree mechanism around the min and max, i.e., the 0 and 1 rank.
func adjustRank(rank float64) float64 {
	return math.Max(math.Min(rank, 0.995), 0.005)
}

func getNumNodes(treeHeight, branchingFactor int) int {
	return int((math.Pow(float64(branchingFactor), float64(treeHeight+1)) - 1.0)) / (branchingFactor - 1)
}

func getNumLeaves(treeHeight, branchingFactor int) int {
	return int(math.Pow(float64(branchingFactor), float64(treeHeight)))
}

// Merge merges bq2 into bq (i.e., adds to bq all entries that were added to
// bq2). bq2 is consumed by this operation: bq2 may not be used after it is
// merged into bq.
func (bq *BoundedQuantiles) Merge(bq2 *BoundedQuantiles) error {
	if err := checkMergeBoundedQuantiles(bq, bq2); err != nil {
		return err
	}

	for index, count := range bq2.tree {
		bq.tree[index] += count
	}
	bq2.state = merged
	return nil
}

func checkMergeBoundedQuantiles(bq1, bq2 *BoundedQuantiles) error {
	if bq1.state != defaultState {
		return fmt.Errorf("checkMergeBoundedQuantiles: bq1 cannot be merged with another BoundedQuantiles instance: %v", bq1.state.errorMessage())
	}
	if bq2.state != defaultState {
		return fmt.Errorf("checkMergeBoundedQuantiles: bq2 cannot be merged with another BoundedQuantiles instance: %v", bq2.state.errorMessage())
	}

	if !bqEquallyInitialized(bq1, bq2) {
		return fmt.Errorf("checkMergeBoundedQuantiles: bq1 and bq2 are not compatible")
	}

	return nil
}

func bqEquallyInitialized(bq1, bq2 *BoundedQuantiles) bool {
	return bq1.epsilon == bq2.epsilon &&
		bq1.delta == bq2.delta &&
		bq1.l0Sensitivity == bq2.l0Sensitivity &&
		bq1.lInfSensitivity == bq2.lInfSensitivity &&
		bq1.lower == bq2.lower &&
		bq1.upper == bq2.upper &&
		bq1.treeHeight == bq2.treeHeight &&
		bq1.branchingFactor == bq2.branchingFactor &&
		bq1.noiseKind == bq2.noiseKind &&
		bq1.state == bq2.state
}

// encodableBoundedQuantiles can be encoded by the gob package.
type encodableBoundedQuantiles struct {
	Epsilon           float64
	Delta             float64
	L0Sensitivity     int64
	LInfSensitivity   float64
	TreeHeight        int
	BranchingFactor   int
	Lower             float64
	Upper             float64
	NumLeaves         int
	LeftmostLeafIndex int
	NoiseKind         noise.Kind
	QuantileTree      map[int]int64
}

// GobEncode encodes BoundedQuantiles.
func (bq *BoundedQuantiles) GobEncode() ([]byte, error) {
	if bq.state != defaultState && bq.state != serialized {
		return nil, fmt.Errorf("BoundedQuantiles object cannot be serialized: " + bq.state.errorMessage())
	}
	enc := encodableBoundedQuantiles{
		Epsilon:           bq.epsilon,
		Delta:             bq.delta,
		L0Sensitivity:     bq.l0Sensitivity,
		LInfSensitivity:   bq.lInfSensitivity,
		TreeHeight:        bq.treeHeight,
		BranchingFactor:   bq.branchingFactor,
		Lower:             bq.lower,
		Upper:             bq.upper,
		NumLeaves:         bq.numLeaves,
		LeftmostLeafIndex: bq.leftmostLeafIndex,
		NoiseKind:         noise.ToKind(bq.Noise),
		QuantileTree:      bq.tree,
	}
	bq.state = serialized
	return encode(enc)
}

// GobDecode decodes BoundedQuantiles.
func (bq *BoundedQuantiles) GobDecode(data []byte) error {
	var enc encodableBoundedQuantiles
	err := decode(&enc, data)
	if err != nil {
		return fmt.Errorf("couldn't decode BoundedQuantiles from bytes")
	}
	*bq = BoundedQuantiles{
		epsilon:           enc.Epsilon,
		delta:             enc.Delta,
		l0Sensitivity:     enc.L0Sensitivity,
		lInfSensitivity:   enc.LInfSensitivity,
		treeHeight:        enc.TreeHeight,
		branchingFactor:   enc.BranchingFactor,
		lower:             enc.Lower,
		upper:             enc.Upper,
		noiseKind:         enc.NoiseKind,
		Noise:             noise.ToNoise(enc.NoiseKind),
		numLeaves:         enc.NumLeaves,
		leftmostLeafIndex: enc.LeftmostLeafIndex,
		tree:              enc.QuantileTree,
		noisedTree:        make(map[int]float64),
		state:             defaultState,
	}
	return nil
}
