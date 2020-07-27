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

package pbeam

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	testpb "github.com/google/differential-privacy/privacy-on-beam/testdata"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*testpb.TestAnon)(nil)))
	beam.RegisterType(reflect.TypeOf(pairII{}))
	beam.RegisterType(reflect.TypeOf(pairII64{}))
	beam.RegisterType(reflect.TypeOf(pairIF64{}))
	beam.RegisterType(reflect.TypeOf(pairICodedKV{}))
	beam.RegisterType(reflect.TypeOf(protoPair{}))

	beam.RegisterType(reflect.TypeOf((*diffInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*diffFloat64Fn)(nil)))

	beam.RegisterType(reflect.TypeOf((*checkSomePartitionsAreDroppedFn)(nil)))

	beam.RegisterFunction(checkNoNegativeValuesInt64Fn)
	beam.RegisterFunction(checkNoNegativeValuesFloat64Fn)
	beam.RegisterFunction(checkAllValuesNegativeFloat64Fn)

	beam.RegisterType(reflect.TypeOf((*checkFloat64MetricsAreNoisyFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*checkInt64MetricsAreNoisyFn)(nil)))
	beam.RegisterType(reflect.TypeOf(testInt64Metric{}))
	beam.RegisterType(reflect.TypeOf(testFloat64Metric{}))
}

func TestMain(m *testing.M) {
	ptest.Main(m)
}

// pairII, pairII64, pairIF64 and the related functions are helpers necessary to
// get a PCollection of KV type as input of a test Beam pipeline.
type pairII struct {
	A int
	B int
}

func pairToKV(p pairII) (a, b int) {
	return p.A, p.B
}

func kvToPair(a, b int) pairII {
	return pairII{a, b}
}

type pairII64 struct {
	A int
	B int64
}

func pairII64ToKV(p pairII64) (a int, b int64) {
	return p.A, p.B
}

type pairIF64 struct {
	A int
	B float64
}

func pairIFToKV(p pairIF64) (a int, b float64) {
	return p.A, p.B
}

type pairICodedKV struct {
	A int
	B kv.Pair
}

func kvToPairICodedKV(a int, b kv.Pair) pairICodedKV {
	return pairICodedKV{a, b}
}

func pairICodedKVToKV(p pairICodedKV) (k int, v kv.Pair) {
	return p.A, p.B
}

type protoPair struct {
	key string
	pb  *testpb.TestAnon
}

func kvToProtoPair(key string, pb *testpb.TestAnon) protoPair {
	return protoPair{key, pb}
}

// makePairsWithFixedV returns dummy data where the same value is associated to
// multiple privacy keys: it returns a slice of pairs {0, v}, {1, v}, ..., {numKeys-1, v}.
func makePairsWithFixedV(numKeys, v int) []pairII {
	s := make([]pairII, 0, numKeys)
	for k := 0; k < numKeys; k++ {
		s = append(s, pairII{k, v})
	}
	return s
}

// makePairsWithFixedVStartingFromKey returns dummy data where the same value is associated to
// multiple privacy keys: it returns a slice of pairs {0, v}, {1, v}, ..., {numKeys-1, v}.
// Privacy keys start from kOffset.
func makePairsWithFixedVStartingFromKey(kOffset, numKeys, v int) []pairII {
	s := make([]pairII, 0, numKeys)
	for k := 0; k < numKeys; k++ {
		s = append(s, pairII{k + kOffset, v})
	}
	return s
}

// tripleWithIntValue contains a privacy ID, a partition ID, and an int value.
type tripleWithIntValue struct {
	ID        int
	Partition int
	Value     int
}

// makeDummyTripleWithIntValue returns dummy int data where the same partition ID is
// associated to multiple privacy keys, every time with the value 1: it returns
// a slice of tripleInts {0,p,1}, {1,p,1}, ..., {numKeys-1,p,1}.
func makeDummyTripleWithIntValue(numKeys, p int) []tripleWithIntValue {
	return makeTripleWithIntValue(numKeys, p, 1)
}

// makeDummyTripleWithIntValueStartingFromKey returns dummy int data where the same partition ID is
// associated to multiple privacy keys, every time with the value 1: it returns
// a slice of tripleInts {0,p,1}, {1,p,1}, ..., {numKeys-1,p,1}.
// Privacy keys start from kOffset.
func makeDummyTripleWithIntValueStartingFromKey(kOffset, numKeys, p int) []tripleWithIntValue {
	return makeTripleWithIntValueStartingFromKey(kOffset, numKeys, p, 1)
}

// makeTripleWithIntValueStartingFromKey returns int data where the same partition ID is
// associated to multiple privacy keys (starting from provided key), to the given value v: it returns
// a slice of tripleInts {k,p,v}, {k + 1,p,v}, ..., {numKeys + k - 1,p,v}.
// Privacy keys start from kOffset.
func makeTripleWithIntValueStartingFromKey(kOffset, numKeys, p, v int) []tripleWithIntValue {
	s := make([]tripleWithIntValue, numKeys, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = tripleWithIntValue{k + kOffset, p, v}
	}
	return s
}

// makeTripleWithIntValue returns int data where the same partition ID is
// associated to multiple privacy keys, to the given value v: it returns
// a slice of tripleInts {0,p,v}, {1,p,v}, ..., {numKeys-1,p,v}.
func makeTripleWithIntValue(numKeys, p, v int) []tripleWithIntValue {
	s := make([]tripleWithIntValue, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = tripleWithIntValue{k, p, v}
	}
	return s
}

// tripleWithIntValueToKV extracts the partition ID and the value from a tripleWithIntValue. It is
// used once the PrivatePCollection has been initialized, to transform it into a
// PrivatePCollection<partitionID,value>.
func tripleWithIntValueToKV(t tripleWithIntValue) (int, int) {
	return t.Partition, t.Value
}

// extractIDFromTripleWithIntValue extracts and returns the ID from a tripleWithIntValue. It is used to
// initialize PrivatePCollections.
func extractIDFromTripleWithIntValue(t tripleWithIntValue) (int, tripleWithIntValue) {
	return t.ID, t
}

// concatenateTriplesWithIntValue concatenates tripleWithIntValue slices.
func concatenateTriplesWithIntValue(slices ...[]tripleWithIntValue) []tripleWithIntValue {
	var t []tripleWithIntValue
	for _, slice := range slices {
		t = append(t, slice...)
	}
	return t
}

// tripleWithFloatValue contains a privacy ID, a partition ID, and a float value.
type tripleWithFloatValue struct {
	ID        int
	Partition int
	Value     float32
}

// makeDummyTripleWithFloatValue returns dummy float data where the same partition ID is
// associated to multiple privacy keys, every time with the value 1.0: it returns
// a slice of tripleFloats {0,p,1}, {1,p,1}, ..., {numKeys-1,p,1}.
func makeDummyTripleWithFloatValue(numKeys, p int) []tripleWithFloatValue {
	return makeTripleWithFloatValue(numKeys, p, 1.0)
}

// makeDummyTripleWithFloatValueStartingFromKey returns dummy float data where the same partition ID is
// associated to multiple privacy keys, every time with the value 1.0: it returns
// a slice of tripleFloats {0,p,1}, {1,p,1}, ..., {numKeys-1,p,1}.
// Privacy keys start from kOffset.
func makeDummyTripleWithFloatValueStartingFromKey(kOffset, numKeys, p int) []tripleWithFloatValue {
	return makeTripleWithFloatValueStartingFromKey(kOffset, numKeys, p, 1.0)
}

// makeTripleWithIntValue returns float data where the same partition ID is
// associated to multiple privacy keys, to the given value v: it returns
// a slice of tripleInts {0,p,v}, {1,p,v}, ..., {numKeys-1,p,v}.
func makeTripleWithFloatValue(numKeys, p int, v float32) []tripleWithFloatValue {
	s := make([]tripleWithFloatValue, numKeys, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = tripleWithFloatValue{k, p, v}
	}
	return s
}

// makeTripleWithFloatValueStartingFromKey returns float data where the same partition ID is
// associated to multiple privacy keys (starting from provided key), to the given value v: it returns
// a slice of tripleFloats {k,p,v}, {k + 1,p,v}, ..., {numKeys + k - 1,p,v}.
// Privacy keys start from kOffset.
func makeTripleWithFloatValueStartingFromKey(kOffset, numKeys, p int, v float32) []tripleWithFloatValue {
	s := make([]tripleWithFloatValue, numKeys, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = tripleWithFloatValue{k + kOffset, p, v}
	}
	return s
}

// concatenateTriplesWithFloatValue concatenates tripleWithFloatValue slices.
func concatenateTriplesWithFloatValue(slices ...[]tripleWithFloatValue) []tripleWithFloatValue {
	var t []tripleWithFloatValue
	for _, slice := range slices {
		t = append(t, slice...)
	}
	return t
}

// extractIDFromTripleWithFloatValue extracts and returns the ID from a tripleWithFloatValue. It is used to
// initialize PrivatePCollections.
func extractIDFromTripleWithFloatValue(t tripleWithFloatValue) (int, tripleWithFloatValue) {
	return t.ID, t
}

// tripleWithFloatValueToKV extracts the partition ID and the value from a tripleWithFloatValue. It is
// used once the PrivatePCollection has been initialized, to transform it into a
// PrivatePCollection<partitionID,value>.
func tripleWithFloatValueToKV(t tripleWithFloatValue) (int, float32) {
	return t.Partition, t.Value
}

// concatenatePairs concatenates pairII slices.
func concatenatePairs(slices ...[]pairII) []pairII {
	var s []pairII
	for _, slice := range slices {
		s = append(s, slice...)
	}
	return s
}

// testInt64Metric, testFloat64Metric and associated functions are used to test DP aggregations.
type testInt64Metric struct {
	Value  int
	Metric int64
}

func kvToInt64Metric(v int, m int64) testInt64Metric {
	return testInt64Metric{v, m}
}

func int64MetricToKV(tm testInt64Metric) (int, int64) {
	return tm.Value, tm.Metric
}

type testFloat64Metric struct {
	Value  int
	Metric float64
}

func kvToFloat64Metric(v int, m float64) testFloat64Metric {
	return testFloat64Metric{v, m}
}

func float64MetricToKV(tm testFloat64Metric) (int, float64) {
	return tm.Value, tm.Metric
}

func float64MetricToInt64Metric(tm testFloat64Metric) testInt64Metric {
	return testInt64Metric{tm.Value, int64(tm.Metric)}
}

// approxEqualsKVInt64 checks that two PCollections col1 and col2 of type
// <K,int64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and the value associated to
// key k in col1 is within the specified tolerance of the value associated to k
// in col2". Each key can only hold a single value.
func approxEqualsKVInt64(s beam.Scope, col1, col2 beam.PCollection, tolerance float64) error {
	wantV := reflect.TypeOf(int64(0))
	if err := checkValueType(col1, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffInt64Fn{Tolerance: tolerance}, coGroupToValue)
	combinedDiff := beam.Combine(s, combineDiffs, diffs)
	beam.ParDo0(s, reportDiffs, combinedDiff)
	return nil
}

// approxEqualsKVFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and the value associated to
// key k in col1 is within the specified tolerance of the value associated to k
// in col2". Each key can only hold a single value.
func approxEqualsKVFloat64(s beam.Scope, col1, col2 beam.PCollection, tolerance float64) error {
	wantV := reflect.TypeOf(float64(0))
	if err := checkValueType(col1, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffFloat64Fn{Tolerance: tolerance}, coGroupToValue)
	combinedDiff := beam.Combine(s, combineDiffs, diffs)
	beam.ParDo0(s, reportDiffs, combinedDiff)
	return nil
}

func reportDiffs(diffs string) error {
	if diffs != "" {
		return fmt.Errorf("collections are not approximately equal. Diff:\n%s", diffs)
	}
	return nil
}

func combineDiffs(diff1, diff2 string) string {
	if diff2 == "" {
		return fmt.Sprintf("%s", diff1)
	}
	return fmt.Sprintf("%s\n%s", diff1, diff2)
}

type diffInt64Fn struct {
	Tolerance float64
}

// ProcessElement returns a diff between values associated with a key. It
// returns an empty string if the values are approximately equal.
func (fn *diffInt64Fn) ProcessElement(k int, v1Iter, v2Iter func(*int64) bool) string {
	var v1 = toSliceInt64(v1Iter)
	var v2 = toSliceInt64(v2Iter)
	if diff := cmp.Diff(v1, v2, cmpopts.EquateApprox(0, fn.Tolerance)); diff != "" {
		return fmt.Sprintf("For k=%d: diff=%s", k, diff)
	}
	return ""
}

func toSliceInt64(vIter func(*int64) bool) []float64 {
	var vSlice []float64
	var v int64
	for vIter(&v) {
		vSlice = append(vSlice, float64(v))
	}
	return vSlice
}

type diffFloat64Fn struct {
	Tolerance float64
}

func (fn *diffFloat64Fn) ProcessElement(k int, v1Iter, v2Iter func(*float64) bool) string {
	var v1 = toSliceFloat64(v1Iter)
	var v2 = toSliceFloat64(v2Iter)

	if diff := cmp.Diff(v1, v2, cmpopts.EquateApprox(0, fn.Tolerance)); diff != "" {
		return fmt.Sprintf("For k=%d: diff=%s", k, diff)
	}

	return "" // No diff
}

func toSliceFloat64(vIter func(*float64) bool) []float64 {
	var vSlice []float64
	var v float64
	for vIter(&v) {
		vSlice = append(vSlice, v)
	}
	return vSlice
}

func checkValueType(col beam.PCollection, wantValueType reflect.Type) error {
	_, vFullType := beam.ValidateKVType(col)
	vType := vFullType.Type()
	if vType != wantValueType {
		return fmt.Errorf("PCollection has (K,V) type with V=%v, want %v", vType, wantValueType)
	}
	return nil
}

// laplaceTolerance returns tolerance to be used in approxEquals for tests
// to pass with 10⁻ᵏ flakiness. flakinessK is the parameter used to specify this.
//
// l1Sensitivity and epsilon are the DP parameters of the test.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/master/privacy-on-beam/docs/Tolerance_Calculation.pdf.
func laplaceTolerance(flakinessK, l1Sensitivity, epsilon float64) float64 {
	return l1Sensitivity * flakinessK * math.Log(10) / epsilon
}

// oneSidedLaplaceTolerance is only supposed be to used in cases where a one sided
// confidence internal is enough to calculate the tolerance, for example for
// finding minimum and maximum noisy counts and sums when calculating the tolerance
// for mean.
func oneSidedLaplaceTolerance(flakinessK, l1Sensitivity, epsilon float64) float64 {
	return l1Sensitivity * (flakinessK*math.Log(10) - math.Log(2)) / epsilon
}

// roundedLaplaceTolerance rounds laplace tolerance value up to the nearest
// integer, in order to work with both integer and float aggregation tests and
// be on the safe side.
func roundedLaplaceTolerance(flakinessK, l1Sensitivity, epsilon float64) float64 {
	return math.Ceil(laplaceTolerance(flakinessK, l1Sensitivity, epsilon))
}

// TODO: Revisit the math in the formulas below or change the solution to the one based on the confidence intervals.
//
// laplaceToleranceForMean returns tolerance to be used in approxEquals for tests
// for mean to pass with 10⁻ᵏ flakiness.
//
// flakinessK is the parameter used to specify k in the flakiness.
// distanceFromMidPoint = upper - midPoint, where midPoint = (lower + upper)/2.
// exactNormalizedSum is a clamped (with boundaries -distanceFromMidPoint and distanceFromMidPoint) sum of distances of the input entities from the midPoint.
// exactNormalizedSum is needed for calculating tolerance because the algorithm of the mean aggregation uses noisy normalized sum in its calculations.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// see https://github.com/google/differential-privacy/blob/master/privacy-on-beam/docs/Tolerance_Calculation.pdf.
func laplaceToleranceForMean(flakinessK, lower, upper float64, maxContributionsPerPartition, maxPartitionsContributed int64, epsilon float64, exactNormalizedSum, exactCount, exactMean float64) (float64, error) {
	halfFlakiness := flakinessK / 2
	halfEpsilon := epsilon / 2
	midPoint := lower + (upper-lower)/2.0

	countTolerance := math.Ceil(oneSidedLaplaceTolerance(halfFlakiness, l1SensitivityForCount(maxContributionsPerPartition, maxPartitionsContributed), halfEpsilon))
	normalizedSumTolerance := oneSidedLaplaceTolerance(halfFlakiness, l1SensitivityForNormalizedSum(lower, upper, maxContributionsPerPartition, maxPartitionsContributed), halfEpsilon)

	minNoisyCount := math.Max(1.0, exactCount-countTolerance)
	maxNoisyCount := math.Max(1.0, exactCount+countTolerance)
	minNoisyNormalizedSum := exactNormalizedSum - normalizedSumTolerance
	maxNoisyNormalizedSum := exactNormalizedSum + normalizedSumTolerance
	maxNoisyMean, err := dpagg.ClampFloat64(maxNoisyNormalizedSum/minNoisyCount+midPoint, lower, upper)
	if err != nil {
		return 0, err
	}
	minNoisyMean, err := dpagg.ClampFloat64(minNoisyNormalizedSum/maxNoisyCount+midPoint, lower, upper)
	if err != nil {
		return 0, err
	}
	distFromMinNoisyMean := distanceBetween(exactMean, minNoisyMean)
	distFromMaxNoisyMean := distanceBetween(maxNoisyMean, exactMean)

	return math.Max(distFromMaxNoisyMean, distFromMinNoisyMean), nil
}

func distanceBetween(a, b float64) float64 {
	return math.Abs(a - b)
}

func l1SensitivityForNormalizedSum(lower, upper float64, maxContributionsPerPartition, maxPartitionsContributed int64) float64 {
	midPoint := lower + (upper-lower)/2.0
	maxDistFromMidpoint := upper - midPoint
	return maxDistFromMidpoint * float64(maxContributionsPerPartition) * float64(maxPartitionsContributed)
}

func l1SensitivityForCount(maxContributionsPerPartition, maxPartitionsContributed int64) float64 {
	return float64(maxContributionsPerPartition) * float64(maxPartitionsContributed)
}

func int64Ptr(i int64) *int64 {
	return &i
}

func float64Ptr(f float64) *float64 {
	return &f
}

// checkFloat64MetricsAreNoisy checks that no values in a PCollection<testFloat64Metric>
// (where testFloat64Metric contains the aggregate statistic) is equal to exactMetric.
func checkFloat64MetricsAreNoisy(s beam.Scope, col beam.PCollection, exactMetric, tolerance float64) {
	beam.ParDo0(s, &checkFloat64MetricsAreNoisyFn{exactMetric, tolerance}, col)
}

type checkFloat64MetricsAreNoisyFn struct {
	ExactMetric float64
	Tolerance   float64
}

func (fn *checkFloat64MetricsAreNoisyFn) ProcessElement(m testFloat64Metric) error {
	if cmp.Equal(m.Metric, fn.ExactMetric, cmpopts.EquateApprox(0, fn.Tolerance)) {
		return fmt.Errorf("found a non-noisy output of %f for value %d", fn.ExactMetric, m.Value)
	}
	return nil
}

// checkInt64MetricsAreNoisy checks that no values in a PCollection<testInt64Metric>
// (where testInt64Metric contains the aggregate statistic) is equal to exactMetric.
func checkInt64MetricsAreNoisy(s beam.Scope, col beam.PCollection, exactMetric int) {
	beam.ParDo0(s, &checkInt64MetricsAreNoisyFn{exactMetric}, col)
}

type checkInt64MetricsAreNoisyFn struct {
	ExactMetric int
}

func (fn *checkInt64MetricsAreNoisyFn) ProcessElement(m testInt64Metric) error {
	if m.Metric == int64(fn.ExactMetric) {
		return fmt.Errorf("found a non-noisy output of %d for value %d", fn.ExactMetric, m.Value)
	}
	return nil
}

func oneFn(beam.V) int { return 1 }

// checkSomePartitionsAreDropped checks that the number of values in the PCollection
// is smaller than numPartitions, but larger than 0.
func checkSomePartitionsAreDropped(s beam.Scope, col beam.PCollection, numPartitions int) {
	ones := beam.ParDo(s, oneFn, col)
	sum := stats.Sum(s, ones)
	beam.ParDo0(s, &checkSomePartitionsAreDroppedFn{numPartitions}, sum)
}

type checkSomePartitionsAreDroppedFn struct {
	NumPartitions int
}

func (fn *checkSomePartitionsAreDroppedFn) ProcessElement(i int) error {
	if i <= 0 {
		return fmt.Errorf("got %d emitted partitions, want a positive number", i)
	}
	if i >= fn.NumPartitions {
		return fmt.Errorf("got %d emitted partitions (all of them), want some partitions to be dropped", i)
	}
	return nil
}

func checkNoNegativeValuesInt64Fn(v int64) error {
	if v < 0 {
		return fmt.Errorf("unexpected negative element: %v", v)
	}
	return nil
}

func checkNoNegativeValuesFloat64Fn(v float64) error {
	if v < 0 {
		return fmt.Errorf("unexpected negative element: %v", v)
	}
	return nil
}

func checkAllValuesNegativeFloat64Fn(v float64) error {
	if v >= 0 {
		return fmt.Errorf("unexpected non-negative element: %v", v)
	}
	return nil
}
