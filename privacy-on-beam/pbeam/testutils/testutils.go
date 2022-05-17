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

// Package testutils provides helper functions, structs, etc. for testing
// Privacy on Beam pipelines.
package testutils

import (
	"fmt"
	"math"
	"math/big"
	"reflect"

	"github.com/google/differential-privacy/go/v2/dpagg"
	"github.com/google/differential-privacy/go/v2/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v2/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func init() {
	beam.RegisterType(reflect.TypeOf(PairII{}))
	beam.RegisterType(reflect.TypeOf(PairII64{}))
	beam.RegisterType(reflect.TypeOf(PairIF64{}))
	beam.RegisterType(reflect.TypeOf(PairICodedKV{}))
	beam.RegisterType(reflect.TypeOf(TestInt64Metric{}))
	beam.RegisterType(reflect.TypeOf(TestFloat64Metric{}))
	beam.RegisterType(reflect.TypeOf(TestFloat64SliceMetric{}))

	beam.RegisterType(reflect.TypeOf((*diffInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*diffFloat64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*diffFloat64SliceFn)(nil)))

	beam.RegisterFunction(CheckNoNegativeValuesInt64Fn)
	beam.RegisterFunction(CheckNoNegativeValuesFloat64Fn)
	beam.RegisterFunction(CheckAllValuesNegativeFloat64Fn)

	beam.RegisterType(reflect.TypeOf((*checkFloat64MetricsAreNoisyFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*checkInt64MetricsAreNoisyFn)(nil)))

	beam.RegisterType(reflect.TypeOf((*checkSomePartitionsAreDroppedFn)(nil)))
	beam.RegisterFunction(gotExpectedNumPartitionsFn)
}

// PairII, pairII64, pairIF64, PairICodedKV and the related functions are helpers
// necessary to get a PCollection of KV type as input of a test Beam pipeline.

// PairII holds a key-value pair of type (int, int).
type PairII struct {
	A int
	B int
}

// PairToKV transforms a PairII into an (int, int) key-value pair.
func PairToKV(p PairII) (a, b int) {
	return p.A, p.B
}

// KVToPair transforms an (int, int) key-value pair into a PairII.
func KVToPair(a, b int) PairII {
	return PairII{a, b}
}

// PairII64 holds a key-value pair of type (int, int64).
type PairII64 struct {
	A int
	B int64
}

// PairII64ToKV transforms a PairII64 into an (int, int64) key-value pair.
func PairII64ToKV(p PairII64) (a int, b int64) {
	return p.A, p.B
}

// PairIF64 holds a key-value pair of type (int, float64).
type PairIF64 struct {
	A int
	B float64
}

// PairIFToKV transforms a PairIF64 into an (int, float64) key-value pair.
func PairIFToKV(p PairIF64) (a int, b float64) {
	return p.A, p.B
}

// PairICodedKV holds a key-value pair of type (int, kv.Pair).
type PairICodedKV struct {
	A int
	B kv.Pair
}

// PairICodedKVToKV transforms a PairICodedKV into an (int, kv.Pair) key-value pair.
func PairICodedKVToKV(p PairICodedKV) (k int, v kv.Pair) {
	return p.A, p.B
}

// KVToPairICodedKV transforms an (int, kv.Pair) key-value pair into a PairICodedKV.
func KVToPairICodedKV(a int, b kv.Pair) PairICodedKV {
	return PairICodedKV{a, b}
}

// MakePairsWithFixedV returns sample data where the same value is associated with
// multiple privacy keys: it returns a slice of pairs {0, v}, {1, v}, ..., {numKeys-1, v}.
func MakePairsWithFixedV(numKeys, v int) []PairII {
	s := make([]PairII, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = PairII{k, v}
	}
	return s
}

// MakePairsWithFixedVStartingFromKey returns sample data where the same value is associated with
// multiple privacy keys: it returns a slice of pairs {0, v}, {1, v}, ..., {numKeys-1, v}.
// Privacy keys start from kOffset.
func MakePairsWithFixedVStartingFromKey(kOffset, numKeys, v int) []PairII {
	s := make([]PairII, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = PairII{k + kOffset, v}
	}
	return s
}

// TripleWithIntValue contains a privacy ID, a partition ID, and an int value.
type TripleWithIntValue struct {
	ID        int
	Partition int
	Value     int
}

// MakeSampleTripleWithIntValue returns sample int data where the same partition ID is
// associated with multiple privacy keys, every time with the value 1: it returns
// a slice of tripleInts {0,p,1}, {1,p,1}, ..., {numKeys-1,p,1}.
func MakeSampleTripleWithIntValue(numKeys, p int) []TripleWithIntValue {
	return MakeTripleWithIntValue(numKeys, p, 1)
}

// MakeTripleWithIntValueStartingFromKey returns int data where the same partition ID is
// associated with multiple privacy keys (starting from provided key), to the given value v: it returns
// a slice of tripleInts {kOffset,p,v}, {kOffset + 1,p,v}, ..., {numKeys + kOffset - 1,p,v}.
// Privacy keys start from kOffset.
func MakeTripleWithIntValueStartingFromKey(kOffset, numKeys, p, v int) []TripleWithIntValue {
	s := make([]TripleWithIntValue, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = TripleWithIntValue{k + kOffset, p, v}
	}
	return s
}

// MakeTripleWithIntValue returns int data where the same partition ID is
// associated with multiple privacy keys, to the given value v: it returns
// a slice of tripleInts {0,p,v}, {1,p,v}, ..., {numKeys-1,p,v}.
func MakeTripleWithIntValue(numKeys, p, v int) []TripleWithIntValue {
	s := make([]TripleWithIntValue, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = TripleWithIntValue{k, p, v}
	}
	return s
}

// TripleWithIntValueToKV extracts the partition ID and the value from a tripleWithIntValue. It is
// used once the PrivatePCollection has been initialized, to transform it into a
// PrivatePCollection<partitionID,value>.
func TripleWithIntValueToKV(t TripleWithIntValue) (int, int) {
	return t.Partition, t.Value
}

// ExtractIDFromTripleWithIntValue extracts and returns the ID from a tripleWithIntValue. It is used to
// initialize PrivatePCollections.
func ExtractIDFromTripleWithIntValue(t TripleWithIntValue) (int, TripleWithIntValue) {
	return t.ID, t
}

// ConcatenateTriplesWithIntValue concatenates tripleWithIntValue slices.
func ConcatenateTriplesWithIntValue(slices ...[]TripleWithIntValue) []TripleWithIntValue {
	var t []TripleWithIntValue
	for _, slice := range slices {
		t = append(t, slice...)
	}
	return t
}

// TripleWithFloatValue contains a privacy ID, a partition ID, and a float value.
type TripleWithFloatValue struct {
	ID        int
	Partition int
	Value     float32
}

// MakeSampleTripleWithFloatValue returns sample float data where the same partition ID is
// associated with multiple privacy keys, every time with the value 1.0: it returns
// a slice of tripleFloats {0,p,1}, {1,p,1}, ..., {numKeys-1,p,1}.
func MakeSampleTripleWithFloatValue(numKeys, p int) []TripleWithFloatValue {
	return MakeTripleWithFloatValue(numKeys, p, 1.0)
}

// MakeTripleWithFloatValue returns float data where the same partition ID is
// associated with multiple privacy keys, to the given value v: it returns
// a slice of tripleInts {0,p,v}, {1,p,v}, ..., {numKeys-1,p,v}.
func MakeTripleWithFloatValue(numKeys, p int, v float32) []TripleWithFloatValue {
	s := make([]TripleWithFloatValue, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = TripleWithFloatValue{k, p, v}
	}
	return s
}

// MakeTripleWithFloatValueStartingFromKey returns float data where the same partition ID is
// associated with multiple privacy keys (starting from provided key), to the given value v: it returns
// a slice of tripleFloats {kOffset,p,v}, {kOffset + 1,p,v}, ..., {numKeys + kOffset - 1,p,v}.
// Privacy keys start from kOffset.
func MakeTripleWithFloatValueStartingFromKey(kOffset, numKeys, p int, v float32) []TripleWithFloatValue {
	s := make([]TripleWithFloatValue, numKeys)
	for k := 0; k < numKeys; k++ {
		s[k] = TripleWithFloatValue{k + kOffset, p, v}
	}
	return s
}

// ConcatenateTriplesWithFloatValue concatenates tripleWithFloatValue slices.
func ConcatenateTriplesWithFloatValue(slices ...[]TripleWithFloatValue) []TripleWithFloatValue {
	var t []TripleWithFloatValue
	for _, slice := range slices {
		t = append(t, slice...)
	}
	return t
}

// ExtractIDFromTripleWithFloatValue extracts and returns the ID from a tripleWithFloatValue. It is used to
// initialize PrivatePCollections.
func ExtractIDFromTripleWithFloatValue(t TripleWithFloatValue) (int, TripleWithFloatValue) {
	return t.ID, t
}

// TripleWithFloatValueToKV extracts the partition ID and the value from a tripleWithFloatValue. It is
// used once the PrivatePCollection has been initialized, to transform it into a
// PrivatePCollection<partitionID,value>.
func TripleWithFloatValueToKV(t TripleWithFloatValue) (int, float32) {
	return t.Partition, t.Value
}

// ConcatenatePairs concatenates pairII slices.
func ConcatenatePairs(slices ...[]PairII) []PairII {
	var s []PairII
	for _, slice := range slices {
		s = append(s, slice...)
	}
	return s
}

// TestInt64Metric holds a Value and an associated int64 metric (aggregation).
type TestInt64Metric struct {
	Value  int
	Metric int64
}

// KVToInt64Metric transforms an (int, int64) key-value pair into a TestInt64Metric.
func KVToInt64Metric(v int, m int64) TestInt64Metric {
	return TestInt64Metric{v, m}
}

// Int64MetricToKV transforms a TestInt64Metric into an (int, int64) key-value pair.
func Int64MetricToKV(tm TestInt64Metric) (int, int64) {
	return tm.Value, tm.Metric
}

// TestFloat64Metric holds a Value and an associated float64 metric (aggregation).
type TestFloat64Metric struct {
	Value  int
	Metric float64
}

// KVToFloat64Metric transforms an (int, float64) key-value pair into a TestFloat64Metric.
func KVToFloat64Metric(v int, m float64) TestFloat64Metric {
	return TestFloat64Metric{v, m}
}

// Float64MetricToKV transforms a TestFloat64Metric into an (int, float64) key-value pair.
func Float64MetricToKV(tm TestFloat64Metric) (int, float64) {
	return tm.Value, tm.Metric
}

// Float64MetricToInt64Metric transforms a TestFloat64Metric into a TestInt64Metric.
func Float64MetricToInt64Metric(tm TestFloat64Metric) TestInt64Metric {
	return TestInt64Metric{tm.Value, int64(tm.Metric)}
}

// TestFloat64SliceMetric holds a Value and an associated []float64 metric (aggregation).
type TestFloat64SliceMetric struct {
	Value  int
	Metric []float64
}

// Float64SliceMetricToKV transforms a TestFloat64SliceMetric into an (int, []float64) key-value pair.
func Float64SliceMetricToKV(tm TestFloat64SliceMetric) (int, []float64) {
	return tm.Value, tm.Metric
}

// KVToFloat64SliceMetric transforms an (int, []float64) key-value pair into a TestFloat64SliceMetric.
func KVToFloat64SliceMetric(v int, m []float64) TestFloat64SliceMetric {
	return TestFloat64SliceMetric{v, m}
}

// EqualsKVInt checks that two PCollections col1 and col2 of type
// <K,int> are exactly equal.
func EqualsKVInt(s beam.Scope, col1, col2 beam.PCollection) error {
	wantV := reflect.TypeOf(int(0))
	if err := checkValueType(col1, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, diffIntFn, coGroupToValue)
	combinedDiff := beam.Combine(s, combineDiffs, diffs)
	beam.ParDo0(s, reportDiffs, combinedDiff)
	return nil
}

// EqualsKVInt64 checks that two PCollections col1 and col2 of type
// <K,int64> are exactly equal. Each key can only hold a single value.
func EqualsKVInt64(s beam.Scope, col1, col2 beam.PCollection) error {
	return ApproxEqualsKVInt64(s, col1, col2, 0.0)
}

// EqualsKVFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are exactly equal. Each key can only hold a single value.
func EqualsKVFloat64(s beam.Scope, col1, col2 beam.PCollection) error {
	return ApproxEqualsKVFloat64(s, col1, col2, 0.0)
}

// NotEqualsFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are different. Each key can only hold a single value.
func NotEqualsFloat64(s beam.Scope, col1, col2 beam.PCollection) error {
	wantV := reflect.TypeOf(float64(0))
	if err := checkValueType(col1, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffFloat64Fn{Tolerance: 0.0}, coGroupToValue)
	combinedDiff := beam.Combine(s, combineDiffs, diffs)
	beam.ParDo0(s, reportEquals, combinedDiff)
	return nil
}

// ApproxEqualsKVInt64 checks that two PCollections col1 and col2 of type
// <K,int64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and the value associated with
// key k in col1 is within the specified tolerance of the value associated with k
// in col2". Each key can only hold a single value.
func ApproxEqualsKVInt64(s beam.Scope, col1, col2 beam.PCollection, tolerance float64) error {
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

// ApproxEqualsKVFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and the value associated with
// key k in col1 is within the specified tolerance of the value associated with k
// in col2". Each key can only hold a single value.
func ApproxEqualsKVFloat64(s beam.Scope, col1, col2 beam.PCollection, tolerance float64) error {
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

// LessThanOrEqualToKVFloat64 checks that for PCollections col1 and col2 of type
// <K,float64>, for each key k, value corresponding to col1 is less than or equal
// to the value corresponding in col2. Each key can only hold a single value.
func LessThanOrEqualToKVFloat64(s beam.Scope, col1, col2 beam.PCollection) error {
	wantV := reflect.TypeOf(float64(0))
	if err := checkValueType(col1, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, lessThanOrEqualTo, coGroupToValue)
	combinedDiff := beam.Combine(s, combineDiffs, diffs)
	beam.ParDo0(s, reportGreaterThan, combinedDiff)
	return nil
}

// ApproxEqualsKVFloat64Slice checks that two PCollections col1 and col2 of type
// <K,[]float64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and each value in the slice
// associated with key k in col1 is within the specified tolerance of each value
// in the slice associated with k in col2". Each key can only hold a single slice.
func ApproxEqualsKVFloat64Slice(s beam.Scope, col1, col2 beam.PCollection, tolerance float64) error {
	wantV := reflect.TypeOf([]float64{0.0})
	if err := checkValueType(col1, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		return fmt.Errorf("unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffFloat64SliceFn{Tolerance: tolerance}, coGroupToValue)
	combinedDiff := beam.Combine(s, combineDiffs, diffs)
	beam.ParDo0(s, reportDiffs, combinedDiff)
	return nil
}

func reportEquals(diffs string) error {
	if diffs != "" {
		return nil
	}
	return fmt.Errorf("collections are equal")
}

func reportDiffs(diffs string) error {
	if diffs != "" {
		return fmt.Errorf("collections are not approximately equal. Diff (-got, +want):\n%s", diffs)
	}
	return nil
}

func reportGreaterThan(errors string) error {
	if errors != "" {
		return fmt.Errorf("col1 is not less than or equal to col2: %s", errors)
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
	var v1 = int64PtrToSlice(v1Iter)
	var v2 = int64PtrToSlice(v2Iter)
	if diff := cmp.Diff(v1, v2, cmpopts.EquateApprox(0, fn.Tolerance)); diff != "" {
		return fmt.Sprintf("For k=%d: diff=%s", k, diff)
	}
	return ""
}

// ProcessElement returns a diff between values associated with a key. It
// returns an empty string if the values are approximately equal.
func diffIntFn(k beam.X, v1Iter, v2Iter func(*int) bool) string {
	var v1 = intPtrToSlice(v1Iter)
	var v2 = intPtrToSlice(v2Iter)
	if diff := cmp.Diff(v1, v2); diff != "" {
		return fmt.Sprintf("For k=%d: diff=%s", k, diff)
	}
	return ""
}

func int64PtrToSlice(vIter func(*int64) bool) []float64 {
	var vSlice []float64
	var v int64
	for vIter(&v) {
		vSlice = append(vSlice, float64(v))
	}
	return vSlice
}

func intPtrToSlice(vIter func(*int) bool) []float64 {
	var vSlice []float64
	var v int
	for vIter(&v) {
		vSlice = append(vSlice, float64(v))
	}
	return vSlice
}

func lessThanOrEqualTo(k int, v1Iter, v2Iter func(*float64) bool) string {
	var v1 = float64PtrToSlice(v1Iter)
	var v2 = float64PtrToSlice(v2Iter)
	if len(v1) != 1 {
		return fmt.Sprintf("For k=%d, col1 has %d values, it needs to have exactly 1 value", k, len(v1))
	}
	if len(v2) != 1 {
		return fmt.Sprintf("For k=%d, col2 has %d values, it needs to have exactly 1 value", k, len(v2))
	}
	if v1[0] > v2[0] {
		return fmt.Sprintf("For k=%d, v1=%f is greater than v2=%f", k, v1[0], v2[0])
	}

	return ""
}

type diffFloat64Fn struct {
	Tolerance float64
}

func (fn *diffFloat64Fn) ProcessElement(k int, v1Iter, v2Iter func(*float64) bool) string {
	var v1 = float64PtrToSlice(v1Iter)
	var v2 = float64PtrToSlice(v2Iter)
	if diff := cmp.Diff(v1, v2, cmpopts.EquateApprox(0, fn.Tolerance)); diff != "" {
		return fmt.Sprintf("For k=%d: diff=%s", k, diff)
	}

	return "" // No diff
}

type diffFloat64SliceFn struct {
	Tolerance float64
}

func (fn *diffFloat64SliceFn) ProcessElement(k int, v1Iter, v2Iter func(*[]float64) bool) string {
	var v1 = float64SlicePtrToSlice(v1Iter)
	var v2 = float64SlicePtrToSlice(v2Iter)
	if diff := cmp.Diff(v1, v2, cmpopts.EquateApprox(0, fn.Tolerance)); diff != "" {
		return fmt.Sprintf("For k=%d: diff=%s", k, diff)
	}

	return "" // No diff
}

func float64PtrToSlice(vIter func(*float64) bool) []float64 {
	var vSlice []float64
	var v float64
	for vIter(&v) {
		vSlice = append(vSlice, v)
	}
	return vSlice
}

func float64SlicePtrToSlice(vIter func(*[]float64) bool) []float64 {
	var vSlice []float64
	vIter(&vSlice) // We are expecting a single slice.
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

// LaplaceTolerance returns tolerance to be used in approxEquals or in threshold
// computations for tests with Laplace Noise to pass with 10⁻ᵏ flakiness.
// flakinessK is the parameter used to specify this.
//
// l1Sensitivity and epsilon are the DP parameters of the test.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
func LaplaceTolerance(flakinessK, l1Sensitivity, epsilon float64) float64 {
	return l1Sensitivity * flakinessK * math.Log(10) / epsilon
}

// ComplementaryLaplaceTolerance returns tolerance to be used in checkMetricsAreNoisy
// for tests with Laplace Noise to pass with 10⁻ᵏ flakiness. flakinessK is the
// parameter used to specify this.
//
// l1Sensitivity, epsilon and delta are the DP parameters of the test.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func ComplementaryLaplaceTolerance(flakinessK, l1Sensitivity, epsilon float64) float64 {
	// We need arbitrary precision arithmetics here because ln(1-10⁻ᵏ) evaluates to
	// 0 with float64, making the output 0.
	sum := big.NewFloat(math.Pow(10, -flakinessK)).SetMode(big.AwayFromZero) // 10⁻ᵏ
	sum.Neg(sum)                                                             // -10⁻ᵏ
	sum.SetMode(big.ToZero).Add(sum, big.NewFloat(1))                        // 1-10⁻ᵏ
	log, _ := sum.Float64()
	log = math.Log(log) // ln(1-10⁻ᵏ)
	return -l1Sensitivity * log / epsilon
}

// RoundedLaplaceTolerance rounds laplace tolerance value to the nearest integer,
// in order to work with tests for integer-valued aggregations.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func RoundedLaplaceTolerance(flakinessK, l1Sensitivity, epsilon float64) float64 {
	return math.Round(LaplaceTolerance(flakinessK, l1Sensitivity, epsilon))
}

// GaussianTolerance returns tolerance to be used in approxEquals or in threshold
// computations for tests with Gaussian Noise to pass with 10⁻ᵏ flakiness.
// flakinessK is the parameter used to specify this.
//
// l0Sensitivity, lInfSensitivity, epsilon and delta are the DP parameters of the test.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func GaussianTolerance(flakinessK, l0Sensitivity, lInfSensitivity, epsilon, delta float64) float64 {
	// We need arbitrary precision arithmetics here because (1-10⁻ᵏ) evaluates to
	// 1 with float64, making the output Inf.
	sum := big.NewFloat(math.Pow(10, -flakinessK)).SetMode(big.AwayFromZero) // 10⁻ᵏ
	sum.Neg(sum)                                                             // -10⁻ᵏ
	sum.SetMode(big.ToZero).Add(sum, big.NewFloat(1))                        // 1-10⁻ᵏ
	erfinv, _ := sum.Float64()
	erfinv = math.Erfinv(erfinv) // Erfinv(1-10⁻ᵏ)
	return erfinv * noise.SigmaForGaussian(int64(l0Sensitivity), lInfSensitivity, epsilon, delta) * math.Sqrt(2)
}

// ComplementaryGaussianTolerance returns tolerance to be used in checkMetricsAreNoisy
// for tests with Gaussian Noise to pass with 10⁻ᵏ flakiness. flakinessK is the
// parameter used to specify this.
//
// l0Sensitivity, lInfSensitivity, epsilon and delta are the DP parameters of the test.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func ComplementaryGaussianTolerance(flakinessK, l0Sensitivity, lInfSensitivity, epsilon, delta float64) float64 {
	return math.Erfinv(math.Pow(10, -flakinessK)) * noise.SigmaForGaussian(int64(l0Sensitivity), lInfSensitivity, epsilon, delta) * math.Sqrt(2)
}

// LaplaceToleranceForMean returns tolerance to be used in approxEquals for tests
// for mean to pass with 10⁻ᵏ flakiness.
//
// flakinessK is the parameter used to specify k in the flakiness.
// distanceFromMidPoint = upper - midPoint, where midPoint = (lower + upper)/2.
// exactNormalizedSum is a clamped (with boundaries -distanceFromMidPoint and distanceFromMidPoint) sum of distances of the input entities from the midPoint.
// exactNormalizedSum is needed for calculating tolerance because the algorithm of the mean aggregation uses noisy normalized sum in its calculations.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func LaplaceToleranceForMean(flakinessK, lower, upper float64, maxContributionsPerPartition, maxPartitionsContributed int64, epsilon float64, exactNormalizedSum, exactCount, exactMean float64) (float64, error) {
	// The term below is equivalent to -log_10(1-sqrt(1-1e-k)).
	// It is formulated this way to increase precision and to avoid having this term go to infinity.
	countFlakinessK := -math.Log10(-math.Expm1(0.5 * math.Log1p(-math.Pow(10, -flakinessK))))
	normalizedSumFlakinessK := countFlakinessK // We use the same flakiness for simplicity.
	halfEpsilon := epsilon / 2

	_, l1Count, _ := sensitivitiesForCount(maxContributionsPerPartition, maxPartitionsContributed)
	_, l1NormalizedSum, _ := sensitivitiesForNormalizedSum(lower, upper, maxContributionsPerPartition, maxPartitionsContributed)

	countTolerance := math.Ceil(LaplaceTolerance(countFlakinessK, l1Count, halfEpsilon))
	normalizedSumTolerance := LaplaceTolerance(normalizedSumFlakinessK, l1NormalizedSum, halfEpsilon)
	return ToleranceForMean(lower, upper, exactNormalizedSum, exactCount, exactMean, countTolerance, normalizedSumTolerance)
}

// ToleranceForMean returns tolerance to be used in approxEquals or checkMetricsAreNoisy for tests
// for mean to pass with 10⁻ᵏ flakiness.
//
// flakinessK is the parameter used to specify k in the flakiness.
// distanceFromMidPoint = upper - midPoint, where midPoint = (lower + upper)/2.
// exactNormalizedSum is a clamped (with boundaries -distanceFromMidPoint and distanceFromMidPoint) sum of distances of the input entities from the midPoint.
// exactNormalizedSum is needed for calculating tolerance because the algorithm of the mean aggregation uses noisy normalized sum in its calculations.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// see https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
//
// TODO: Use confidence intervals from the DP library instead of calculation tolerance here.
func ToleranceForMean(lower, upper, exactNormalizedSum, exactCount, exactMean, countTolerance, normalizedSumTolerance float64) (float64, error) {
	midPoint := lower + (upper-lower)/2.0

	minNoisyCount := math.Max(1.0, exactCount-countTolerance)
	maxNoisyCount := math.Max(1.0, exactCount+countTolerance)
	minNoisyNormalizedSum := exactNormalizedSum - normalizedSumTolerance
	maxNoisyNormalizedSum := exactNormalizedSum + normalizedSumTolerance
	minNoisyMeanCount := maxNoisyCount
	maxNoisyMeanCount := minNoisyCount
	// If the numerator (min/max noisy normalized sum) of the mean is negative,
	// min/max noisy counts should switch places to find min/max noisy mean.
	if minNoisyNormalizedSum < 0 {
		minNoisyMeanCount = minNoisyCount
	}
	if maxNoisyNormalizedSum < 0 {
		maxNoisyMeanCount = maxNoisyCount
	}
	minNoisyMean, err := dpagg.ClampFloat64(minNoisyNormalizedSum/minNoisyMeanCount+midPoint, lower, upper)
	if err != nil {
		return 0, err
	}
	maxNoisyMean, err := dpagg.ClampFloat64(maxNoisyNormalizedSum/maxNoisyMeanCount+midPoint, lower, upper)
	if err != nil {
		return 0, err
	}
	distFromMinNoisyMean := distanceBetween(exactMean, minNoisyMean)
	distFromMaxNoisyMean := distanceBetween(maxNoisyMean, exactMean)

	return math.Max(distFromMaxNoisyMean, distFromMinNoisyMean), nil
}

// QuantilesTolerance returns tolerance to be used in approxEquals for tests
// for quantiles to pass with negligible flakiness.
//
// When no noise is added, the quantiles should return a value that differs from the true
// quantile by no more than the size of the buckets the range is partitioned into, i.e.,
// (upper - lower) / (branchingFactor^treeHeight - 1).
//
// The tests don't disable noise, hence we multiply the tolerance by a reasonably small number,
// in this case 5, to account for the noise addition.
// TODO: Implement more accurate tolerance based on confidence intervals.
func QuantilesTolerance(lower, upper float64) float64 {
	return 5 * (upper - lower) / (math.Pow(float64(dpagg.DefaultBranchingFactor), float64(dpagg.DefaultTreeHeight)) - 1.0)
}

func distanceBetween(a, b float64) float64 {
	return math.Abs(a - b)
}

func sensitivitiesForNormalizedSum(lower, upper float64, maxContributionsPerPartition, maxPartitionsContributed int64) (l0Sensitivity, l1Sensitivity, lInfSensitivity float64) {
	midPoint := lower + (upper-lower)/2.0
	maxDistFromMidpoint := upper - midPoint
	l0Sensitivity = float64(maxPartitionsContributed)
	lInfSensitivity = maxDistFromMidpoint * float64(maxContributionsPerPartition)
	l1Sensitivity = l0Sensitivity * lInfSensitivity
	return l0Sensitivity, l1Sensitivity, lInfSensitivity
}

func sensitivitiesForCount(maxContributionsPerPartition, maxPartitionsContributed int64) (l0Sensitivity, l1Sensitivity, lInfSensitivity float64) {
	l0Sensitivity = float64(maxPartitionsContributed)
	lInfSensitivity = float64(maxContributionsPerPartition)
	l1Sensitivity = l0Sensitivity * lInfSensitivity
	return l0Sensitivity, l1Sensitivity, lInfSensitivity
}

// Int64Ptr transforms an int64 into an *int64.
func Int64Ptr(i int64) *int64 {
	return &i
}

// Float64Ptr transforms a float64 into a *float64.
func Float64Ptr(f float64) *float64 {
	return &f
}

// CheckFloat64MetricsAreNoisy checks that no values in a PCollection<testFloat64Metric>
// (where testFloat64Metric contains the aggregate statistic) is equal to exactMetric.
func CheckFloat64MetricsAreNoisy(s beam.Scope, col beam.PCollection, exactMetric, tolerance float64) {
	beam.ParDo0(s, &checkFloat64MetricsAreNoisyFn{exactMetric, tolerance}, col)
}

type checkFloat64MetricsAreNoisyFn struct {
	ExactMetric float64
	Tolerance   float64
}

func (fn *checkFloat64MetricsAreNoisyFn) ProcessElement(m TestFloat64Metric) error {
	if cmp.Equal(m.Metric, fn.ExactMetric, cmpopts.EquateApprox(0, fn.Tolerance)) {
		return fmt.Errorf("found a non-noisy output of %f for (value, exactOutput)=(%d, %f)", m.Metric, m.Value, fn.ExactMetric)
	}
	return nil
}

// CheckInt64MetricsAreNoisy checks that no values in a PCollection<testInt64Metric>
// (where testInt64Metric contains the aggregate statistic) is equal to exactMetric.
func CheckInt64MetricsAreNoisy(s beam.Scope, col beam.PCollection, exactMetric int, tolerance float64) {
	beam.ParDo0(s, &checkInt64MetricsAreNoisyFn{exactMetric, tolerance}, col)
}

type checkInt64MetricsAreNoisyFn struct {
	ExactMetric int
	Tolerance   float64
}

func (fn *checkInt64MetricsAreNoisyFn) ProcessElement(m TestInt64Metric) error {
	if cmp.Equal(float64(m.Metric), float64(fn.ExactMetric), cmpopts.EquateApprox(0, fn.Tolerance)) {
		return fmt.Errorf("found a non-noisy output of %d for (value, exactOutput)=(%d, %d)", m.Metric, m.Value, fn.ExactMetric)
	}
	return nil
}

// OneFn always returns 1.
func OneFn(beam.V) int { return 1 }

// CheckSomePartitionsAreDropped checks that the number of values in the PCollection
// is smaller than numPartitions, but larger than 0.
func CheckSomePartitionsAreDropped(s beam.Scope, col beam.PCollection, numPartitions int) {
	ones := beam.ParDo(s, OneFn, col)
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

// CheckNoNegativeValuesInt64Fn returns an error if an int64 value is negative.
func CheckNoNegativeValuesInt64Fn(v int64) error {
	if v < 0 {
		return fmt.Errorf("unexpected negative element: %v", v)
	}
	return nil
}

// CheckNoNegativeValuesFloat64Fn returns an error if an float64 value is negative.
func CheckNoNegativeValuesFloat64Fn(v float64) error {
	if v < 0 {
		return fmt.Errorf("unexpected negative element: %v", v)
	}
	return nil
}

// CheckAllValuesNegativeFloat64Fn returns an error if an float64 value is non-negative.
func CheckAllValuesNegativeFloat64Fn(v float64) error {
	if v >= 0 {
		return fmt.Errorf("unexpected non-negative element: %v", v)
	}
	return nil
}

// ApproxEquals returns true if x and y are approximately equal within
// a tolerance of 1e-10.
func ApproxEquals(x, y float64) bool {
	return cmp.Equal(x, y, cmpopts.EquateApprox(0, 1e-10))
}

// CheckNumPartitions checks that col has expected number of partitions.
func CheckNumPartitions(s beam.Scope, col beam.PCollection, expected int) {
	ones := beam.ParDo(s, OneFn, col)
	numPartitions := stats.Sum(s, ones)
	numPartitions = beam.AddFixedKey(s, numPartitions)

	want := beam.Create(s, expected)
	want = beam.AddFixedKey(s, want)
	coGroupToValue := beam.CoGroupByKey(s, numPartitions, want)
	beam.ParDo0(s, gotExpectedNumPartitionsFn, coGroupToValue)
}

func gotExpectedNumPartitionsFn(_ int, v1Iter, v2Iter func(*int) bool) error {
	got := getNumPartitions(v1Iter)
	want := getNumPartitions(v2Iter)
	if got != want {
		return fmt.Errorf("got %d emitted partitions, want %d", got, want)
	}
	return nil
}

func getNumPartitions(vIter func(*int) bool) (v int) {
	ok := vIter(&v)
	if !ok {
		return 0
	}
	return v
}

// DereferenceFloat64Slice returns the first and only element of the slice for
// each key in a PCollection<K, []float64>. Returns an error if the slice
// does not contain exactly 1 element.
func DereferenceFloat64Slice(v beam.V, r []float64) (beam.V, float64, error) {
	if len(r) != 1 {
		return v, 0.0, fmt.Errorf("dereferenceFloat64: r=%v does not contain a single element", r)
	}
	return v, r[0], nil
}
