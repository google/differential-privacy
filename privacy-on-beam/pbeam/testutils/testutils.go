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
	"errors"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"sort"
	"testing"

	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/filter"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func init() {
	register.DoFn3x1[beam.W, func(*int64) bool, func(*int64) bool, string](&diffInt64Fn{})
	register.Iter1[int64]()
	register.DoFn3x1[int, func(*float64) bool, func(*float64) bool, string](&diffFloat64Fn{})
	register.Iter1[float64]()
	register.DoFn3x1[int, func(*[]float64) bool, func(*[]float64) bool, string](&diffFloat64SliceFn{})
	register.Iter1[[]float64]()
	register.DoFn1x1[PairIF64, error](&checkFloat64MetricsAreNoisyFn{})
	register.DoFn1x1[PairII64, error](&checkInt64MetricsAreNoisyFn{})
	register.DoFn1x1[int, error](&checkSomePartitionsAreDroppedFn{})
	register.DoFn3x1[int, func(*int) bool, func(*int) bool, error](&gotExpectedNumPartitionsFn{})
	register.Iter1[int]()

	register.Function1x1[int64, *int64](Int64Ptr)
	register.Function1x1[float64, *float64](Float64Ptr)
	register.Function1x1[beam.V, int](OneFn)
	register.Function1x1[int64, error](CheckNoNegativeValuesInt64)
	register.Function1x1[float64, error](CheckNoNegativeValuesFloat64)
	register.Function1x1[float64, error](CheckAllValuesNegativeFloat64)
	register.Function2x1[int, int, PairII](KVToPair)
	register.Function1x2[PairII, int, int](PairToKV)
	register.Function2x1[int, int64, PairII64](KVToPairII64)
	register.Function1x2[PairII64, int, int64](PairII64ToKV)
	register.Function2x1[int, float64, PairIF64](KVToPairIF64)
	register.Function1x2[PairIF64, int, float64](PairIF64ToKV)
	register.Function2x1[int, []float64, PairIF64Slice](KVToPairIF64Slice)
	register.Function1x2[PairIF64Slice, int, []float64](PairIF64SliceToKV)
	register.Function2x1[int, kv.Pair, PairICodedKV](KVToPairICodedKV)
	register.Function1x2[PairICodedKV, int, kv.Pair](PairICodedKVToKV)
	register.Function2x3[beam.V, []float64, beam.V, float64, error](DereferenceFloat64Slice)
	register.Function1x2[TripleWithIntValue, int, int](TripleWithIntValueToKV)
	register.Function1x2[TripleWithIntValue, int, TripleWithIntValue](ExtractIDFromTripleWithIntValue)
	register.Function1x2[TripleWithFloatValue, int, float32](TripleWithFloatValueToKV)
	register.Function1x2[TripleWithFloatValue, int, TripleWithFloatValue](ExtractIDFromTripleWithFloatValue)
	register.Function3x1[int, func(*float64) bool, func(*float64) bool, string](lessThanOrEqualTo)
	register.Function2x1[string, string, string](CombineDiffs)
	register.Function1x1[string, error](ReportDiffs)
	register.Function1x1[string, error](reportEquals)
	register.Function1x1[string, error](reportGreaterThan)
	register.Function3x1[beam.X, func(*int) bool, func(*int) bool, string](diffIntFn)
	register.Function1x1[int64, bool](isNegativeInt64)
	register.Function1x1[int, error](checkNumNegativeElemCountIsPositive)
}

// PairII, PairII64, PairIF64, PairICodedKV and the related functions are helpers
// necessary to get a PCollection of KV type as input of a test Beam pipeline.

// PairII holds a key-value pair of type (int, int).
type PairII struct {
	Key   int
	Value int
}

// PairToKV transforms a PairII into an (int, int) key-value pair.
func PairToKV(p PairII) (k, v int) {
	return p.Key, p.Value
}

// KVToPair transforms an (int, int) key-value pair into a PairII.
func KVToPair(k, v int) PairII {
	return PairII{k, v}
}

// PairII64 holds a key-value pair of type (int, int64).
type PairII64 struct {
	Key   int
	Value int64
}

// KVToPairII64 transforms an (int, int64) key-value pair into a PairII64.
func KVToPairII64(v int, m int64) PairII64 {
	return PairII64{v, m}
}

// PairII64ToKV transforms a PairII64 into an (int, int64) key-value pair.
func PairII64ToKV(tm PairII64) (int, int64) {
	return tm.Key, tm.Value
}

// PairIF64 holds a key-value pair of type (int, float64).
type PairIF64 struct {
	Key   int
	Value float64
}

// KVToPairIF64 transforms an (int, float64) key-value pair into a PairIF64.
func KVToPairIF64(v int, m float64) PairIF64 {
	return PairIF64{v, m}
}

// PairIF64ToKV transforms a PairIF64 into an (int, float64) key-value pair.
func PairIF64ToKV(tm PairIF64) (int, float64) {
	return tm.Key, tm.Value
}

// PairIF64Slice holds a key-value pair of type (int, []float64).
type PairIF64Slice struct {
	Key   int
	Value []float64
}

// PairIF64SliceToKV transforms a PairIF64Slice into an (int, []float64) key-value pair.
func PairIF64SliceToKV(tm PairIF64Slice) (int, []float64) {
	return tm.Key, tm.Value
}

// KVToPairIF64Slice transforms an (int, []float64) key-value pair into a PairIF64Slice.
func KVToPairIF64Slice(v int, m []float64) PairIF64Slice {
	return PairIF64Slice{v, m}
}

// PairICodedKV holds a key-value pair of type (int, kv.Pair).
type PairICodedKV struct {
	Key   int
	Value kv.Pair
}

// PairICodedKVToKV transforms a PairICodedKV into an (int, kv.Pair) key-value pair.
func PairICodedKVToKV(p PairICodedKV) (k int, v kv.Pair) {
	return p.Key, p.Value
}

// KVToPairICodedKV transforms an (int, kv.Pair) key-value pair into a PairICodedKV.
func KVToPairICodedKV(k int, v kv.Pair) PairICodedKV {
	return PairICodedKV{k, v}
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

// EqualsKVInt checks that two PCollections col1 and col2 of type
// <K,int> are exactly equal.
func EqualsKVInt(t *testing.T, s beam.Scope, col1, col2 beam.PCollection) {
	t.Helper()
	wantV := reflect.TypeOf(int(0))
	if err := checkValueType(col1, wantV); err != nil {
		t.Fatalf("EqualsKVInt: unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		t.Fatalf("EqualsKVInt: unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, diffIntFn, coGroupToValue)
	combinedDiff := beam.Combine(s, CombineDiffs, diffs)
	beam.ParDo0(s, ReportDiffs, combinedDiff)
}

// EqualsKVInt64 checks that two PCollections col1 and col2 of type
// <K,int64> are exactly equal. Each key can only hold a single value.
func EqualsKVInt64(t *testing.T, s beam.Scope, col1, col2 beam.PCollection) {
	ApproxEqualsKVInt64(t, s, col1, col2, 0.0)
}

// EqualsKVFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are exactly equal. Each key can only hold a single value.
func EqualsKVFloat64(t *testing.T, s beam.Scope, col1, col2 beam.PCollection) {
	ApproxEqualsKVFloat64(t, s, col1, col2, 0.0)
}

// NotEqualsFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are different. Each key can only hold a single value.
func NotEqualsFloat64(t *testing.T, s beam.Scope, col1, col2 beam.PCollection) {
	t.Helper()
	wantV := reflect.TypeOf(float64(0))
	if err := checkValueType(col1, wantV); err != nil {
		t.Fatalf("NotEqualsFloat64: unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		t.Fatalf("NotEqualsFloat64: unexpected value type for col2: %v", err)
	}
	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffFloat64Fn{Tolerance: 0.0}, coGroupToValue)
	combinedDiff := beam.Combine(s, CombineDiffs, diffs)
	beam.ParDo0(s, reportEquals, combinedDiff)
}

// ApproxEqualsKVInt64 checks that two PCollections col1 and col2 of type
// <K,int64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and the value associated with
// key k in col1 is within the specified tolerance of the value associated with k
// in col2". Each key can only hold a single value.
func ApproxEqualsKVInt64(t *testing.T, s beam.Scope, col1, col2 beam.PCollection, tolerance float64) {
	t.Helper()
	wantV := reflect.TypeOf(int64(0))
	if err := checkValueType(col1, wantV); err != nil {
		t.Fatalf("ApproxEqualsKVInt64: unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		t.Fatalf("ApproxEqualsKVInt64: unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffInt64Fn{Tolerance: tolerance}, coGroupToValue)
	combinedDiff := beam.Combine(s, CombineDiffs, diffs)
	beam.ParDo0(s, ReportDiffs, combinedDiff)
}

// ApproxEqualsKVFloat64 checks that two PCollections col1 and col2 of type
// <K,float64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and the value associated with
// key k in col1 is within the specified tolerance of the value associated with k
// in col2". Each key can only hold a single value.
func ApproxEqualsKVFloat64(t *testing.T, s beam.Scope, col1, col2 beam.PCollection, tolerance float64) {
	t.Helper()
	wantV := reflect.TypeOf(float64(0))
	if err := checkValueType(col1, wantV); err != nil {
		t.Fatalf("ApproxEqualsKVFloat64: unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		t.Fatalf("ApproxEqualsKVFloat64: unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffFloat64Fn{Tolerance: tolerance}, coGroupToValue)
	combinedDiff := beam.Combine(s, CombineDiffs, diffs)
	beam.ParDo0(s, ReportDiffs, combinedDiff)
}

// LessThanOrEqualToKVFloat64 checks that for PCollections col1 and col2 of type
// <K,float64>, for each key k, value corresponding to col1 is less than or equal
// to the value corresponding in col2. Each key can only hold a single value.
func LessThanOrEqualToKVFloat64(t *testing.T, s beam.Scope, col1, col2 beam.PCollection) {
	t.Helper()
	wantV := reflect.TypeOf(float64(0))
	if err := checkValueType(col1, wantV); err != nil {
		t.Fatalf("LessThanOrEqualToKVFloat64: unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		t.Fatalf("LessThanOrEqualToKVFloat64: unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, lessThanOrEqualTo, coGroupToValue)
	combinedDiff := beam.Combine(s, CombineDiffs, diffs)
	beam.ParDo0(s, reportGreaterThan, combinedDiff)
}

// ApproxEqualsKVFloat64Slice checks that two PCollections col1 and col2 of type
// <K,[]float64> are approximately equal, where "approximately equal" means
// "the keys are the same in both col1 and col2, and each value in the slice
// associated with key k in col1 is within the specified tolerance of each value
// in the slice associated with k in col2". Each key can only hold a single slice.
func ApproxEqualsKVFloat64Slice(t *testing.T, s beam.Scope, col1, col2 beam.PCollection, tolerance float64) {
	t.Helper()
	wantV := reflect.TypeOf([]float64{0.0})
	if err := checkValueType(col1, wantV); err != nil {
		t.Fatalf("ApproxEqualsKVFloat64Slice: unexpected value type for col1: %v", err)
	}
	if err := checkValueType(col2, wantV); err != nil {
		t.Fatalf("ApproxEqualsKVFloat64Slice: unexpected value type for col2: %v", err)
	}

	coGroupToValue := beam.CoGroupByKey(s, col1, col2)
	diffs := beam.ParDo(s, &diffFloat64SliceFn{Tolerance: tolerance}, coGroupToValue)
	combinedDiff := beam.Combine(s, CombineDiffs, diffs)
	beam.ParDo0(s, ReportDiffs, combinedDiff)
}

func reportEquals(diffs string) error {
	if diffs != "" {
		return nil
	}
	return fmt.Errorf("collections are equal")
}

// ReportDiffs returns an error if diffs is not empty.
func ReportDiffs(diffs string) error {
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

// CombineDiffs concatenates two diff strings into a single string.
func CombineDiffs(diff1, diff2 string) string {
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
func (fn *diffInt64Fn) ProcessElement(k beam.W, v1Iter, v2Iter func(*int64) bool) string {
	var v1 = int64PtrToSlice(v1Iter)
	var v2 = int64PtrToSlice(v2Iter)
	if diff := cmp.Diff(v1, v2, cmpopts.EquateApprox(0, fn.Tolerance)); diff != "" {
		return fmt.Sprintf("For k=%v: diff=%s", k, diff)
	}
	return ""
}

// diffIntFn returns a diff between values associated with a key. It
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
	sort.Float64s(vSlice)
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
//   - flakinessK: parameter used to specify k in the flakiness.
//   - lower: minimum possible value of the input entities.
//   - upper: maximum possible value of the input entities.
//   - epsilon: the differential privacy parameter epsilon.
//   - exactNormalizedSum: clamped (with boundaries -distanceFromMidPoint and distanceFromMidPoint)
//     sum of distances of the input entities from the mid.
//
// distanceFromMidPoint = upper - midPoint, where midPoint = (lower + upper)/2.
//
// exactNormalizedSum is needed for calculating tolerance because the algorithm of the mean
// aggregation uses noisy normalized sum in its calculations.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func LaplaceToleranceForMean(flakinessK, lower, upper float64, maxContributionsPerPartition, maxPartitionsContributed int64, epsilon float64, exactNormalizedSum, exactCount, exactMean float64) (float64, error) {
	// The term below is equivalent to -log_10(1-sqrt(1-1e-k)).
	// It is formulated this way to increase precision and to avoid having this term go to infinity.
	countFlakinessK := -math.Log10(-math.Expm1(0.5 * math.Log1p(-math.Pow(10, -flakinessK))))
	normalizedSumFlakinessK := countFlakinessK // We use the same flakiness for simplicity.
	halfEpsilon := epsilon / 2

	computer := sensitivityComputer{
		Lower:                        lower,
		Upper:                        upper,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MaxPartitionsContributed:     maxPartitionsContributed,
	}
	l1Count := computer.SensitivitiesForCount().L1
	l1NormalizedSum := computer.SensitivitiesForNormalizedSum().L1

	countTolerance := math.Ceil(LaplaceTolerance(countFlakinessK, l1Count, halfEpsilon))
	normalizedSumTolerance := LaplaceTolerance(normalizedSumFlakinessK, l1NormalizedSum, halfEpsilon)
	return ToleranceForMean(lower, upper, exactNormalizedSum, exactCount, exactMean, countTolerance, normalizedSumTolerance)
}

// ToleranceForMean returns tolerance to be used in approxEquals or checkMetricsAreNoisy for tests
// for mean to pass with 10⁻ᵏ flakiness.
//
//   - flakinessK: parameter used to specify k in the flakiness.
//   - exactNormalizedSum: clamped (with boundaries -distanceFromMidPoint and distanceFromMidPoint)
//     sum of distances of the input entities from the midPoint.
//
// distanceFromMidPoint = upper - midPoint, where midPoint = (lower + upper)/2.
//
// exactNormalizedSum is needed for calculating tolerance because the algorithm of the mean
// aggregation uses noisy normalized sum in its calculations.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// see https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
func ToleranceForMean(lower, upper, exactNormalizedSum, exactCount, exactMean, countTolerance, normalizedSumTolerance float64) (float64, error) {
	midPoint := lower + (upper-lower)/2.0

	minNoisyCount := math.Max(1.0, exactCount-countTolerance)            // c_-
	maxNoisyCount := math.Max(1.0, exactCount+countTolerance)            // c_+
	minNoisyNormalizedSum := exactNormalizedSum - normalizedSumTolerance // s_-
	maxNoisyNormalizedSum := exactNormalizedSum + normalizedSumTolerance // s_+

	// Find m_- and m_+ such that {s \in [s_-, s_+] and c \in [c_-, c_+]} implies {m \in [m_-, m_+]}.
	//
	// 1. For m_-:
	//   - If s_- >= 0, then m_- = s_-/c_+.
	//   - Otherwise, m_- = s_-/c_-.
	// 2. For m_+:
	//   - If s_+ >= 0, then m_+ = s_+/c_-.
	//   - Otherwise, m_+ = s_+/c_+.

	getMBound := func(a, b, c float64) (float64, error) {
		// If the numerator (min/max noisy normalized sum) of the mean is negative,
		// min/max noisy counts should switch places to find min/max noisy mean.
		normalizedBound := a / b
		if a < 0 {
			normalizedBound = a / c
		}
		return dpagg.ClampFloat64(normalizedBound+midPoint, lower, upper)
	}
	// Get M_- = m_- + midPoint.
	minNoisyMean, err := getMBound(minNoisyNormalizedSum, maxNoisyCount, minNoisyCount)
	if err != nil {
		return 0, err
	}
	// Get M_+ = m_+ + midPoint.
	maxNoisyMean, err := getMBound(maxNoisyNormalizedSum, minNoisyCount, maxNoisyCount)
	if err != nil {
		return 0, err
	}

	// Return the tolerance as max(|exactMean - M_-| , |exactMean - M_+|).
	return math.Max(
		distanceBetween(exactMean, minNoisyMean),
		distanceBetween(maxNoisyMean, exactMean),
	), nil
}

// VarianceStatistics is a struct that contains the statistics related to a variance aggregation.
type VarianceStatistics struct {
	Count                  float64
	NormalizedSum          float64
	NormalizedSumOfSquares float64
	Mean                   float64
	Variance               float64
}

// ComputeMeanVariance computes the mean and variance fields based on the other fields in
// the original struct, plus the given bounds.
// Remember to call this function after having Count, NormalizedSum, and NormalizedSumOfSquares,
// and do not pass in inf or -inf as bounds.
// If the count is zero, the mean is set to the midPoint, and variance is set to zero.
func (s *VarianceStatistics) ComputeMeanVariance(upper, lower float64) {
	midPoint := (lower + upper) / 2
	if s.Count == 0 {
		s.Mean = midPoint
		s.Variance = 0
	} else {
		normalizedMean := s.NormalizedSum / s.Count
		s.Mean = normalizedMean + midPoint
		s.Variance = s.NormalizedSumOfSquares/s.Count - normalizedMean*normalizedMean
	}
}

// PerPartitionVarianceStatistics calculates the variance related statistics of each partition and
// returns a map of partition to varianceStatistics.
func PerPartitionVarianceStatistics(
	minValue, maxValue float64, contributions []TripleWithFloatValue,
) map[int]VarianceStatistics {
	midPoint := (minValue + maxValue) / 2
	m := make(map[int]VarianceStatistics)
	for _, triple := range contributions {
		partition := triple.Partition
		normalizedValue := min(maxValue, max(minValue, float64(triple.Value))) - midPoint
		var newStats VarianceStatistics
		if stats, ok := m[partition]; ok {
			newStats = stats
		}

		// Insert or update the statistics for the partition.
		newStats.Count++
		newStats.NormalizedSum += normalizedValue
		newStats.NormalizedSumOfSquares += normalizedValue * normalizedValue
		m[partition] = newStats
	}
	for partition, stats := range m {
		stats.ComputeMeanVariance(minValue, maxValue)
		m[partition] = stats
	}
	return m
}

// PerPartitionVarianceStatisticsInt is similar to PerPartitionVarianceStatistics but for input
// with TripleWithIntValue type.
func PerPartitionVarianceStatisticsInt(
	minValue, maxValue float64, contributions []TripleWithIntValue,
) map[int]VarianceStatistics {
	var floatTriples []TripleWithFloatValue
	for _, t := range contributions {
		floatTriples = append(floatTriples, TripleWithFloatValue{
			ID:        t.ID,
			Partition: t.Partition,
			Value:     float32(t.Value),
		})
	}
	return PerPartitionVarianceStatistics(minValue, maxValue, floatTriples)
}

// LaplaceToleranceForVariance returns tolerances to be used in approxEquals for tests
// for variance to pass with 10⁻ᵏ flakiness.
//
// The return values include the tolerances for count, normalized sum, normalized sum of squares,
// mean, and variance.
//
//   - flakinessK: parameter used to specify k in the flakiness.
//   - lower: minimum possible value of the input entities.
//   - upper: maximum possible value of the input entities.
//   - epsilon: the differential privacy parameter epsilon.
//   - stats.NormalizedSumOfSquares: \sum { (clamp(x_i, lower, upper) - midPoint) ^ 2 }
//   - stats.NormalizedSum: \sum { clamp(x_i, lower, upper) - midPoint }.
//
// distanceFromMidPoint = upper - midPoint, where midPoint = (lower + upper)/2.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf
func LaplaceToleranceForVariance(
	flakinessK, lower, upper float64, maxContributionsPerPartition, maxPartitionsContributed int64,
	epsilon float64, stats VarianceStatistics,
) (VarianceStatistics, error) {
	// The term below is equivalent to -log_10(1-cbrt(1-1e-k)).
	// It is formulated this way to increase precision and to avoid having this term go to infinity.
	// Count, normalized sum, and normalized square sum uses the same following flakiness for simplicity.
	newFlakinessK := -math.Log10(-math.Expm1(1 / 3. * math.Log1p(-math.Pow(10, -flakinessK))))
	newEpsilon := epsilon / 3

	computer := sensitivityComputer{
		Lower:                        lower,
		Upper:                        upper,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MaxPartitionsContributed:     maxPartitionsContributed,
	}
	l1Count := computer.SensitivitiesForCount().L1
	l1NormalizedSum := computer.SensitivitiesForNormalizedSum().L1
	l1NormalizedSumOfSquares := computer.SensitivitiesForNormalizedSumOfSquares().L1

	countTolerance := math.Ceil(LaplaceTolerance(newFlakinessK, l1Count, newEpsilon))
	normalizedSumTolerance := LaplaceTolerance(newFlakinessK, l1NormalizedSum, newEpsilon)
	normalizedSumOfSquaresTolerance := LaplaceTolerance(newFlakinessK, l1NormalizedSumOfSquares, newEpsilon)

	tolerances := VarianceStatistics{
		Count:                  countTolerance,
		NormalizedSum:          normalizedSumTolerance,
		NormalizedSumOfSquares: normalizedSumOfSquaresTolerance,
	}

	meanTolerance, err := ToleranceForMean(
		lower, upper, stats.NormalizedSum, stats.Count, stats.Mean,
		countTolerance, normalizedSumTolerance,
	)
	if err != nil {
		return VarianceStatistics{}, fmt.Errorf("ToleranceForMean: %w", err)
	}

	varianceTolerance, err := ToleranceForVariance(lower, upper, stats, tolerances)
	if err != nil {
		return VarianceStatistics{}, fmt.Errorf("ToleranceForVariance: %w", err)
	}

	tolerances.Mean = meanTolerance
	tolerances.Variance = varianceTolerance

	return tolerances, nil
}

// ToleranceForVariance returns tolerance to be used in approxEquals or checkMetricsAreNoisy for
// tests for variance to pass with 10⁻ᵏ flakiness.
//
//   - flakinessK: parameter used to specify k in the flakiness.
//   - lower: minimum possible value of the input entities.
//   - upper: maximum possible value of the input entities.
//   - exactStats: Count, NormalizedSum, and NormalizedSumOfSquares of the input entities.
//   - tolerances: tolerances for count, normalized sum, and normalized sum of squares.
//
// To see the logic and the math behind flakiness and tolerance calculation,
// see https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
func ToleranceForVariance(
	lower, upper float64,
	exactStats, tolerances VarianceStatistics,
) (float64, error) {

	minNoisyC := math.Max(1.0, exactStats.Count-tolerances.Count)                        // c_-
	maxNoisyC := math.Max(1.0, exactStats.Count+tolerances.Count)                        // c_+
	minNoisyNS := exactStats.NormalizedSum - tolerances.NormalizedSum                    // s_-
	maxNoisyNS := exactStats.NormalizedSum + tolerances.NormalizedSum                    // s_+
	minNoisyNSS := exactStats.NormalizedSumOfSquares - tolerances.NormalizedSumOfSquares // ss_-
	maxNoisyNSS := exactStats.NormalizedSumOfSquares + tolerances.NormalizedSumOfSquares // ss_+

	// Let mm denote the mean of noisy normalized squares, i.e. mm = ss/c.
	// Find the lower and upper bounds of mm, i.e. mm_- <= mm <= mm_+, using the following rules:
	//
	// 1. For mm_-:
	//   - If ss_- >= 0, then mm_- = ss_-/c_+.
	//   - Otherwise, mm_- = ss_-/c_-.
	// 2. For mm_+:
	//   - Since ss_+ is always non-negative, mm_+ = ss_+/c_-.

	// Calculate mm_-.
	minNoisyNMM := minNoisyNSS / maxNoisyC
	if minNoisyNSS < 0 {
		minNoisyNMM = minNoisyNSS / minNoisyC
	}
	// Calculate mm_+
	maxNoisyNMM := maxNoisyNSS / minNoisyC

	// Let m denote the mean of noisy normalized values, i.e. m = s/c.
	// Find the lower and upper bounds of m^2, i.e. (m2_-)^2 <= m^2 <= (m2_+)^2, as follows:
	//
	// 1. If s_- <= s_+ <= 0, then
	//   - m2_- = (s_+/c_+)^2.
	//   - m2_+ = (s_-/c_-)^2.
	// 2. If s_- <= 0 <= s_+, then
	//   - m2_- = 0.
	//   - m2_+ = ( max(|s_-|, |s_+|)/c_- )^2.
	// 3. If 0 <= s_- <= s_+, then
	//   - m2_- = (s_-/c_+ )^2.
	//   - m2_+ = (s_+/c_- )^2.

	// Calculate m2_- and m2_+.
	minNoisyNM2, maxNoisyNM2 := math.Pow(maxNoisyNS/maxNoisyC, 2), math.Pow(minNoisyNS/minNoisyC, 2)
	if minNoisyNS <= 0 && 0 <= maxNoisyNS {
		minNoisyNM2 = 0
		maxNoisyNM2 = math.Pow(math.Max(-minNoisyNS, maxNoisyNS)/minNoisyC, 2)
	} else if minNoisyNS >= 0 {
		minNoisyNM2 = math.Pow(minNoisyNS/maxNoisyC, 2)
		maxNoisyNM2 = math.Pow(maxNoisyNS/minNoisyC, 2)
	}

	// Because shifting the element value by midPoint does not change variance,
	// that is we have the noisy variance V = MM - M2 = mm - m2.
	// Given the bounds of mm and m2, we have (mm_- - m2_+) <= V <= (mm_+ - m2_-).

	// Return the tolerance as max(|exactVariance - V_-| , |exactVariance - V_+|).
	// However, we first clamp V_- and V_+ to the range [0, maxVariance],
	// where maxVariance = (upper - lower)^2/4.
	maxVariance := math.Pow(upper-lower, 2) / 4
	minNoisyV := minNoisyNMM - maxNoisyNM2
	minNoisyVariance, err := dpagg.ClampFloat64(minNoisyV, 0, maxVariance)
	if err != nil {
		return 0, fmt.Errorf("clamping minNoisyVariance(%v, %v, %v): %w",
			minNoisyV, 0, maxVariance, err)
	}
	maxNoisyV := maxNoisyNMM - minNoisyNM2
	maxNoisyVariance, err := dpagg.ClampFloat64(maxNoisyV, 0, maxVariance)
	if err != nil {
		return 0, fmt.Errorf("clamping maxNoisyVariance(%v, %v, %v): %w",
			maxNoisyV, 0, maxVariance, err)
	}

	return math.Max(
		distanceBetween(exactStats.Variance, minNoisyVariance),
		distanceBetween(exactStats.Variance, maxNoisyVariance),
	), nil
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
func QuantilesTolerance(lower, upper float64) float64 {
	return 5 * (upper - lower) / (math.Pow(float64(dpagg.DefaultBranchingFactor), float64(dpagg.DefaultTreeHeight)) - 1.0)
}

func distanceBetween(a, b float64) float64 {
	return math.Abs(a - b)
}

type sensitivityComputer struct {
	Lower, Upper                 float64
	MaxContributionsPerPartition int64
	MaxPartitionsContributed     int64
}

type sensitivity struct {
	L0   float64 // L0 sensitivity
	LInf float64 // LInf sensitivity
	L1   float64 // L1 sensitivity
}

func (c *sensitivityComputer) MaxDistFromMidPoint() float64 {
	midPoint := c.Lower + (c.Upper-c.Lower)/2.0
	return c.Upper - midPoint
}

func (c *sensitivityComputer) MaxContributions() float64 {
	return float64(c.MaxContributionsPerPartition) * float64(c.MaxPartitionsContributed)
}

func (c *sensitivityComputer) SensitivitiesForNormalizedSumOfSquares() sensitivity {
	maxDistFromMidPoint := c.MaxDistFromMidPoint()
	return sensitivity{
		L0:   float64(c.MaxPartitionsContributed),
		LInf: math.Pow(maxDistFromMidPoint, 2) * float64(c.MaxContributionsPerPartition),
		L1:   math.Pow(maxDistFromMidPoint, 2) * c.MaxContributions(),
	}
}

func (c *sensitivityComputer) SensitivitiesForNormalizedSum() sensitivity {
	maxDistFromMidpoint := c.MaxDistFromMidPoint()
	return sensitivity{
		L0:   float64(c.MaxPartitionsContributed),
		LInf: maxDistFromMidpoint * float64(c.MaxContributionsPerPartition),
		L1:   maxDistFromMidpoint * c.MaxContributions(),
	}
}

func (c *sensitivityComputer) SensitivitiesForCount() sensitivity {
	return sensitivity{
		L0:   float64(c.MaxPartitionsContributed),
		LInf: float64(c.MaxContributionsPerPartition),
		L1:   float64(c.MaxContributionsPerPartition) * float64(c.MaxPartitionsContributed),
	}
}

// Int64Ptr transforms an int64 into an *int64.
func Int64Ptr(i int64) *int64 {
	return &i
}

// Float64Ptr transforms a float64 into a *float64.
func Float64Ptr(f float64) *float64 {
	return &f
}

// CheckFloat64MetricsAreNoisy checks that no values in a PCollection<pairIF64>
// (where pairIF64 contains the aggregate statistic) is equal to exactMetric.
func CheckFloat64MetricsAreNoisy(s beam.Scope, col beam.PCollection, exactMetric, tolerance float64) {
	beam.ParDo0(s, &checkFloat64MetricsAreNoisyFn{exactMetric, tolerance}, col)
}

type checkFloat64MetricsAreNoisyFn struct {
	ExactMetric float64
	Tolerance   float64
}

func (fn *checkFloat64MetricsAreNoisyFn) ProcessElement(m PairIF64) error {
	if cmp.Equal(m.Value, fn.ExactMetric, cmpopts.EquateApprox(0, fn.Tolerance)) {
		return fmt.Errorf("found a non-noisy output of %f for (key, exactOutput)=(%d, %f)", m.Value, m.Key, fn.ExactMetric)
	}
	return nil
}

// CheckInt64MetricsAreNoisy checks that no values in a PCollection<pairII64>
// (where pairII64 contains the aggregate statistic) is equal to exactMetric.
func CheckInt64MetricsAreNoisy(s beam.Scope, col beam.PCollection, exactMetric int, tolerance float64) {
	beam.ParDo0(s, &checkInt64MetricsAreNoisyFn{exactMetric, tolerance}, col)
}

type checkInt64MetricsAreNoisyFn struct {
	ExactMetric int
	Tolerance   float64
}

func (fn *checkInt64MetricsAreNoisyFn) ProcessElement(m PairII64) error {
	if cmp.Equal(float64(m.Value), float64(fn.ExactMetric), cmpopts.EquateApprox(0, fn.Tolerance)) {
		return fmt.Errorf("found a non-noisy output of %d for (key, exactOutput)=(%d, %d)", m.Value, m.Key, fn.ExactMetric)
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

// CheckNoNegativeValuesInt64 returns an error if an int64 value is negative.
func CheckNoNegativeValuesInt64(v int64) error {
	if v < 0 {
		return fmt.Errorf("unexpected negative element: %v", v)
	}
	return nil
}

func isNegativeInt64(v int64) bool {
	return v < 0
}

func checkNumNegativeElemCountIsPositive(elemCount int) error {
	if elemCount == 0 {
		return errors.New("want at least one negative value, but got 0")
	}
	return nil
}

// CheckAtLeastOneValueNegativeInt64 operates on a PCollection<int64> and will
// return an error during runtime if none of the int64 values is negative.
func CheckAtLeastOneValueNegativeInt64(s beam.Scope, col beam.PCollection) {
	negativeValues := filter.Include(s, col, isNegativeInt64)
	numNegativeValues := stats.CountElms(s, negativeValues)
	beam.ParDo0(s, checkNumNegativeElemCountIsPositive, numNegativeValues)
}

// CheckNoNegativeValuesFloat64 returns an error if an float64 value is negative.
func CheckNoNegativeValuesFloat64(v float64) error {
	if v < 0 {
		return fmt.Errorf("unexpected negative element: %v", v)
	}
	return nil
}

// CheckAllValuesNegativeFloat64 returns an error if an float64 value is non-negative.
func CheckAllValuesNegativeFloat64(v float64) error {
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
	CheckApproxNumPartitions(s, col, expected, 0)
}

// CheckApproxNumPartitions checks that col has approximately expected number of partitions.
// col is allowed to have number of partitions within tolerance of expected.
func CheckApproxNumPartitions(s beam.Scope, col beam.PCollection, expected, tolerance int) {
	ones := beam.ParDo(s, OneFn, col)
	numPartitions := stats.Sum(s, ones)
	numPartitions = beam.AddFixedKey(s, numPartitions)

	want := beam.Create(s, expected)
	want = beam.AddFixedKey(s, want)
	coGroupToValue := beam.CoGroupByKey(s, numPartitions, want)
	beam.ParDo0(s, &gotExpectedNumPartitionsFn{tolerance}, coGroupToValue)
}

type gotExpectedNumPartitionsFn struct {
	Tolerance int
}

func (fn *gotExpectedNumPartitionsFn) ProcessElement(_ int, v1Iter, v2Iter func(*int) bool) error {
	got := getNumPartitions(v1Iter)
	want := getNumPartitions(v2Iter)
	if math.Abs(float64(got-want)) > float64(fn.Tolerance) {
		if fn.Tolerance != 0 {
			return fmt.Errorf("got %d emitted partitions, want %d +- %d", got, want, fn.Tolerance)
		}
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
