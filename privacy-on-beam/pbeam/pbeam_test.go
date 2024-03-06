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
	"reflect"
	"testing"

	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	testpb "github.com/google/differential-privacy/privacy-on-beam/v3/testdata"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/passert"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*testpb.TestAnon)(nil)))
	beam.RegisterType(reflect.TypeOf(protoPair{}))

	register.Function2x1[string, *testpb.TestAnon, protoPair](kvToProtoPair)
	register.Function2x1[string, ComplexStruct, structPair](kvToStructPair)
	register.Function1x2[int, int, int](addZeroIntKeyFn)
	register.Function1x2[int, int, int](addZeroIntValueFn)
}

func TestNewPrivacySpec(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		params  PrivacySpecParams
		wantErr bool
	}{
		{
			"all fields set",
			PrivacySpecParams{
				AggregationEpsilon:        1.0,
				AggregationDelta:          0.0,
				PartitionSelectionEpsilon: 1.0,
				PartitionSelectionDelta:   1e-5,
			},
			false,
		},
		{
			"only aggregation budget set",
			PrivacySpecParams{
				AggregationEpsilon: 1.0,
				AggregationDelta:   0.0,
			},
			false,
		},
		{
			"only partition selection budget set",
			PrivacySpecParams{
				PartitionSelectionEpsilon: 1.0,
				PartitionSelectionDelta:   1e-5,
			},
			false,
		},
		{
			"no budget is set",
			PrivacySpecParams{},
			true,
		},
		{
			"negative PartitionSelectionEpsilon",
			PrivacySpecParams{
				PartitionSelectionEpsilon: -1,
				PartitionSelectionDelta:   1e-5,
			},
			true,
		},
		{
			"negative PreThreshold",
			PrivacySpecParams{
				AggregationEpsilon:      1.0,
				PartitionSelectionDelta: 1e-5,
				PreThreshold:            -1,
			},
			true,
		},
		{
			"partition selection budget is not set when PreThreshold is set",
			PrivacySpecParams{
				AggregationEpsilon: 1.0,
				PreThreshold:       1,
			},
			true,
		},
		{
			"partition selection delta is not set when partition selection budget is set",
			PrivacySpecParams{
				PartitionSelectionEpsilon: 1.0,
			},
			true,
		},
		{
			"negative AggregationEpsilon",
			PrivacySpecParams{
				AggregationEpsilon: -1.0,
			},
			true,
		},
		{
			"negative AggregationDelta",
			PrivacySpecParams{
				AggregationEpsilon: 1.0,
				AggregationDelta:   -1e-5,
			},
			true,
		},
		{
			"AggregationDelta > 1",
			PrivacySpecParams{
				AggregationEpsilon: 1.0,
				AggregationDelta:   -1e-5,
			},
			true,
		},
		{
			"negative PartitionSelectionEpsilon",
			PrivacySpecParams{
				PartitionSelectionEpsilon: -1.0,
			},
			true,
		},
		{
			"negative PartitionSelectionDelta",
			PrivacySpecParams{
				PartitionSelectionEpsilon: 1.0,
				PartitionSelectionDelta:   -1e-5,
			},
			true,
		},
		{
			"PartitionSelectionDelta > 1",
			PrivacySpecParams{
				PartitionSelectionEpsilon: 1.0,
				PartitionSelectionDelta:   -1e-5,
			},
			true,
		},
	} {
		if _, err := NewPrivacySpec(tc.params); (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v, wantErr=%t", tc.desc, err, tc.wantErr)
		}
	}
}

type protoPair struct {
	Key string
	Pb  *testpb.TestAnon
}

func kvToProtoPair(key string, pb *testpb.TestAnon) protoPair {
	return protoPair{key, pb}
}

func TestMakePrivate(t *testing.T) {
	values := []testutils.PairII{
		{17, 42},
		{99, 0},
	}
	p, s, col := ptest.CreateList(values)
	colKV := beam.ParDo(s, testutils.PairToKV, col)

	// pcol should contain 17→42 and 99→0.
	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	got := beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, got, col)
	if err := ptest.Run(p); err != nil {
		t.Errorf("MakePrivate(%v) = %v, expected %v: %v", col, got, col, err)
	}
}

type SimpleStruct struct {
	String string
	Int    int
}

type ComplexStruct struct {
	String         string
	Int            int
	StringPointer  *string
	StringSlice    []string
	SubStruct      *SimpleStruct
	SubStructSlice []SimpleStruct
}

type structPair struct {
	Key   string
	Value ComplexStruct
}

func kvToStructPair(key string, value ComplexStruct) structPair {
	return structPair{Key: key, Value: value}
}

type RecursiveStruct struct {
	String          string
	Int             int
	SubStruct       ComplexStruct
	RecursiveStruct *RecursiveStruct
}

func TestMakePrivateFromStruct(t *testing.T) {
	fortyTwo := "42"
	seventeen := "17"
	for _, tc := range []struct {
		desc        string
		idFieldPath string
		values      []ComplexStruct
		want        []structPair
	}{
		{"top level string id field",
			"String",
			[]ComplexStruct{
				{String: "42", Int: 42},
				{String: "17", Int: 17}},
			[]structPair{
				{Key: "\"42\"", Value: ComplexStruct{String: "42", Int: 42}},
				{Key: "\"17\"", Value: ComplexStruct{String: "17", Int: 17}}},
		},
		{"top level string pointer id field",
			"StringPointer",
			[]ComplexStruct{
				{StringPointer: &fortyTwo, Int: 42},
				{StringPointer: &seventeen, Int: 17}},
			[]structPair{
				{Key: "\"42\"", Value: ComplexStruct{StringPointer: &fortyTwo, Int: 42}},
				{Key: "\"17\"", Value: ComplexStruct{StringPointer: &seventeen, Int: 17}}},
		},
		{"bottom level string id field",
			"SubStruct.String",
			[]ComplexStruct{
				{SubStruct: &SimpleStruct{String: "42"}, Int: 42},
				{SubStruct: &SimpleStruct{String: "17"}, Int: 17}},
			[]structPair{
				{Key: "\"42\"", Value: ComplexStruct{SubStruct: &SimpleStruct{String: "42"}, Int: 42}},
				{Key: "\"17\"", Value: ComplexStruct{SubStruct: &SimpleStruct{String: "17"}, Int: 17}}},
		},
	} {
		p, s, col, want := ptest.CreateList2(tc.values, tc.want)

		pcol := MakePrivateFromStruct(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}), tc.idFieldPath)
		got := beam.ParDo(s, kvToStructPair, pcol.col)
		passert.Equals(s, got, want)
		if err := ptest.Run(p); err != nil {
			t.Errorf("MakePrivateFromStruct output does not match input values with %s. got %v, expected %v: %v", tc.desc, got, want, err)
		}
	}
}

// Tests the GetIDField method in extractStructFieldFn.
func TestGetIDField(t *testing.T) {
	eight := "8"
	val := RecursiveStruct{
		String: "0",
		Int:    0,
		SubStruct: ComplexStruct{
			String:      "1",
			Int:         1,
			StringSlice: []string{"2", "3", "4"},
			SubStruct:   &SimpleStruct{String: "5", Int: 5},
			SubStructSlice: []SimpleStruct{
				SimpleStruct{String: "6", Int: 6},
				SimpleStruct{String: "7", Int: 7},
			},
			StringPointer: &eight},
		RecursiveStruct: &RecursiveStruct{String: "9", Int: 9},
	}
	for _, tc := range []struct {
		idFieldPath string
		want        any
		wantErr     bool
	}{
		{"String", "0", false},
		{"Int", 0, false},
		{"SubStruct", nil, true},
		{"SubStruct.String", "1", false},
		{"SubStruct.Int", 1, false},
		{"SubStruct.StringSlice", nil, true},
		{"SubStruct.SubStruct", nil, true},
		{"SubStruct.SubStruct.String", "5", false},
		{"SubStruct.SubStruct.Int", 5, false},
		{"SubStruct.SubStructSlice", nil, true},
		{"SubStruct.SubStructSlice.String", nil, true},
		{"SubStruct.SubStructSlice.Int", nil, true},
		{"SubStruct.StringPointer", "8", false},
		{"RecursiveStruct", nil, true},
		{"RecursiveStruct.String", "9", false},
		{"RecursiveStruct.Int", 9, false},
		{"RecursiveStruct.RecursiveStruct", nil, true},
		{"RecursiveStruct.RecursiveStruct.String", "", false},
		{"RecursiveStruct.RecursiveStruct.Int", 0, false},
		{"nonexistent", nil, true},
	} {
		ext := extractStructFieldFn{IDFieldPath: tc.idFieldPath}
		got, err := ext.getIDField(val)
		if (err != nil) != tc.wantErr {
			t.Errorf("GetIDField with idFieldPath=%s: got error %v, wantErr=%t.", tc.idFieldPath, err, tc.wantErr)
		}
		if !cmp.Equal(got, tc.want) {
			t.Errorf("GetIDField with idFieldPath=%s: retrieved field %v, wanted=%v.", tc.idFieldPath, got, tc.want)
		}
	}
}

func TestMakePrivateFromProto(t *testing.T) {
	values := []*testpb.TestAnon{
		&testpb.TestAnon{Foo: proto.Int64(42), Bar: proto.String("fourty-two")},
		&testpb.TestAnon{Foo: proto.Int64(17), Bar: proto.String("seventeen")},
		&testpb.TestAnon{Bar: proto.String("zero")},
	}
	result := []protoPair{
		{"42", &testpb.TestAnon{Foo: proto.Int64(42), Bar: proto.String("fourty-two")}},
		{"17", &testpb.TestAnon{Foo: proto.Int64(17), Bar: proto.String("seventeen")}},
		{"0", &testpb.TestAnon{Bar: proto.String("zero")}},
	}
	p, s, col, want := ptest.CreateList2(values, result)

	pcol := MakePrivateFromProto(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}), "foo")
	got := beam.ParDo(s, kvToProtoPair, pcol.col)
	passert.Equals(s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Errorf("MakePrivateFromProto(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

var (
	repeat    = []string{"bar", "baz"}
	subrepeat = []*testpb.TestComplex_Submessage{
		&testpb.TestComplex_Submessage{
			Simple: proto.String("oob"),
			Repeat: repeat,
		},
		&testpb.TestComplex_Submessage{
			Simple: proto.String("obo"),
			Repeat: repeat,
		},
	}
	complexMsg = &testpb.TestComplex{
		Simple: proto.String("foo"),
		Repeat: repeat,
		Sub: &testpb.TestComplex_Submessage{
			Simple: proto.String("boo"),
			Repeat: repeat,
		},
		Subrepeat: subrepeat,
	}
	withoutSimple = &testpb.TestComplex{
		Repeat: repeat,
		Sub: &testpb.TestComplex_Submessage{
			Simple: proto.String("boo"),
			Repeat: repeat,
		},
		Subrepeat: subrepeat,
	}
	withoutSubSimple = &testpb.TestComplex{
		Simple: proto.String("foo"),
		Repeat: repeat,
		Sub: &testpb.TestComplex_Submessage{
			Repeat: repeat,
		},
		Subrepeat: subrepeat,
	}
)

// Tests the extraction logic in extractProtoFieldFn.
func TestExtractProtoField(t *testing.T) {
	for _, tc := range []struct {
		idFieldPath string
		wantField   string
		wantMsg     *testpb.TestComplex
		ok          bool
	}{
		{"simple", "foo", complexMsg, true},
		{"empty", "", complexMsg, true},
		{"sub.simple", "boo", complexMsg, true},
		{"repeat", "", nil, false},
		{"sub.repeat", "", nil, false},
		{"subrepeat.simple", "", nil, false},
		{"subrepeat.repeat", "", nil, false},
		{"nonexistent", "", nil, false},
	} {
		ext := &extractProtoFieldFn{
			IDFieldPath: tc.idFieldPath,
			desc:        (&testpb.TestComplex{}).ProtoReflect().Descriptor(),
		}
		clone := &testpb.TestComplex{}
		proto.Merge(clone, complexMsg)
		gotField, err := ext.extractField(clone.ProtoReflect())
		if (err == nil) != tc.ok {
			t.Errorf("extractField with IDFieldPath=%s: got error %v, want ok=%t.", tc.idFieldPath, err, tc.ok)
		}
		if err == nil {
			gotField := gotField.(string)
			if gotField != tc.wantField {
				t.Errorf("extractField with IDFieldPath=%s: got field %v, want %s", tc.idFieldPath, gotField, tc.wantField)
			}
			if !proto.Equal(clone, tc.wantMsg) {
				t.Errorf("extractField with IDFieldPath=%s: got msg %v, want %v", tc.idFieldPath, clone, tc.wantMsg)
			}
		}
	}
}

// Tests that we can get get the whole budget and consume it partially afterwards.
func TestGetFullBudget(t *testing.T) {
	budget := &privacyBudget{epsilon: 2, delta: 2e-10}
	eps, del, err := budget.get(0, 0)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 2.0 || del != 2e-10 {
		t.Errorf("Trying to get the whole budget: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 2.0, 2e-10)
	}

	// Split the budget and consume it in two calls.
	eps, del, err = budget.consume(1, 1e-10)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 1.0 || del != 1e-10 {
		t.Errorf("Trying to consume the budget after budget.get() call: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 1.0, 1e-10)
	}
	eps, del, err = budget.consume(1, 1e-10)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 1.0 || del != 1e-10 {
		t.Errorf("Trying to consume the budget after budget.get() call: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 1.0, 1e-10)
	}
}

// Tests that we can get and consume the budget partially.
func TestGetPartialBudget(t *testing.T) {
	budget := &privacyBudget{epsilon: 2, delta: 2e-10}
	eps, del, err := budget.get(1, 1e-10)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 1.0 || del != 1e-10 {
		t.Errorf("Trying to get first half of the budget: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 1.0, 1e-10)
	}

	eps, del, err = budget.consume(1, 1e-10)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 1.0 || del != 1e-10 {
		t.Errorf("Trying to consume second half of the budget after budget.get() call: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 1.0, 1e-10)
	}

	eps, del, err = budget.get(1, 1e-10)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 1.0 || del != 1e-10 {
		t.Errorf("Trying to get second half of the budget: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 1.0, 1e-10)
	}

	eps, del, err = budget.consume(1, 1e-10)
	if err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	if eps != 1.0 || del != 1e-10 {
		t.Errorf("Trying to consume second half the budget after budget.get() call: Got (epsilon,delta)=(%f,%e), expected=(%f,%e)", eps, del, 1.0, 1e-10)
	}
}

// Tests that we can consume all the budget at once.
func TestBudgetFullyConsumed(t *testing.T) {
	values := []testutils.PairII{
		{1, 1},
		{2, 2},
	}
	p, s, col := ptest.CreateList(values)
	colKV := beam.ParDo(s, testutils.PairToKV, col)
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1, PartitionSelectionEpsilon: 1, PartitionSelectionDelta: 1e-30})
	pcol := MakePrivate(s, colKV, spec)
	got := Count(s, pcol, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	passert.Empty(s, got)
	if err := ptest.Run(p); err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	// Try consuming 1% of the initial budget.
	if eps, del, err := spec.aggregationBudget.consume(0.01, 0); err == nil {
		t.Errorf("expected spec to be out of budget, but could consume (%f,%e) without any error", eps, del)
	}
	if eps, del, err := spec.partitionSelectionBudget.consume(0.01, 1e-32); err == nil {
		t.Errorf("expected spec to be out of budget, but could consume (%f,%e) without any error", eps, del)
	}
}

// Tests that two distinct budgets can be independently consumed.
func TestTwoDistinctBudgets(t *testing.T) {
	values := []testutils.PairII{
		{1, 1},
		{2, 2},
	}
	p, s, col := ptest.CreateList(values)
	colKV := beam.ParDo(s, testutils.PairToKV, col)
	spec1 := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1, PartitionSelectionEpsilon: 1, PartitionSelectionDelta: 1e-30})
	spec2 := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1, PartitionSelectionEpsilon: 1, PartitionSelectionDelta: 1e-30})
	pcol1 := MakePrivate(s, colKV, spec1)
	pcol2 := MakePrivate(s, colKV, spec2)
	got1 := Count(s, pcol1, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	got2 := Count(s, pcol2, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	passert.Empty(s, got1)
	passert.Empty(s, got2)
	if err := ptest.Run(p); err != nil {
		t.Errorf("expected no error but got error: %v", err)
	}
	// Try consuming 1% of the initial budget independently for ε and δ.
	if eps, del, err := spec1.aggregationBudget.consume(0.01, 0); err == nil {
		t.Errorf("expected spec to be out of budget, but could consume (%f,%e) without any error", eps, del)
	}
	if eps, del, err := spec1.partitionSelectionBudget.consume(0.01, 1e-32); err == nil {
		t.Errorf("expected spec to be out of budget, but could consume (%f,%e) without any error", eps, del)
	}
	if eps, del, err := spec2.aggregationBudget.consume(0.01, 0); err == nil {
		t.Errorf("expected spec to be out of budget, but could consume (%f,%e) without any error", eps, del)
	}
	if eps, del, err := spec2.partitionSelectionBudget.consume(0.01, 1e-32); err == nil {
		t.Errorf("expected spec to be out of budget, but could consume (%f,%e) without any error", eps, del)
	}
}

// Test for rounding errors during budget allocation. For example, dividing the overall
// epsilon by 3 would lead to rounding errors unless mitigated in getPartialBudget.
func TestBudgetRounding(t *testing.T) {
	for numSplits := 1; numSplits <= 10; numSplits++ {
		budget := &privacyBudget{epsilon: 1, delta: 1e-10}
		epsPerAggregation := 1. / float64(numSplits)
		delPerAggregation := 1e-10 / float64(numSplits)
		for i := 0; i < numSplits; i++ {
			if _, _, err := budget.consume(epsPerAggregation, delPerAggregation); err != nil {
				t.Errorf("with %d aggregations, expected no error but got error: %v", numSplits, err)
			}
		}
		// Now, the budget should be really empty.
		if eps, del, err := budget.consume(1e-15, 1e-40); err == nil {
			t.Errorf("with %d aggregations, expected spec to be out of budget, but could consume (%f,%e) without any error", numSplits, eps, del)
		}
	}
}

func TestDropKey(t *testing.T) {
	// Input is two contributions: (privacy_id, value) = {(1, 100), (2, 100)}.
	// We add a test key of 0 and remove it with DropKey(), which should be a no-op.
	// We then compute a Count() with high epsilon and public partitions, the output should be:
	// (partition_key, value) = (100, 2)
	values := []testutils.PairII{
		{1, 100},
		{2, 100},
	}
	result := []testutils.PairII64{
		{100, 2},
	}
	p, s, col, want := ptest.CreateList2(values, result)
	colKV := beam.ParDo(s, testutils.PairToKV, col)
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e10})
	pcol := MakePrivate(s, colKV, spec)
	pcol = ParDo(s, addZeroIntKeyFn, pcol)
	pcol = DropKey(s, pcol)

	// Assert that adding a test key of 0 and removing it is a no-op.
	testutils.EqualsKVInt(t, s, pcol.col, colKV)
	publicPartitionsSlice := []int{100}
	publicPartitions := beam.CreateList(s, publicPartitionsSlice)
	got := Count(s, pcol, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)

	testutils.EqualsKVInt64(t, s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Error(err)
	}
}

func TestDropValue(t *testing.T) {
	// Input is two contributions: (privacy_id, value) = {(1, 100), (2, 100)}.
	// We add a test value of 0 and remove it with DropValue(), which should be a no-op.
	// If we then compute a Count() with high epsilon and public partitions, the output should be:
	// (partition_key, value) = (100, 2)
	values := []testutils.PairII{
		{1, 100},
		{2, 100},
	}
	result := []testutils.PairII64{
		{100, 2},
	}
	p, s, col, want := ptest.CreateList2(values, result)
	colKV := beam.ParDo(s, testutils.PairToKV, col)
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e10})
	pcol := MakePrivate(s, colKV, spec)
	pcol = ParDo(s, addZeroIntValueFn, pcol)
	pcol = DropValue(s, pcol)

	// Assert that adding a test value of 0 and removing it is a no-op.
	testutils.EqualsKVInt(t, s, pcol.col, colKV)

	publicPartitionsSlice := []int{100}
	publicPartitions := beam.CreateList(s, publicPartitionsSlice)
	got := Count(s, pcol, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)

	testutils.EqualsKVInt64(t, s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Error(err)
	}
}

func addZeroIntKeyFn(v int) (int, int) {
	return 0, v
}

func addZeroIntValueFn(k int) (int, int) {
	return k, 0
}
