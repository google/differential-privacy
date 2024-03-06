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

package pbeam

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/generated"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/funcx"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/runners/direct"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/passert"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/google/go-cmp/cmp"
)

func execute(ctx context.Context, p *beam.Pipeline) error {
	_, err := direct.Execute(ctx, p)
	return err
}

// Reusable between ParDo tests
var codec = kv.NewCodec(reflect.TypeOf(1), reflect.TypeOf(1))

var values = []testutils.PairII{
	{17, 42},
	{99, 0},
}

var goodResult = []testutils.PairII{
	{17, 26},
	{99, 5},
}

// Expected result for a cancelled context. The direct runner has a bug: when a
// doFn returns an error, the output is still collected (and contains the
// default value for the type).
var zeroResult = []testutils.PairII{
	{17, 0},
	{99, 0},
}

var goodResult2x1 = []testutils.PairII{
	{17, 106},
	{99, 1},
}

// testutils.PairICodedKVs and kv.Pairs shared among different tests.
// Initialized in init().
var goodResult2x2 []testutils.PairICodedKV
var valuesCodedKV []testutils.PairICodedKV
var zeroValuedCodedKV []testutils.PairICodedKV
var pairK84V22, pairK0V1, pairK0V0, pairK106V62, pairK1VMinus1, pairK42V11, pairK53V31 kv.Pair

func init() {
	// We call the Setup method to supply the encoders and decoders inside codec at runtime.
	codec.Setup()
	var err error
	pairK84V22, err = codec.Encode(84, 22)
	if err != nil {
		log.Exit(err)
	}
	pairK0V1, err = codec.Encode(0, 1)
	if err != nil {
		log.Exit(err)
	}
	pairK0V0, err = codec.Encode(0, 0)
	if err != nil {
		log.Exit(err)
	}
	pairK106V62, err = codec.Encode(106, 62)
	if err != nil {
		log.Exit(err)
	}
	pairK1VMinus1, err = codec.Encode(1, -1)
	if err != nil {
		log.Exit(err)
	}
	pairK42V11, err = codec.Encode(42, 11)
	if err != nil {
		log.Exit(err)
	}
	pairK53V31, err = codec.Encode(53, 31)
	if err != nil {
		log.Exit(err)
	}
	valuesCodedKV = []testutils.PairICodedKV{
		{17, pairK84V22},
		{99, pairK0V1},
	}
	zeroValuedCodedKV = []testutils.PairICodedKV{
		{17, pairK0V0},
		{99, pairK0V0},
	}
	goodResult2x2 = []testutils.PairICodedKV{
		{17, pairK106V62},
		{99, pairK1VMinus1},
	}
}

func compareCodecs(codec1, codec2 *kv.Codec) bool {
	if codec1 == nil && codec2 == nil {
		return true
	}
	if (codec1 == nil) != (codec2 == nil) {
		return false
	}
	if codec1.KType == codec2.KType && codec1.VType == codec2.VType {
		return true
	}
	return false
}

func compareTypeDefs(typeDef1, typeDef2 beam.TypeDefinition) bool {
	if typeDef1.Var == typeDef2.Var && typeDef1.T == typeDef2.T {
		return true
	}
	return false
}

func TestParDo1x1(t *testing.T) {
	doFn := func(v int) int { return v/2 + 5 }
	p, s, col, wantCol := ptest.CreateList2(values, goodResult)
	colKV := beam.ParDo(s, testutils.PairToKV, col)

	// pcol should contain 17→42 and 99→0.
	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	// We change that to 17→26 and 99→5 in the PrivatePCollection
	pcol = ParDo(s, doFn, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, gotCol, wantCol)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("Got an error: %v", err)
	}
}

func TestParDo1x2(t *testing.T) {
	doFn := func(v int) (int, int) { return v * 2, v/2 + 1 }

	p, s, col, wantCol := ptest.CreateList2(values, valuesCodedKV)
	wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
	colKV := beam.ParDo(s, testutils.PairToKV, col)

	// pcol should contain 17→42 and 99→0.
	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
	pcol = ParDo(s, doFn, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
	passert.Equals(s, gotCol, wantCol)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("Got an error: %v", err)
	}
	if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
		t.Errorf("DoFn returned a PrivatePCollection with wrong codec, diff=%s", diff)
	}
}

func TestParDoCtx1x2(t *testing.T) {
	for _, tc := range []struct {
		desc string
		m    func(context.Context) context.Context // function that modifies a context
		want []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context", unchangedContext, valuesCodedKV},
		// various context modifiers that will cancel the context before any work starts
		{"cancel", cancel, zeroValuedCodedKV},
		{"instantDeadline", instantDeadline, zeroValuedCodedKV},
		{"instantTimeout", instantTimeout, zeroValuedCodedKV},
	} {
		doFn := func(ctx context.Context, v int) (int, int) {
			if ctx.Err() != nil {
				return 0, 0
			}
			return v * 2, v/2 + 1
		}

		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDo1x2Err(t *testing.T) {
	hasErr := func(v int) (int, int, error) {
		return 0, 0, errors.New("sample error")
	}
	noErr := func(v int) (int, int, error) {
		return v * 2, v/2 + 1, nil
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(v int) (int, int, error)
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		{"doFn that does not return an error", noErr, false, valuesCodedKV},
		{"doFn that returns an error", hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDoCtx1x2Err(t *testing.T) {
	hasErr := func(ctx context.Context, v int) (int, int, error) {
		return 0, 0, errors.New("sample error")
	}
	noErr := func(ctx context.Context, v int) (int, int, error) {
		if ctx.Err() != nil {
			return 0, 0, nil
		}
		return v * 2, v/2 + 1, nil
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int) (int, int, error)
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, valuesCodedKV},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroValuedCodedKV},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDo2x1(t *testing.T) {
	doFn := func(k int, v int) int { return k + v }

	p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, goodResult2x1)
	colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

	// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	// We change that to 17→106 and 99→1 in the PrivatePCollection
	pcol = ParDo(s, doFn, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, gotCol, wantCol)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("Got an error: %v", err)
	}
}

func TestParDoCtx2x1(t *testing.T) {
	for _, tc := range []struct {
		desc string
		// m is a function that modifies a context
		m    func(context.Context) context.Context
		want []testutils.PairII
	}{
		// good context
		{"unchanged context", unchangedContext, goodResult2x1},
		// various context modifiers that will cancel the context before any work starts
		{"cancel", cancel, zeroResult},
		{"instantDeadline", instantDeadline, zeroResult},
		{"instantTimeout", instantTimeout, zeroResult},
	} {
		doFn := func(ctx context.Context, k, v int) int {
			if ctx.Err() != nil {
				return 0
			}
			return k + v
		}

		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<codedKV{84, 22}> and 99→<codedKV{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
	}
}

func TestParDo2x1Err(t *testing.T) {
	noErr := func(k, v int) (int, error) {
		return k + v, nil
	}
	hasErr := func(_, _ int) (int, error) {
		return 0, errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(int, int) (int, error)
		returnsErr bool
		want       []testutils.PairII
	}{
		{"doFn that does not return an error", noErr, false, goodResult2x1},
		{"doFn that returns an error", hasErr, true, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<codedKV{84, 22}> and 99→<codedKV{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
	}
}

func TestParDoCtx2x1Err(t *testing.T) {
	noErr := func(ctx context.Context, k, v int) (int, error) {
		if ctx.Err() != nil {
			return 0, nil
		}
		return k + v, nil
	}
	hasErr := func(_ context.Context, _, _ int) (int, error) {
		return 0, errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int, int) (int, error)
		returnsErr bool
		want       []testutils.PairII
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, goodResult2x1},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroResult},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroResult},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroResult},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<codedKV{84, 22}> and 99→<codedKV{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
	}
}

func TestParDo2x2(t *testing.T) {
	doFn := func(k int, v int) (int, int) { return k + v, k - v }

	p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, goodResult2x2)
	wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
	colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

	// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
	pcol = ParDo(s, doFn, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
	passert.Equals(s, gotCol, wantCol)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("Got an error: %v", err)
	}
	if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
		t.Errorf("DoFn returned a PrivatePCollection with wrong codec, diff=%s", diff)
	}
}

func TestParDoCtx2x2(t *testing.T) {
	for _, tc := range []struct {
		desc string
		m    func(context.Context) context.Context // function that modifies a context
		want []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context", unchangedContext, goodResult2x2},
		// various context modifiers that will cancel the context before any work starts
		{"cancel", cancel, zeroValuedCodedKV},
		{"instantDeadline", instantDeadline, zeroValuedCodedKV},
		{"instantTimeout", instantTimeout, zeroValuedCodedKV},
	} {
		doFn := func(ctx context.Context, k, v int) (int, int) {
			if ctx.Err() != nil {
				return 0, 0
			}
			return k + v, k - v
		}

		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDo2x2Err(t *testing.T) {
	noErr := func(k, v int) (int, int, error) {
		return k + v, k - v, nil
	}
	hasErr := func(_, _ int) (int, int, error) {
		return 0, 0, errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(int, int) (int, int, error)
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		{"doFn that does not return an error", noErr, false, goodResult2x2},
		{"doFn that returns an error", hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDoCtx2x2Err(t *testing.T) {
	noErr := func(ctx context.Context, k, v int) (int, int, error) {
		if ctx.Err() != nil {
			return 0, 0, nil
		}
		return k + v, k - v, nil
	}
	hasErr := func(_ context.Context, _, _ int) (int, int, error) {
		return 0, 0, errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int, int) (int, int, error)
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, goodResult2x2},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroValuedCodedKV},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

// Functions that modify a context.Context, so that tests can have a
// cancelled or expired context
func unchangedContext(ctx context.Context) context.Context {
	return ctx
}

func cancel(ctx context.Context) context.Context {
	newCtx, cancel := context.WithCancel(ctx)
	cancel()
	return newCtx
}

// These two functions cancel the context in a goroutine
// so that the context can "expire" naturally
func instantDeadline(ctx context.Context) context.Context {
	newCtx, cancel := context.WithDeadline(ctx, time.Now())
	go time.AfterFunc(time.Second, cancel)
	return newCtx
}

func instantTimeout(ctx context.Context) context.Context {
	newCtx, cancel := context.WithTimeout(ctx, 0)
	go time.AfterFunc(time.Second, cancel)
	return newCtx
}

// Reusable doFn that uses contexts
func doFnWithContext(ctx context.Context, v int) int {
	if ctx.Err() != nil {
		return 0
	}
	return v/2 + 5
}

func TestParDoCtx1x1(t *testing.T) {
	for _, tc := range []struct {
		// m is a function that modifies a context
		m    func(context.Context) context.Context
		want []testutils.PairII
	}{
		// good context
		{unchangedContext, goodResult},
		// various context modifiers that will cancel the context before any work starts
		{cancel, zeroResult},
		{instantDeadline, zeroResult},
		{instantTimeout, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// If the context was cancelled, we change that to 17→0 and 99→0 in the PrivatePCollection,
		// Otherwise, we change that to 17→26 and 99→5
		pcol = ParDo(s, doFnWithContext, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		// Run the beam pipeline with the modified context
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("Got an error: %v", err)
		}
	}
}

func TestParDo1x1Err(t *testing.T) {
	doFn := func(v int) (int, error) {
		return v/2 + 5, nil
	}
	p, s, col, wantCol := ptest.CreateList2(values, goodResult)
	colKV := beam.ParDo(s, testutils.PairToKV, col)

	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	pcol = ParDo(s, doFn, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, gotCol, wantCol)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("Got an error: %v", err)
	}
}

func TestParDo1x1ErrReturnsError(t *testing.T) {
	doFn := func(v int) (int, error) {
		return 0, errors.New("this function always returns an error")
	}
	p, s, col, wantCol := ptest.CreateList2(values, zeroResult)
	colKV := beam.ParDo(s, testutils.PairToKV, col)

	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	pcol = ParDo(s, doFn, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, gotCol, wantCol)
	if err := execute(context.Background(), p); err == nil {
		t.Errorf("Expected runner to return an error, but didn't get one")
	}
}

func TestParDoCtx1x1Err(t *testing.T) {
	doFn := func(ctx context.Context, v int) (int, error) {
		return doFnWithContext(ctx, v), nil
	}
	for _, tc := range []struct {
		// m is a function that modifies a context
		m    func(context.Context) context.Context
		want []testutils.PairII
	}{
		// good context
		{unchangedContext, goodResult},
		// various context modifiers that will cancel the context before any work starts
		{cancel, zeroResult},
		{instantDeadline, zeroResult},
		{instantTimeout, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// If the context was cancelled, we change that to 17→0 and 99→0 in the PrivatePCollection,
		// Otherwise, we change that to 17→26 and 99→5
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		// Run the beam pipeline with the modified context
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("Got an error: %v", err)
		}
	}
}

func doFnPair(b int, emit func(int)) {
	emit(b)
}

func doFnPairWithCtx(_ context.Context, b int, emit func(int)) {
	emit(b)
}

func TestParDo1x1Emit(t *testing.T) {
	values := []testutils.PairII{
		{17, 42},
		{19, 10},
		{80, 99},
		{99, 0},
	}

	p, s, col := ptest.CreateList(values)
	colKV := beam.ParDo(s, testutils.PairToKV, col)

	pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	pcol = ParDo(s, doFnPair, pcol)
	gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, gotCol, col)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("Without context, got an error: %v", err)
	}

	// Check for values with ctx passed in doFn
	pcol = MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
	pcol = ParDo(s, doFnPairWithCtx, pcol)
	gotCol = beam.ParDo(s, testutils.KVToPair, pcol.col)
	passert.Equals(s, gotCol, col)
	if err := execute(context.Background(), p); err != nil {
		t.Errorf("With context, got an error: %v", err)
	}
}

func TestParDo1x1ErrEmit(t *testing.T) {
	noErr := func(v int, emit func(int)) error {
		emit(v/2 + 5)
		return nil
	}

	hasErr := func(v int, emit func(int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(int, func(int)) error
		returnsErr bool
		want       []testutils.PairII
	}{
		{"doFn that does not return an error", noErr, false, goodResult},
		{"doFn that returns an error", hasErr, true, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→26 and 99→5 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
	}
}

func TestParDoCtx1x1ErrEmit(t *testing.T) {
	noErr := func(ctx context.Context, v int, emit func(int)) error {
		if ctx.Err() != nil {
			emit(0)
			return nil
		}
		emit(v/2 + 5)
		return nil
	}
	hasErr := func(ctx context.Context, v int, emit func(int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int, func(int)) error
		returnsErr bool
		want       []testutils.PairII
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, goodResult},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroResult},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroResult},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroResult},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→26 and 99→5 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
	}
}

func TestParDo1x2Emit(t *testing.T) {
	for _, tc := range []struct {
		desc string
		doFn any
		want []testutils.PairICodedKV
	}{
		{"doFn that emits only non-zero inputs",
			func(v int, emit func(int, int)) {
				if v != 0 {
					emit(v*2, v/2+1)
				}
			},
			[]testutils.PairICodedKV{
				{17, pairK84V22},
			}},
		{"doFn that emits each input once",
			func(v int, emit func(int, int)) {
				emit(v*2, v/2+1)
			},
			valuesCodedKV},
		{"doFn that emits once or twice",
			func(v int, emit func(int, int)) {
				emit(v*2, v/2+1)
				if v != 0 {
					v = v / 2
					emit(v*2, v/2+1)
				}
			},
			[]testutils.PairICodedKV{
				{17, pairK84V22},
				{17, pairK42V11},
				{99, pairK0V1},
			}},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDoCtx1x2Emit(t *testing.T) {
	for _, tc := range []struct {
		desc string
		m    func(context.Context) context.Context // function that modifies a context
		want []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context", unchangedContext, valuesCodedKV},
		// various context modifiers that will cancel the context before any work starts
		{"cancel", cancel, zeroValuedCodedKV},
		{"instantDeadline", instantDeadline, zeroValuedCodedKV},
		{"instantTimeout", instantTimeout, zeroValuedCodedKV},
	} {
		doFn := func(ctx context.Context, v int, emit func(int, int)) {
			if ctx.Err() != nil {
				emit(0, 0)
			} else {
				emit(v*2, v/2+1)
			}
		}
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDo1x2ErrEmit(t *testing.T) {
	noErr := func(v int, emit func(int, int)) error {
		emit(v*2, v/2+1)
		return nil
	}
	hasErr := func(v int, emit func(int, int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(int, func(int, int)) error
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		{"doFn that does not return an error", noErr, false, valuesCodedKV},
		{"doFn that returns an error", hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got one: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDoCtx1x2ErrEmit(t *testing.T) {
	noErr := func(ctx context.Context, v int, emit func(int, int)) error {
		if ctx.Err() != nil {
			emit(0, 0)
			return nil
		}
		emit(v*2, v/2+1)
		return nil
	}
	hasErr := func(ctx context.Context, v int, emit func(int, int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int, func(int, int)) error
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, valuesCodedKV},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroValuedCodedKV},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(values, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairToKV, col)

		// pcol should contain 17→42 and 99→0.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDo2x1Emit(t *testing.T) {
	for _, tc := range []struct {
		desc string
		doFn any
		want []testutils.PairII
	}{
		{"doFn that emits only non-zero input k",
			func(k, v int, emit func(int)) {
				if k != 0 {
					emit(k + v)
				}
			},
			[]testutils.PairII{
				{17, 106},
			}},
		{"doFn that emits each input once",
			func(k, v int, emit func(int)) {
				emit(k + v)
			},
			goodResult2x1},
		{"doFn that emits once or twice",
			func(k, v int, emit func(int)) {
				emit(k + v)
				if k != 0 {
					k = k / 2
					v = v / 2
					emit(k + v)
				}
			},
			[]testutils.PairII{
				{17, 106},
				{17, 53},
				{99, 1},
			}},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
	}
}

func TestParDoCtx2x1Emit(t *testing.T) {
	for _, tc := range []struct {
		desc string
		m    func(context.Context) context.Context // function that modifies a context
		want []testutils.PairII
	}{
		// good context
		{"unchanged context", unchangedContext, goodResult2x1},
		// various context modifiers that will cancel the context before any work starts
		{"cancel", cancel, zeroResult},
		{"instantDeadline", instantDeadline, zeroResult},
		{"instantTimeout", instantTimeout, zeroResult},
	} {
		doFn := func(ctx context.Context, k, v int, emit func(int)) {
			if ctx.Err() != nil {
				emit(0)
			} else {
				emit(k + v)
			}
		}
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
	}
}

func TestParDo2x1ErrEmit(t *testing.T) {
	noErr := func(k, v int, emit func(int)) error {
		emit(k + v)
		return nil
	}
	hasErr := func(k, v int, emit func(int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(int, int, func(int)) error
		returnsErr bool
		want       []testutils.PairII
	}{
		{"doFn that does not return an error", noErr, false, goodResult2x1},
		{"doFn that returns an error", hasErr, true, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<codedKV{84, 22}> and 99→<codedKV{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got on: %v", tc.desc, err)
		}
	}
}

func TestParDoCtx2x1ErrEmit(t *testing.T) {
	noErr := func(ctx context.Context, k, v int, emit func(int)) error {
		if ctx.Err() != nil {
			emit(0)
			return nil
		}
		emit(k + v)
		return nil
	}
	hasErr := func(ctx context.Context, k, v int, emit func(int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int, int, func(int)) error
		returnsErr bool
		want       []testutils.PairII
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, goodResult2x1},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroResult},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroResult},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroResult},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroResult},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<codedKV{84, 22}> and 99→<codedKV{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→106 and 99→1 in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPair, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, did not expect an error, but got on: %v", tc.desc, err)
		}
	}
}

func TestParDo2x2Emit(t *testing.T) {
	for _, tc := range []struct {
		desc string
		doFn any
		want []testutils.PairICodedKV
	}{
		{"doFn that emits only non-zero input k",
			func(k, v int, emit func(int, int)) {
				if k != 0 {
					emit(k+v, k-v)
				}
			},
			[]testutils.PairICodedKV{
				{17, pairK106V62},
			}},
		{"doFn that emits each input once",
			func(k, v int, emit func(int, int)) {
				emit(k+v, k-v)
			},
			goodResult2x2},
		{"doFn that emits once or twice",
			func(k, v int, emit func(int, int)) {
				emit(k+v, k-v)
				if k != 0 {
					k = k / 2
					v = v / 2
					emit(k+v, k-v)
				}
			},
			[]testutils.PairICodedKV{
				{17, pairK106V62},
				{17, pairK53V31},
				{99, pairK1VMinus1},
			}},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDoCtx2x2Emit(t *testing.T) {
	for _, tc := range []struct {
		desc string
		m    func(context.Context) context.Context // function that modifies a context
		want []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context", unchangedContext, goodResult2x2},
		// various context modifiers that will cancel the context before any work starts
		{"cancel", cancel, zeroValuedCodedKV},
		{"instantDeadline", instantDeadline, zeroValuedCodedKV},
		{"instantTimeout", instantTimeout, zeroValuedCodedKV},
	} {
		doFn := func(ctx context.Context, k, v int, emit func(int, int)) {
			if ctx.Err() != nil {
				emit(0, 0)
			} else {
				emit(k+v, k-v)
			}
		}
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDo2x2ErrEmit(t *testing.T) {
	noErr := func(k, v int, emit func(int, int)) error {
		emit(k+v, k-v)
		return nil
	}
	hasErr := func(k, v int, emit func(int, int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		doFn       func(int, int, func(int, int)) error
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		{"doFn that does not return an error", noErr, false, goodResult2x2},
		{"doFn that returns an error", hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(context.Background(), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(context.Background(), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

func TestParDoCtx2x2ErrEmit(t *testing.T) {
	noErr := func(ctx context.Context, k, v int, emit func(int, int)) error {
		if ctx.Err() != nil {
			emit(0, 0)
			return nil
		}
		emit(k+v, k-v)
		return nil
	}
	hasErr := func(ctx context.Context, k, v int, emit func(int, int)) error {
		return errors.New("sample error")
	}
	for _, tc := range []struct {
		desc       string
		m          func(context.Context) context.Context // function that modifies a context
		doFn       func(context.Context, int, int, func(int, int)) error
		returnsErr bool
		want       []testutils.PairICodedKV
	}{
		// good context
		{"unchanged context and no error", unchangedContext, noErr, false, goodResult2x2},
		// various context modifiers that will cancel the context before any work starts
		{"cancel and no error", cancel, noErr, false, zeroValuedCodedKV},
		{"instantDeadline and no error", instantDeadline, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and no error", instantTimeout, noErr, false, zeroValuedCodedKV},
		{"instantTimeout and error", instantTimeout, hasErr, true, zeroValuedCodedKV},
	} {
		p, s, col, wantCol := ptest.CreateList2(valuesCodedKV, tc.want)
		wantCodec := kv.NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0)))
		colKV := beam.ParDo(s, testutils.PairICodedKVToKV, col)

		// pcol should contain 17→<kv.Pair{84, 22}> and 99→<kv.Pair{0, 1}>.
		pcol := MakePrivate(s, colKV, privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1}))
		// We change that to 17→<kv.Pair{106, 62}> and 99→<kv.Pair{1, -1}> in the PrivatePCollection
		pcol = ParDo(s, tc.doFn, pcol)
		gotCol := beam.ParDo(s, testutils.KVToPairICodedKV, pcol.col)
		passert.Equals(s, gotCol, wantCol)
		if tc.returnsErr {
			if err := execute(tc.m(context.Background()), p); err == nil {
				t.Errorf("With %s, expected runner to return an error, but didn't get one", tc.desc)
			}
			continue
		}
		if err := execute(tc.m(context.Background()), p); err != nil {
			t.Errorf("With %s, got an error: %v", tc.desc, err)
		}
		if diff := cmp.Diff(pcol.codec, wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("With %s, DoFn returned a PrivatePCollection with wrong codec, diff=%s", tc.desc, diff)
		}
	}
}

// Ensure that valid DoFns built are built correctly
func TestBuildDoFn(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		doFn        any
		wantType    reflect.Type
		wantTypeDef beam.TypeDefinition
		wantCodec   *kv.Codec
	}{
		{"string → int",
			func(x string) int { return len(x) },
			reflect.TypeOf(&generated.TransformFn1x1{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"(context, string) → int",
			func(_ context.Context, x string) int { return len(x) },
			reflect.TypeOf(&generated.TransformFnCtx1x1{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"string → (int, error)",
			func(x string) (int, error) { return len(x), nil },
			reflect.TypeOf(&generated.TransformFn1x1Err{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"(context, string) → (int, error)",
			func(_ context.Context, x string) (int, error) { return len(x), nil },
			reflect.TypeOf(&generated.TransformFnCtx1x1Err{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"string → (string, int)",
			func(x string) (string, int) { return x, len(x) },
			reflect.TypeOf(&generated.TransformFn1x2{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string) → (string, int)",
			func(_ context.Context, x string) (string, int) { return x, len(x) },
			reflect.TypeOf(&generated.TransformFnCtx1x2{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"string→(string, int, error)",
			func(x string) (string, int, error) { return x, len(x), nil },
			reflect.TypeOf(&generated.TransformFn1x2Err{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string) → (string, int, error)",
			func(_ context.Context, x string) (string, int, error) { return x, len(x), nil },
			reflect.TypeOf(&generated.TransformFnCtx1x2Err{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(string, int) → int",
			func(x string, y int) int { return len(x) },
			reflect.TypeOf(&generated.TransformFn2x1{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"(context, string, int) → int",
			func(_ context.Context, x string, y int) int { return len(x) },
			reflect.TypeOf(&generated.TransformFnCtx2x1{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"(string, int) → (int, error)",
			func(x string, y int) (int, error) { return len(x), nil },
			reflect.TypeOf(&generated.TransformFn2x1Err{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"(context, string, int) → (int, error)",
			func(_ context.Context, x string, y int) (int, error) { return len(x), nil },
			reflect.TypeOf(&generated.TransformFnCtx2x1Err{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf(int(0))},
			nil,
		},
		{"(string, int) → (string, int)",
			func(x string, y int) (string, int) { return x, len(x) + y },
			reflect.TypeOf(&generated.TransformFn2x2{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string, int) → (string, int)",
			func(_ context.Context, x string, y int) (string, int) { return x, len(x) + y },
			reflect.TypeOf(&generated.TransformFnCtx2x2{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(string, int) → (string, int, error)",
			func(x string, y int) (string, int, error) { return x, len(x) + y, nil },
			reflect.TypeOf(&generated.TransformFn2x2Err{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string, int) → (string, int, error)",
			func(_ context.Context, x string, y int) (string, int, error) { return x, len(x) + y, nil },
			reflect.TypeOf(&generated.TransformFnCtx2x2Err{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(string, emit(string)) → <no output>",
			func(x string, emit func(string)) { emit(x) },
			reflect.TypeOf(&generated.TransformFn1x1Emit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(context, string, emit(string)) → <no output>",
			func(_ context.Context, x string, emit func(string)) { emit(x) },
			reflect.TypeOf(&generated.TransformFnCtx1x1Emit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(string, emit(string)) → error",
			func(x string, emit func(string)) error {
				emit(x)
				return nil
			},
			reflect.TypeOf(&generated.TransformFn1x1ErrEmit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(context, string, emit(string)) → error",
			func(_ context.Context, x string, emit func(string)) error {
				emit(x)
				return nil
			},
			reflect.TypeOf(&generated.TransformFnCtx1x1ErrEmit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(string, emit(string, int)) → <no output>",
			func(x string, emit func(string, int)) { emit(x, len(x)) },
			reflect.TypeOf(&generated.TransformFn1x2Emit{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string, emit(string, int)) → <no output>",
			func(ctx context.Context, x string, emit func(string, int)) { emit(x, len(x)) },
			reflect.TypeOf(&generated.TransformFnCtx1x2Emit{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(string, emit(string, int)) → error",
			func(x string, emit func(string, int)) error {
				emit(x, len(x))
				return nil
			},
			reflect.TypeOf(&generated.TransformFn1x2ErrEmit{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string, emit(string, int)) → error",
			func(ctx context.Context, x string, emit func(string, int)) error {
				emit(x, len(x))
				return nil
			},
			reflect.TypeOf(&generated.TransformFnCtx1x2ErrEmit{}),
			beam.TypeDefinition{Var: beam.ZType, T: reflect.TypeOf(kv.Pair{})},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(string, int, emit(string)) → <no output>",
			func(x string, y int, emit func(string)) { emit(x) },
			reflect.TypeOf(&generated.TransformFn2x1Emit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(context, string, int, emit(string)) → <no output>",
			func(ctx context.Context, x string, y int, emit func(string)) { emit(x) },
			reflect.TypeOf(&generated.TransformFnCtx2x1Emit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(string, int, emit(string)) → error",
			func(x string, y int, emit func(string)) error {
				emit(x)
				return nil
			},
			reflect.TypeOf(&generated.TransformFn2x1ErrEmit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(context, string, int, emit(string)) → error",
			func(ctx context.Context, x string, y int, emit func(string)) error {
				emit(x)
				return nil
			},
			reflect.TypeOf(&generated.TransformFnCtx2x1ErrEmit{}),
			beam.TypeDefinition{Var: beam.YType, T: reflect.TypeOf("")},
			nil,
		},
		{"(string, int, emit(string, int)) → <no output>",
			func(x string, y int, emit func(string, int)) { emit(x, len(x)) },
			reflect.TypeOf(&generated.TransformFn2x2Emit{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string, int, emit(string, int)) → <no output>",
			func(ctx context.Context, x string, y int, emit func(string, int)) { emit(x, len(x)) },
			reflect.TypeOf(&generated.TransformFnCtx2x2Emit{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(string, int, emit(string, int)) → error",
			func(x string, y int, emit func(string, int)) error {
				emit(x, len(x))
				return nil
			},
			reflect.TypeOf(&generated.TransformFn2x2ErrEmit{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
		{"(context, string, int, emit(string, int)) → error",
			func(ctx context.Context, x string, y int, emit func(string, int)) error {
				emit(x, len(x))
				return nil
			},
			reflect.TypeOf(&generated.TransformFnCtx2x2ErrEmit{}),
			beam.TypeDefinition{},
			kv.NewCodec(reflect.TypeOf(""), reflect.TypeOf(int(0))),
		},
	} {
		got, err := buildDoFn(tc.doFn)
		if err != nil {
			t.Errorf("%s: buildDoFn returned error %v (expected no error)", tc.desc, err)
		}
		if got == nil {
			t.Errorf("%s: buildDoFn returned nil function (wanted function of type %v)", tc.desc, tc.wantType.String())
		}
		if typ := reflect.TypeOf(got.fn); typ != tc.wantType {
			t.Errorf("%s: buildDoFn returned wrong type %v (want %v)", tc.desc, typ.String(), tc.wantType.String())
		}
		if diff := cmp.Diff(got.typeDef, tc.wantTypeDef, cmp.Comparer(compareTypeDefs)); diff != "" {
			t.Errorf("%s: buildDoFn returned wrong beam.TypeDefinition, diff=%s", tc.desc, diff)
		}
		if diff := cmp.Diff(got.codec, tc.wantCodec, cmp.Comparer(compareCodecs)); diff != "" {
			t.Errorf("%s: buildDoFn returned wrong kv.Codec, diff=%s", tc.desc, diff)
		}
	}
}

type testStructuralDoFn struct {
	state int
}

func (fn *testStructuralDoFn) ProcessElement(x int) int {
	return fn.state + x
}

// Ensure that invalid DoFns return an error
func TestInvalidDoFn(t *testing.T) {
	for _, tc := range []struct {
		desc string
		doFn any
	}{
		{"structural doFn", &testStructuralDoFn{1}},
		// bad inputs
		{"(beam.W) → int", func(x beam.W) int { return 0 }},
		{"(string, emit(beam.W)) → <no output>", func(x string, emit func(beam.W)) { emit(x) }},
		{"(string, string, string) → int", func(x, y, z string) int { return len(x) }},
		{"(EventTime, string) → int", func(_ beam.EventTime, x string) int { return len(x) }},
		{"(Window, string) → int", func(_ beam.Window, x string) int { return len(x) }},
		{"(string, emit(string), unWantedEmit(string)) → <no output>", func(x string, emit func(string), unWantedEmit func(string)) { emit(x) }},
		{"(context, int, string, emit(string)) → string", func(_ context.Context, _ int, x string, emit func(string)) string { return x }},
		{"(string, emit(string), unWantedEmit(string)) → <no output>", func(x string, emit func(string), unWantedEmit func(string)) { emit(x) }},
		{"(context, int, string, emit(string)) → string", func(_ context.Context, _ int, x string, emit func(string)) string { return x }},
		// bad outputs
		{"(int) → beam.W", func(x int) beam.W { return x }},
		{"string → (EventTime, int)", func(x string) (beam.EventTime, int) { return 0, len(x) }},
		{"string → (int, int, int)", func(x string) (int, int, int) { return len(x), 0, -1 }},
		{"string → <no output>", func(x string) {}},
		{"(string, emit(string)) → string", func(x string, emit func(string)) string { return x }},
	} {
		// All of these doFns should return an error
		got, err := buildDoFn(tc.doFn)
		if got != nil {
			t.Errorf("%s: buildDoFn returned (non-nil function),%v; expected nil function and error", tc.desc, err)
		}
		if err == nil {
			t.Errorf("%s: buildDoFn returned <nil function>,<nil error>; expected an error", tc.desc)
		}
		// Check that we return a human-readable error, otherwise the
		// tested fn is not a valid DoFn for beam.ParDo in the first place
		if err != nil && strings.Contains(err.Error(), "couldn't create funcx.Fn") {
			t.Errorf("%s: buildDoFn failed with a Beam error: %v", tc.desc, err)
		}
	}
}

func TestValidateArgOrder(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		order []funcx.FnParamKind
		valid bool
	}{
		{
			desc:  "(context)",
			order: []funcx.FnParamKind{funcx.FnContext},
			valid: true,
		}, {
			desc:  "(time)",
			order: []funcx.FnParamKind{funcx.FnEventTime},
			valid: true,
		}, {
			desc:  "(value)",
			order: []funcx.FnParamKind{funcx.FnValue},
			valid: true,
		}, {
			desc:  "(context,time)",
			order: []funcx.FnParamKind{funcx.FnContext, funcx.FnEventTime},
			valid: true,
		}, {
			desc:  "(context, value)",
			order: []funcx.FnParamKind{funcx.FnContext, funcx.FnValue},
			valid: true,
		}, {
			desc:  "(time, context)",
			order: []funcx.FnParamKind{funcx.FnEventTime, funcx.FnContext},
			valid: false,
		}, {
			desc:  "(time, value)",
			order: []funcx.FnParamKind{funcx.FnEventTime, funcx.FnValue},
			valid: true,
		}, {
			desc:  "(value, context)",
			order: []funcx.FnParamKind{funcx.FnValue, funcx.FnContext},
			valid: false,
		}, {
			desc:  "(value, time)",
			order: []funcx.FnParamKind{funcx.FnValue, funcx.FnEventTime},
			valid: false,
		}, {
			desc:  "(context,time,value)",
			order: []funcx.FnParamKind{funcx.FnContext, funcx.FnEventTime, funcx.FnValue},
			valid: true,
		}, {
			desc:  "(context,value,time)",
			order: []funcx.FnParamKind{funcx.FnContext, funcx.FnValue, funcx.FnEventTime},
			valid: false,
		}, {
			desc:  "(time,context,value)",
			order: []funcx.FnParamKind{funcx.FnEventTime, funcx.FnContext, funcx.FnValue},
			valid: false,
		}, {
			desc:  "(time,value,context)",
			order: []funcx.FnParamKind{funcx.FnEventTime, funcx.FnContext, funcx.FnValue},
			valid: false,
		}, {
			desc:  "(value,context,time)",
			order: []funcx.FnParamKind{funcx.FnValue, funcx.FnContext, funcx.FnEventTime},
			valid: false,
		}, {
			desc:  "(value,time,context)",
			order: []funcx.FnParamKind{funcx.FnValue, funcx.FnEventTime, funcx.FnValue},
			valid: false,
		},
		{
			desc:  "(emit, context,value)",
			order: []funcx.FnParamKind{funcx.FnEmit, funcx.FnContext, funcx.FnValue},
			valid: false,
		},
		{
			desc:  "(value, emit)",
			order: []funcx.FnParamKind{funcx.FnValue, funcx.FnEmit},
			valid: true,
		},
		{
			desc:  "(context,value, emit)",
			order: []funcx.FnParamKind{funcx.FnContext, funcx.FnValue, funcx.FnEmit},
			valid: true,
		},
	} {
		var fn funcx.Fn
		for _, kind := range tc.order {
			fn.Param = append(fn.Param, funcx.FnParam{Kind: kind})
		}
		err := validateArgOrder(&fn)
		if tc.valid {
			if err != nil {
				t.Errorf("Test case %s should have been valid, but got error %v", tc.desc, err)
			}
		} else if err == nil {
			t.Errorf("Test case %s should not have been valid, but no error was returned", tc.desc)
		}
	}
}

func TestValidateReturnOrder(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		order []funcx.ReturnKind
		valid bool
	}{
		{
			desc:  "(time)",
			order: []funcx.ReturnKind{funcx.RetEventTime},
			valid: true,
		}, {
			desc:  "(value)",
			order: []funcx.ReturnKind{funcx.RetValue},
			valid: true,
		}, {
			desc:  "(error)",
			order: []funcx.ReturnKind{funcx.RetError},
			valid: true,
		}, {
			desc:  "(time,value)",
			order: []funcx.ReturnKind{funcx.RetEventTime, funcx.RetValue},
			valid: true,
		}, {
			desc:  "(time,error)",
			order: []funcx.ReturnKind{funcx.RetEventTime, funcx.RetError},
			valid: true,
		}, {
			desc:  "(value,time)",
			order: []funcx.ReturnKind{funcx.RetValue, funcx.RetEventTime},
			valid: false,
		}, {
			desc:  "(value,error)",
			order: []funcx.ReturnKind{funcx.RetValue, funcx.RetError},
			valid: true,
		}, {
			desc:  "(error,time)",
			order: []funcx.ReturnKind{funcx.RetError, funcx.RetEventTime},
			valid: false,
		}, {
			desc:  "(error,value)",
			order: []funcx.ReturnKind{funcx.RetError, funcx.RetValue},
			valid: false,
		}, {
			desc:  "(time,value,error)",
			order: []funcx.ReturnKind{funcx.RetEventTime, funcx.RetValue, funcx.RetError},
			valid: true,
		}, {
			desc:  "(time,error,value)",
			order: []funcx.ReturnKind{funcx.RetEventTime, funcx.RetError, funcx.RetValue},
			valid: false,
		}, {
			desc:  "(value, time, error)",
			order: []funcx.ReturnKind{funcx.RetValue, funcx.RetEventTime, funcx.RetError},
			valid: false,
		}, {
			desc:  "(value,error,time)",
			order: []funcx.ReturnKind{funcx.RetValue, funcx.RetError, funcx.RetEventTime},
			valid: false,
		}, {
			desc:  "(error,time,value)",
			order: []funcx.ReturnKind{funcx.RetError, funcx.RetEventTime, funcx.RetValue},
			valid: false,
		}, {
			desc:  "(error,value,time)",
			order: []funcx.ReturnKind{funcx.RetError, funcx.RetValue, funcx.RetEventTime},
			valid: false,
		},
	} {
		t.Run(fmt.Sprintf("function with return values %s", tc.desc), func(t *testing.T) {
			var fn funcx.Fn
			for _, kind := range tc.order {
				fn.Ret = append(fn.Ret, funcx.ReturnParam{Kind: kind})
			}
			err := validateRetOrder(&fn)
			if tc.valid != (err == nil) {
				t.Errorf("Testcase %s returned error %v, valid: %v", tc.desc, err, tc.valid)
			}
		})
	}
}
