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

// Package generated was generated automatically.
// Do not edit manually.
package generated

import (
	"context"
	"reflect"

	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/util/reflectx"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
)

func init() {
	register.DoFn2x2[beam.W, beam.X, beam.W, beam.Y](&TransformFn1x1{})
	register.DoFn3x2[context.Context, beam.W, beam.X, beam.W, beam.Y](&TransformFnCtx1x1{})
	register.DoFn2x3[beam.W, beam.X, beam.W, beam.Y, error](&TransformFn1x1Err{})
	register.DoFn3x3[context.Context, beam.W, beam.X, beam.W, beam.Y, error](&TransformFnCtx1x1Err{})
	register.DoFn2x3[beam.W, beam.X, beam.W, kv.Pair, error](&TransformFn1x2{})
	register.DoFn3x3[context.Context, beam.W, beam.X, beam.W, kv.Pair, error](&TransformFnCtx1x2{})
	register.DoFn2x3[beam.W, beam.X, beam.W, kv.Pair, error](&TransformFn1x2Err{})
	register.DoFn3x3[context.Context, beam.W, beam.X, beam.W, kv.Pair, error](&TransformFnCtx1x2Err{})
	register.DoFn2x3[beam.W, kv.Pair, beam.W, beam.Y, error](&TransformFn2x1{})
	register.DoFn3x3[context.Context, beam.W, kv.Pair, beam.W, beam.Y, error](&TransformFnCtx2x1{})
	register.DoFn2x3[beam.W, kv.Pair, beam.W, beam.Y, error](&TransformFn2x1Err{})
	register.DoFn3x3[context.Context, beam.W, kv.Pair, beam.W, beam.Y, error](&TransformFnCtx2x1Err{})
	register.DoFn2x3[beam.W, kv.Pair, beam.W, kv.Pair, error](&TransformFn2x2{})
	register.DoFn3x3[context.Context, beam.W, kv.Pair, beam.W, kv.Pair, error](&TransformFnCtx2x2{})
	register.DoFn2x3[beam.W, kv.Pair, beam.W, kv.Pair, error](&TransformFn2x2Err{})
	register.DoFn3x3[context.Context, beam.W, kv.Pair, beam.W, kv.Pair, error](&TransformFnCtx2x2Err{})
	register.DoFn3x0[beam.W, beam.X, func(beam.W, beam.Y)](&TransformFn1x1Emit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn4x0[context.Context, beam.W, beam.X, func(beam.W, beam.Y)](&TransformFnCtx1x1Emit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn3x1[beam.W, beam.X, func(beam.W, beam.Y), error](&TransformFn1x1ErrEmit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn4x1[context.Context, beam.W, beam.X, func(beam.W, beam.Y), error](&TransformFnCtx1x1ErrEmit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn3x1[beam.W, beam.X, func(beam.W, kv.Pair), error](&TransformFn1x2Emit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn4x1[context.Context, beam.W, beam.X, func(beam.W, kv.Pair), error](&TransformFnCtx1x2Emit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn3x1[beam.W, beam.X, func(beam.W, kv.Pair), error](&TransformFn1x2ErrEmit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn4x1[context.Context, beam.W, beam.X, func(beam.W, kv.Pair), error](&TransformFnCtx1x2ErrEmit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn3x1[beam.W, kv.Pair, func(beam.W, beam.Y), error](&TransformFn2x1Emit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn4x1[context.Context, beam.W, kv.Pair, func(beam.W, beam.Y), error](&TransformFnCtx2x1Emit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn3x1[beam.W, kv.Pair, func(beam.W, beam.Y), error](&TransformFn2x1ErrEmit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn4x1[context.Context, beam.W, kv.Pair, func(beam.W, beam.Y), error](&TransformFnCtx2x1ErrEmit{})
	register.Emitter2[beam.W, beam.Y]()
	register.DoFn3x1[beam.W, kv.Pair, func(beam.W, kv.Pair), error](&TransformFn2x2Emit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn4x1[context.Context, beam.W, kv.Pair, func(beam.W, kv.Pair), error](&TransformFnCtx2x2Emit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn3x1[beam.W, kv.Pair, func(beam.W, kv.Pair), error](&TransformFn2x2ErrEmit{})
	register.Emitter2[beam.W, kv.Pair]()
	register.DoFn4x1[context.Context, beam.W, kv.Pair, func(beam.W, kv.Pair), error](&TransformFnCtx2x2ErrEmit{})
	register.Emitter2[beam.W, kv.Pair]()
}

// TransformFn1x1 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: false
type TransformFn1x1 struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func1x1
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x1) Setup() {
	fn.tfn = reflectx.ToFunc1x1(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x1) ProcessElement(id beam.W, v beam.X) (beam.W, beam.Y) {
	out := fn.tfn.Call1x1(v)
	return id, out
}

// TransformFnCtx1x1 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: false
type TransformFnCtx1x1 struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func2x1
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x1) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x1) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, beam.Y) {
	out := fn.tfn.Call2x1(ctx, v)
	return id, out
}

// TransformFn1x1Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: true
type TransformFn1x1Err struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func1x2
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x1Err) Setup() {
	fn.tfn = reflectx.ToFunc1x2(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x1Err) ProcessElement(id beam.W, v beam.X) (beam.W, beam.Y, error) {
	out, err := fn.tfn.Call1x2(v)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFnCtx1x1Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: true
type TransformFnCtx1x1Err struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func2x2
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x1Err) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x1Err) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, beam.Y, error) {
	out, err := fn.tfn.Call2x2(ctx, v)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFn1x2 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: false
type TransformFn1x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func1x2
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x2) Setup() {
	fn.tfn = reflectx.ToFunc1x2(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x2) ProcessElement(id beam.W, v beam.X) (beam.W, kv.Pair, error) {
	outputK, outputV := fn.tfn.Call1x2(v)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	return id, out, nil
}

// TransformFnCtx1x2 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: false
type TransformFnCtx1x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x2
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x2) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x2) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, kv.Pair, error) {
	outputK, outputV := fn.tfn.Call2x2(ctx, v)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	return id, out, nil
}

// TransformFn1x2Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: true
type TransformFn1x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func1x3
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x2Err) Setup() {
	fn.tfn = reflectx.ToFunc1x3(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x2Err) ProcessElement(id beam.W, v beam.X) (beam.W, kv.Pair, error) {
	outputK, outputV, err := fn.tfn.Call1x3(v)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFnCtx1x2Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: true
type TransformFnCtx1x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x3
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x2Err) Setup() {
	fn.tfn = reflectx.ToFunc2x3(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x2Err) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, kv.Pair, error) {
	outputK, outputV, err := fn.tfn.Call2x3(ctx, v)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFn2x1 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: false
type TransformFn2x1 struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x1
	InputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x1) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x1) ProcessElement(id beam.W, inputKV kv.Pair) (beam.W, beam.Y, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, nil, decodeErr
	}

	out := fn.tfn.Call2x1(inputK, inputV)
	return id, out, nil
}

// TransformFnCtx2x1 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: false
type TransformFnCtx2x1 struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x1
	InputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x1) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x1) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair) (beam.W, beam.Y, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, nil, decodeErr
	}

	out := fn.tfn.Call3x1(ctx, inputK, inputV)
	return id, out, nil
}

// TransformFn2x1Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: true
type TransformFn2x1Err struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x2
	InputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x1Err) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x1Err) ProcessElement(id beam.W, inputKV kv.Pair) (beam.W, beam.Y, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, nil, decodeErr
	}

	out, err := fn.tfn.Call2x2(inputK, inputV)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFnCtx2x1Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: true
type TransformFnCtx2x1Err struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x2
	InputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x1Err) Setup() {
	fn.tfn = reflectx.ToFunc3x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x1Err) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair) (beam.W, beam.Y, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, nil, decodeErr
	}

	out, err := fn.tfn.Call3x2(ctx, inputK, inputV)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFn2x2 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: false
type TransformFn2x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x2
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x2) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x2) ProcessElement(id beam.W, inputKV kv.Pair) (beam.W, kv.Pair, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, kv.Pair{}, decodeErr
	}

	outputK, outputV := fn.tfn.Call2x2(inputK, inputV)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	return id, out, nil
}

// TransformFnCtx2x2 is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: false
type TransformFnCtx2x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x2
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x2) Setup() {
	fn.tfn = reflectx.ToFunc3x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x2) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair) (beam.W, kv.Pair, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, kv.Pair{}, decodeErr
	}

	outputK, outputV := fn.tfn.Call3x2(ctx, inputK, inputV)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	return id, out, nil
}

// TransformFn2x2Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: false
//	HasErrOutput: true
type TransformFn2x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x3
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x2Err) Setup() {
	fn.tfn = reflectx.ToFunc2x3(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x2Err) ProcessElement(id beam.W, inputKV kv.Pair) (beam.W, kv.Pair, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, kv.Pair{}, decodeErr
	}

	outputK, outputV, err := fn.tfn.Call2x3(inputK, inputV)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFnCtx2x2Err is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: false
//	HasErrOutput: true
type TransformFnCtx2x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x3
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x2Err) Setup() {
	fn.tfn = reflectx.ToFunc3x3(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x2Err) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair) (beam.W, kv.Pair, error) {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return nil, kv.Pair{}, decodeErr
	}

	outputK, outputV, err := fn.tfn.Call3x3(ctx, inputK, inputV)
	out, encodeErr := fn.OutputCodec.Encode(outputK, outputV)
	if encodeErr != nil {
		return nil, kv.Pair{}, encodeErr
	}

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

// TransformFn1x1Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: false
type TransformFn1x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x0
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc2x0(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x1Emit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, beam.Y)) {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call2x0(v, rmef.Interface())
}

// TransformFnCtx1x1Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: false
type TransformFnCtx1x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x0
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x1Emit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, beam.Y)) {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(ctx, v, rmef.Interface())
}

// TransformFn1x1ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: true
type TransformFn1x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x1
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x1ErrEmit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, beam.Y)) error {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call2x1(v, rmef.Interface())
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFnCtx1x1ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: true
type TransformFnCtx1x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x1
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x1ErrEmit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, beam.Y)) error {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call3x1(ctx, v, rmef.Interface())
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFn1x2Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: false
type TransformFn1x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x0
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc2x0(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x2Emit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, kv.Pair)) error {
	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call2x0(v, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	return nil
}

// TransformFnCtx1x2Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: false
type TransformFnCtx1x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x0
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x2Emit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, kv.Pair)) error {
	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(ctx, v, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	return nil
}

// TransformFn1x2ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: true
type TransformFn1x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x1
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn1x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn1x2ErrEmit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, kv.Pair)) error {
	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call2x1(v, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFnCtx1x2ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: false
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: true
type TransformFnCtx1x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x1
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx1x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx1x2ErrEmit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, kv.Pair)) error {
	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call3x1(ctx, v, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFn2x1Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: false
type TransformFn2x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x0
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x1Emit) ProcessElement(id beam.W, inputKV kv.Pair, emit func(beam.W, beam.Y)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(inputK, inputV, rmef.Interface())
	return nil
}

// TransformFnCtx2x1Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: false
type TransformFnCtx2x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func4x0
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc4x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x1Emit) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair, emit func(beam.W, beam.Y)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call4x0(ctx, inputK, inputV, rmef.Interface())
	return nil
}

// TransformFn2x1ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: true
type TransformFn2x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x1
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x1ErrEmit) ProcessElement(id beam.W, inputKV kv.Pair, emit func(beam.W, beam.Y)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call3x1(inputK, inputV, rmef.Interface())
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFnCtx2x1ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: false
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: true
type TransformFnCtx2x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func4x1
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc4x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x1ErrEmit) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair, emit func(beam.W, beam.Y)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call4x1(ctx, inputK, inputV, rmef.Interface())
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFn2x2Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: false
type TransformFn2x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x0
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x2Emit) ProcessElement(id beam.W, inputKV kv.Pair, emit func(beam.W, kv.Pair)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(inputK, inputV, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	return nil
}

// TransformFnCtx2x2Emit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: false
type TransformFnCtx2x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func4x0
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc4x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x2Emit) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair, emit func(beam.W, kv.Pair)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call4x0(ctx, inputK, inputV, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	return nil
}

// TransformFn2x2ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: false
//	HasEmit: true
//	HasErrOutput: true
type TransformFn2x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x1
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFn2x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFn2x2ErrEmit) ProcessElement(id beam.W, inputKV kv.Pair, emit func(beam.W, kv.Pair)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call3x1(inputK, inputV, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}

// TransformFnCtx2x2ErrEmit is a wrapper for DoFns of the following type on a PrivatePCollection:
//
//	HasKVInput: true
//	HasKVOutput: true
//	HasCtx: true
//	HasEmit: true
//	HasErrOutput: true
type TransformFnCtx2x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func4x1
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

// Setup initializes the TransformFn.
func (fn *TransformFnCtx2x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc4x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

// ProcessElement runs the wrapped DoFn.
func (fn *TransformFnCtx2x2ErrEmit) ProcessElement(ctx context.Context, id beam.W, inputKV kv.Pair, emit func(beam.W, kv.Pair)) error {
	inputK, inputV, decodeErr := fn.InputCodec.Decode(inputKV)
	if decodeErr != nil {
		return decodeErr
	}

	var emitErr error
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV, encodeErr := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		if encodeErr != nil {
			emitErr = encodeErr
		}
		emit(id, outputKV)
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	err := fn.tfn.Call4x1(ctx, inputK, inputV, rmef.Interface())
	if emitErr != nil {
		return emitErr
	}
	var errOut error
	if err != nil {
		errOut = err.(error)
	}
	return errOut
}
