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

// Package generated includes transformations implemented via code generation.
package generated

import (
	"context"
	"reflect"

	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/core/util/reflectx"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*TransformFn1x1)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x1)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x1Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x1Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x2)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x2)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x2Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x2Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x1)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x1)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x1Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x1Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x2)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x2)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x2Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x2Err)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x1Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x1Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x1ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x1ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x2Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x2Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn1x2ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx1x2ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x1Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x1Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x1ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x1ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x2Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x2Emit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFn2x2ErrEmit)(nil)))
	beam.RegisterType(reflect.TypeOf((*TransformFnCtx2x2ErrEmit)(nil)))
}

type TransformFn1x1 struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func1x1
}

func (fn *TransformFn1x1) Setup() {
	fn.tfn = reflectx.ToFunc1x1(fn.Transform.Fn)
}

func (fn *TransformFn1x1) ProcessElement(id beam.W, v beam.X) (beam.W, beam.Y) {
	out := fn.tfn.Call1x1(v)
	return id, out
}

type TransformFnCtx1x1 struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func2x1
}

func (fn *TransformFnCtx1x1) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
}

func (fn *TransformFnCtx1x1) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, beam.Y) {
	out := fn.tfn.Call2x1(ctx, v)
	return id, out
}

type TransformFn1x1Err struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func1x2
}

func (fn *TransformFn1x1Err) Setup() {
	fn.tfn = reflectx.ToFunc1x2(fn.Transform.Fn)
}

func (fn *TransformFn1x1Err) ProcessElement(id beam.W, v beam.X) (beam.W, beam.Y, error) {
	out, err := fn.tfn.Call1x2(v)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFnCtx1x1Err struct {
	Transform beam.EncodedFunc
	tfn       reflectx.Func2x2
}

func (fn *TransformFnCtx1x1Err) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
}

func (fn *TransformFnCtx1x1Err) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, beam.Y, error) {
	out, err := fn.tfn.Call2x2(ctx, v)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFn1x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func1x2
	OutputCodec *kv.Codec
}

func (fn *TransformFn1x2) Setup() {
	fn.tfn = reflectx.ToFunc1x2(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFn1x2) ProcessElement(id beam.W, v beam.X) (beam.W, beam.Z) {
	outputK, outputV := fn.tfn.Call1x2(v)
	out := fn.OutputCodec.Encode(outputK, outputV)

	return id, out
}

type TransformFnCtx1x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x2
	OutputCodec *kv.Codec
}

func (fn *TransformFnCtx1x2) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx1x2) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, beam.Z) {
	outputK, outputV := fn.tfn.Call2x2(ctx, v)
	out := fn.OutputCodec.Encode(outputK, outputV)

	return id, out
}

type TransformFn1x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func1x3
	OutputCodec *kv.Codec
}

func (fn *TransformFn1x2Err) Setup() {
	fn.tfn = reflectx.ToFunc1x3(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFn1x2Err) ProcessElement(id beam.W, v beam.X) (beam.W, beam.Z, error) {
	outputK, outputV, err := fn.tfn.Call1x3(v)
	out := fn.OutputCodec.Encode(outputK, outputV)

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFnCtx1x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x3
	OutputCodec *kv.Codec
}

func (fn *TransformFnCtx1x2Err) Setup() {
	fn.tfn = reflectx.ToFunc2x3(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx1x2Err) ProcessElement(ctx context.Context, id beam.W, v beam.X) (beam.W, beam.Z, error) {
	outputK, outputV, err := fn.tfn.Call2x3(ctx, v)
	out := fn.OutputCodec.Encode(outputK, outputV)

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFn2x1 struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x1
	InputCodec *kv.Codec
}

func (fn *TransformFn2x1) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFn2x1) ProcessElement(id beam.W, kvp beam.Z) (beam.W, beam.Y) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	out := fn.tfn.Call2x1(inputK, inputV)
	return id, out
}

type TransformFnCtx2x1 struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x1
	InputCodec *kv.Codec
}

func (fn *TransformFnCtx2x1) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFnCtx2x1) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z) (beam.W, beam.Y) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	out := fn.tfn.Call3x1(ctx, inputK, inputV)
	return id, out
}

type TransformFn2x1Err struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x2
	InputCodec *kv.Codec
}

func (fn *TransformFn2x1Err) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFn2x1Err) ProcessElement(id beam.W, kvp beam.Z) (beam.W, beam.Y, error) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	out, err := fn.tfn.Call2x2(inputK, inputV)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFnCtx2x1Err struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x2
	InputCodec *kv.Codec
}

func (fn *TransformFnCtx2x1Err) Setup() {
	fn.tfn = reflectx.ToFunc3x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFnCtx2x1Err) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z) (beam.W, beam.Y, error) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	out, err := fn.tfn.Call3x2(ctx, inputK, inputV)
	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFn2x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x2
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

func (fn *TransformFn2x2) Setup() {
	fn.tfn = reflectx.ToFunc2x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFn2x2) ProcessElement(id beam.W, kvp beam.Z) (beam.W, beam.Z) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	outputK, outputV := fn.tfn.Call2x2(inputK, inputV)
	out := fn.OutputCodec.Encode(outputK, outputV)

	return id, out
}

type TransformFnCtx2x2 struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x2
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

func (fn *TransformFnCtx2x2) Setup() {
	fn.tfn = reflectx.ToFunc3x2(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx2x2) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z) (beam.W, beam.Z) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	outputK, outputV := fn.tfn.Call3x2(ctx, inputK, inputV)
	out := fn.OutputCodec.Encode(outputK, outputV)

	return id, out
}

type TransformFn2x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x3
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

func (fn *TransformFn2x2Err) Setup() {
	fn.tfn = reflectx.ToFunc2x3(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFn2x2Err) ProcessElement(id beam.W, kvp beam.Z) (beam.W, beam.Z, error) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	outputK, outputV, err := fn.tfn.Call2x3(inputK, inputV)
	out := fn.OutputCodec.Encode(outputK, outputV)

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFnCtx2x2Err struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x3
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
}

func (fn *TransformFnCtx2x2Err) Setup() {
	fn.tfn = reflectx.ToFunc3x3(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx2x2Err) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z) (beam.W, beam.Z, error) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	outputK, outputV, err := fn.tfn.Call3x3(ctx, inputK, inputV)
	out := fn.OutputCodec.Encode(outputK, outputV)

	var errOut error
	if err != nil {
		errOut = err.(error)
	}

	return id, out, errOut
}

type TransformFn1x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x0
	EmitFnType beam.EncodedType
}

func (fn *TransformFn1x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc2x0(fn.Transform.Fn)
}

func (fn *TransformFn1x1Emit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, beam.Y)) {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call2x0(v, rmef.Interface())
}

type TransformFnCtx1x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x0
	EmitFnType beam.EncodedType
}

func (fn *TransformFnCtx1x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
}

func (fn *TransformFnCtx1x1Emit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, beam.Y)) {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(ctx, v, rmef.Interface())
}

type TransformFn1x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func2x1
	EmitFnType beam.EncodedType
}

func (fn *TransformFn1x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
}

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

type TransformFnCtx1x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x1
	EmitFnType beam.EncodedType
}

func (fn *TransformFnCtx1x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
}

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

type TransformFn1x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x0
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFn1x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc2x0(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFn1x2Emit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, beam.Z)) {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call2x0(v, rmef.Interface())
}

type TransformFnCtx1x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x0
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFnCtx1x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx1x2Emit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, beam.Z)) {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(ctx, v, rmef.Interface())
}

type TransformFn1x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func2x1
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFn1x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc2x1(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFn1x2ErrEmit) ProcessElement(id beam.W, v beam.X, emit func(beam.W, beam.Z)) error {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
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

type TransformFnCtx1x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x1
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFnCtx1x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx1x2ErrEmit) ProcessElement(ctx context.Context, id beam.W, v beam.X, emit func(beam.W, beam.Z)) error {
	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
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

type TransformFn2x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x0
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

func (fn *TransformFn2x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFn2x1Emit) ProcessElement(id beam.W, kvp beam.Z, emit func(beam.W, beam.Y)) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(inputK, inputV, rmef.Interface())
}

type TransformFnCtx2x1Emit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func4x0
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

func (fn *TransformFnCtx2x1Emit) Setup() {
	fn.tfn = reflectx.ToFunc4x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFnCtx2x1Emit) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z, emit func(beam.W, beam.Y)) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	internalEmit := func(y []reflect.Value) []reflect.Value {
		emit(id, y[0].Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call4x0(ctx, inputK, inputV, rmef.Interface())
}

type TransformFn2x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func3x1
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

func (fn *TransformFn2x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFn2x1ErrEmit) ProcessElement(id beam.W, kvp beam.Z, emit func(beam.W, beam.Y)) error {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

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

type TransformFnCtx2x1ErrEmit struct {
	Transform  beam.EncodedFunc
	tfn        reflectx.Func4x1
	InputCodec *kv.Codec
	EmitFnType beam.EncodedType
}

func (fn *TransformFnCtx2x1ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc4x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
}

func (fn *TransformFnCtx2x1ErrEmit) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z, emit func(beam.W, beam.Y)) error {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

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

type TransformFn2x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x0
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFn2x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc3x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFn2x2Emit) ProcessElement(id beam.W, kvp beam.Z, emit func(beam.W, beam.Z)) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call3x0(inputK, inputV, rmef.Interface())
}

type TransformFnCtx2x2Emit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func4x0
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFnCtx2x2Emit) Setup() {
	fn.tfn = reflectx.ToFunc4x0(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx2x2Emit) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z, emit func(beam.W, beam.Z)) {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
		return nil
	}
	rmef := reflect.MakeFunc(fn.EmitFnType.T, internalEmit)
	fn.tfn.Call4x0(ctx, inputK, inputV, rmef.Interface())
}

type TransformFn2x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func3x1
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFn2x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc3x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFn2x2ErrEmit) ProcessElement(id beam.W, kvp beam.Z, emit func(beam.W, beam.Z)) error {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
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

type TransformFnCtx2x2ErrEmit struct {
	Transform   beam.EncodedFunc
	tfn         reflectx.Func4x1
	InputCodec  *kv.Codec
	OutputCodec *kv.Codec
	EmitFnType  beam.EncodedType
}

func (fn *TransformFnCtx2x2ErrEmit) Setup() {
	fn.tfn = reflectx.ToFunc4x1(fn.Transform.Fn)
	fn.InputCodec.Setup()
	fn.OutputCodec.Setup()
}

func (fn *TransformFnCtx2x2ErrEmit) ProcessElement(ctx context.Context, id beam.W, kvp beam.Z, emit func(beam.W, beam.Z)) error {
	inputKV := kvp.(kv.Pair)
	inputK, inputV := fn.InputCodec.Decode(inputKV)

	internalEmit := func(y []reflect.Value) []reflect.Value {
		outputKV := fn.OutputCodec.Encode(y[0].Interface(), y[1].Interface())
		emit(id, reflect.ValueOf(outputKV).Interface())
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
