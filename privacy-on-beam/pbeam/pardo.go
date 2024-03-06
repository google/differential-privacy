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
	"fmt"
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/generated"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/funcx"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/util/reflectx"
)

// ParDo applies the given function to all records, propagating privacy
// identifiers. For now, it only works if doFn is a function that has one of
// the following types.
//
//	Transforms a PrivatePCollection<X> into a PrivatePCollection<Y>:
//		- func(X) Y
//		- func(context.Context, X) Y
//		- func(X) (Y, error)
//		- func(context.Context, X) (Y, error)
//		- func(X, emit), where emit has type func(Y)
//		- func(context.Context, X, emit), where emit has type func(Y)
//		- func(X, emit) error, where emit has type func(Y)
//		- func(context.Context, X, emit) error, where emit has type func(Y)
//
//	Transforms a PrivatePCollection<X> into a PrivatePCollection<Y,Z>:
//		- func(X) (Y, Z)
//		- func(context.Context, X) (Y, Z)
//		- func(X) (Y, Z, error)
//		- func(context.Context, X) (Y, Z, error)
//		- func(X, emit), where emit has type func(Y, Z)
//		- func(context.Context, X, emit), where emit has type func(Y, Z)
//		- func(X, emit) error, where emit has type func(Y, Z)
//		- func(context.Context, X, emit) error, where emit has type func(Y, Z)
//
//	Transforms a PrivatePCollection<W,X> into a PrivatePCollection<Y>:
//		- func(W, X) Y
//		- func(context.Context, W, X) Y
//		- func(W, X) (Y, error)
//		- func(context.Context, W, X) (Y, error)
//		- func(W, X, emit), where emit has type func(Y)
//		- func(context.Context, W, X, emit), where emit has type func(Y)
//		- func(W, X, emit) error, where emit has type func(Y)
//		- func(context.Context, W, X, emit) error, where emit has type func(Y)
//
//	Transforms a PrivatePCollection<W,X> into a PrivatePCollection<Y,Z>:
//		- func(W, X) (Y, Z)
//		- func(context.Context, W, X) (Y, Z)
//		- func(W, X) (Y, Z, error)
//		- func(context.Context, W, X) (Y, Z, error)
//		- func(W, X, emit), where emit has type func(Y, Z)
//		- func(context.Context, W, X, emit), where emit has type func(Y, Z)
//		- func(W, X, emit) error, where emit has type func(Y, Z)
//		- func(context.Context error, W, X, emit), where emit has type func(Y, Z)
//
// Note that Beam universal types (e.g., beam.V, beam.T, etc.) are not supported:
// each of the X, Y, Z, W above needs to be a concrete type.
func ParDo(s beam.Scope, doFn any, pcol PrivatePCollection) PrivatePCollection {
	s = s.Scope("pbeam.ParDo")
	// Convert the doFn into a anonDoFn.
	anonDoFn, err := buildDoFn(doFn)
	if err != nil {
		log.Fatalf("Couldn't initialize doFn in pbeam.ParDo: %v", err)
	}
	emptyDef := beam.TypeDefinition{}
	if anonDoFn.typeDef != emptyDef {
		return PrivatePCollection{
			col:         beam.ParDo(s, anonDoFn.fn, pcol.col, anonDoFn.typeDef),
			codec:       anonDoFn.codec,
			privacySpec: pcol.privacySpec,
		}
	}
	return PrivatePCollection{
		col:         beam.ParDo(s, anonDoFn.fn, pcol.col),
		codec:       anonDoFn.codec,
		privacySpec: pcol.privacySpec,
	}
}

// transform encodes the parameters/outputs of a transform function.
type transform struct {
	hasEmit      bool // whether the function has Emitter functions
	hasKVInput   bool // whether the function has a KV input pair (as opposed to a single input)
	hasKVOutput  bool // whether the function has a KV output pair (as opposed to a single output)
	hasErrOutput bool // whether the function has an "error" type output
	hasCtxInput  bool // whether the function has a context.Context input
}

// anonDoFn contains the transformed doFn that is passed to Beam, as well as metadata.
type anonDoFn struct {
	fn      any                 // the transformed doFn passed to Beam
	typeDef beam.TypeDefinition // the type definition necessary for Beam to process fn
	codec   *kv.Codec           // if fn outputs a KV pair, the codec that can decode this pair
}

// buildDoFn validates the provided doFn and transforms it into an *anonDoFn.
func buildDoFn(doFn any) (*anonDoFn, error) {
	if reflect.TypeOf(doFn).Kind() != reflect.Func {
		return nil, fmt.Errorf("pbeam.ParDo doesn't support structural DoFns for now: doFn must be a function")
	}
	err := checkUniversalTypes(reflect.TypeOf(doFn))
	if err != nil {
		return nil, err
	}

	reflectxFn := reflectx.MakeFunc(doFn)
	funcxFn, err := funcx.New(reflectxFn)
	if err != nil {
		return nil, fmt.Errorf("couldn't create funcx.Fn from doFn: %v", err)
	}
	if len(funcxFn.Params(funcx.FnIter|funcx.FnReIter)) > 0 {
		return nil, fmt.Errorf("pbeam.ParDo doesn't support DoFns with side inputs")
	}
	if len(funcxFn.Params(funcx.FnEventTime|funcx.FnWindow)) > 0 {
		return nil, fmt.Errorf("pbeam.PrivatePCollection don't support streaming mode, so DoFns with EventTime or Window arguments are forbidden")
	}
	if len(funcxFn.Params(funcx.FnIllegal|funcx.FnType)) > 0 {
		return nil, fmt.Errorf("illegal DoFn argument in pbeam.ParDo")
	}
	if len(funcxFn.Params(funcx.FnValue)) != 1 && len(funcxFn.Params(funcx.FnValue)) != 2 {
		return nil, fmt.Errorf("DoFn should have one or two value argument")
	}
	if len(funcxFn.Returns(funcx.RetEventTime)) > 0 {
		return nil, fmt.Errorf("pbeam.PrivatePCollection don't support streaming mode, so DoFns who return EventTime are forbidden")
	}
	if len(funcxFn.Returns(funcx.RetIllegal)) > 0 {
		return nil, fmt.Errorf("illegal DoFn return parameter in pbeam.ParDo")
	}
	if len(funcxFn.Params(funcx.FnEmit)) <= 0 && len(funcxFn.Returns(funcx.RetValue)) != 1 && len(funcxFn.Returns(funcx.RetValue)) != 2 {
		return nil, fmt.Errorf("DoFn should have one or two value outputs or has an emit function")
	}
	if err := validateArgOrder(funcxFn); err != nil {
		return nil, err
	}
	if err := validateRetOrder(funcxFn); err != nil {
		return nil, err
	}
	if len(funcxFn.Ret) > 3 {
		return nil, fmt.Errorf("DoFn has too many return values (should be one or two values, optionally followed by an error)")
	}
	t := transform{
		hasEmit:      len(funcxFn.Params(funcx.FnEmit)) > 0,
		hasKVInput:   len(funcxFn.Params(funcx.FnValue)) == 2,
		hasKVOutput:  len(funcxFn.Returns(funcx.RetValue)) == 2,
		hasErrOutput: len(funcxFn.Returns(funcx.RetError)) == 1,
		hasCtxInput:  len(funcxFn.Params(funcx.FnContext)) == 1,
	}
	if t.hasEmit {
		t.hasKVOutput = getEmitFn(reflectxFn).NumIn() == 2 // an emit function with two "inputs" constitutes a <K,V> output.
		return buildEmitDoFn(reflectxFn, t)
	}
	return buildFunctionalDoFn(reflectxFn, t)
}

func checkUniversalTypes(t reflect.Type) error {
	universalTypes := [7]reflect.Type{
		reflect.TypeOf((*beam.T)(nil)).Elem(),
		reflect.TypeOf((*beam.U)(nil)).Elem(),
		reflect.TypeOf((*beam.V)(nil)).Elem(),
		reflect.TypeOf((*beam.W)(nil)).Elem(),
		reflect.TypeOf((*beam.X)(nil)).Elem(),
		reflect.TypeOf((*beam.Y)(nil)).Elem(),
		reflect.TypeOf((*beam.Z)(nil)).Elem(),
	}
	for i := 0; i < t.NumIn(); i++ {
		if t.In(i).Kind() == reflect.Func {
			if err := checkUniversalTypes(t.In(i)); err != nil {
				return err
			}
			continue
		}
		for j := 0; j < len(universalTypes); j++ {
			if t.In(i) == universalTypes[j] {
				return fmt.Errorf("pbeam.ParDo doesn't support DoFns with beam universal types, got function with %v", t.In(i))
			}
		}
	}
	for i := 0; i < t.NumOut(); i++ {
		if t.Out(i).Kind() == reflect.Func {
			if err := checkUniversalTypes(t.Out(i)); err != nil {
				return err
			}
			continue
		}
		for j := 0; j < len(universalTypes); j++ {
			if t.Out(i) == universalTypes[j] {
				return fmt.Errorf("pbeam.ParDo doesn't support DoFns with beam universal types, got function with %v", t.Out(i))
			}
		}
	}
	return nil
}

// buildFunctionalDoFn transforms the input functional doFn (without emit) into an anonDoFn.
func buildFunctionalDoFn(doFn reflectx.Func, t transform) (*anonDoFn, error) {
	fn, _ := funcx.New(doFn)

	kvType := reflect.TypeOf(kv.Pair{})
	kvTypeDef := beam.TypeDefinition{Var: beam.ZType, T: kvType}
	encodedDoFn := beam.EncodedFunc{Fn: doFn}
	// We switch on possible values of transform{hasEmit, hasKVInput, hasKVOutput, hasErrOutput, hasCtxInput}.
	switch t {
	// The following cases are for 1x1 transforms. Here, the doFn has one return
	// value, so we will use the return type of the function.
	case transform{false, false, false, false, false}:
		return &anonDoFn{
			fn:      &generated.TransformFn1x1{Transform: encodedDoFn},
			typeDef: outputTypeDef(fn),
		}, nil
	case transform{false, false, false, false, true}:
		return &anonDoFn{
			fn:      &generated.TransformFnCtx1x1{Transform: encodedDoFn},
			typeDef: outputTypeDef(fn),
		}, nil
	case transform{false, false, false, true, false}:
		return &anonDoFn{
			fn:      &generated.TransformFn1x1Err{Transform: encodedDoFn},
			typeDef: outputTypeDef(fn),
		}, nil
	case transform{false, false, false, true, true}:
		return &anonDoFn{
			fn:      &generated.TransformFnCtx1x1Err{Transform: encodedDoFn},
			typeDef: outputTypeDef(fn),
		}, nil
	// The following cases are for 1x2 transforms. Here, the doFn has two return
	// values (<K,V>), so they will be encoded as a kv.Pair.
	case transform{false, false, true, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn1x2{
				Transform:   encodedDoFn,
				OutputCodec: outputCodec(fn)},
			typeDef: kvTypeDef,
			codec:   outputCodec(fn),
		}, nil
	case transform{false, false, true, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx1x2{
				Transform:   encodedDoFn,
				OutputCodec: outputCodec(fn)},
			typeDef: kvTypeDef,
			codec:   outputCodec(fn),
		}, nil
	case transform{false, false, true, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn1x2Err{
				Transform:   encodedDoFn,
				OutputCodec: outputCodec(fn)},
			typeDef: kvTypeDef,
			codec:   outputCodec(fn),
		}, nil
	case transform{false, false, true, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx1x2Err{
				Transform:   encodedDoFn,
				OutputCodec: outputCodec(fn)},
			typeDef: kvTypeDef,
			codec:   outputCodec(fn),
		}, nil
	// The following cases are for 2x1 transforms. Here, the doFn has one return
	// value, so we will use the return type of the function.
	case transform{false, true, false, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x1{
				Transform:  encodedDoFn,
				InputCodec: inputCodec(fn)},
			typeDef: outputTypeDef(fn),
		}, nil
	case transform{false, true, false, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x1{
				Transform:  encodedDoFn,
				InputCodec: inputCodec(fn)},
			typeDef: outputTypeDef(fn),
		}, nil
	case transform{false, true, false, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x1Err{
				Transform:  encodedDoFn,
				InputCodec: inputCodec(fn)},
			typeDef: outputTypeDef(fn),
		}, nil
	case transform{false, true, false, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x1Err{
				Transform:  encodedDoFn,
				InputCodec: inputCodec(fn)},
			typeDef: outputTypeDef(fn),
		}, nil
	// The following cases are for 2x2 transforms. Here, the doFn has two return
	// values, so they will be encoded as a kv.Pair. But we do not need to
	// supply the type definition in these cases because kv.Pair is also
	// the type of the input.
	case transform{false, true, true, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x2{
				Transform:   encodedDoFn,
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodec(fn)},
			typeDef: beam.TypeDefinition{},
			codec:   outputCodec(fn),
		}, nil
	case transform{false, true, true, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x2{
				Transform:   encodedDoFn,
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodec(fn)},
			typeDef: beam.TypeDefinition{},
			codec:   outputCodec(fn),
		}, nil
	case transform{false, true, true, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x2Err{
				Transform:   encodedDoFn,
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodec(fn)},
			typeDef: beam.TypeDefinition{},
			codec:   outputCodec(fn),
		}, nil
	case transform{false, true, true, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x2Err{
				Transform:   encodedDoFn,
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodec(fn)},
			typeDef: beam.TypeDefinition{},
			codec:   outputCodec(fn),
		}, nil
	default:
		// TODO Add instructions for filing bugs.
		return nil, fmt.Errorf("this case should be unreachable because we check for every possible supported type of transform")
	}
}

func outputTypeDef(fn *funcx.Fn) beam.TypeDefinition {
	return beam.TypeDefinition{
		Var: beam.YType,
		T:   fn.Ret[fn.Returns(funcx.RetValue)[0]].T,
	}
}

func inputCodec(fn *funcx.Fn) *kv.Codec {
	return kv.NewCodec(fn.Param[fn.Params(funcx.FnValue)[0]].T, fn.Param[fn.Params(funcx.FnValue)[1]].T)
}

func outputCodec(fn *funcx.Fn) *kv.Codec {
	return kv.NewCodec(fn.Ret[fn.Returns(funcx.RetValue)[0]].T, fn.Ret[fn.Returns(funcx.RetValue)[1]].T)
}

// buildEmitDoFn transforms the input emit-based doFn into an anonDoFn.
func buildEmitDoFn(doFn reflectx.Func, t transform) (*anonDoFn, error) {
	emitFn := getEmitFn(doFn)
	fn, _ := funcx.New(doFn)
	if len(fn.Params(funcx.FnEmit)) > 1 {
		return nil, fmt.Errorf("multiple emit functions not supported")
	}
	// Beam wouldn't allow this, so this path wouldn't be reached. "couldn't create funcx.Fn from the doFn: bad parameter type"
	if numOut := emitFn.NumOut(); numOut > 0 {
		return nil, fmt.Errorf("emit function should have 0 returns, %d provided", numOut)
	}
	if numRet := len(fn.Returns(funcx.RetValue)); numRet > 0 {
		return nil, fmt.Errorf("return value is not supported if DoFn has an emit function in param, got %d returns", numRet)
	}
	if emitFn == nil {
		return nil, fmt.Errorf("DoFn with 0 return values should have an emit function param")
	}

	encodedDoFn := beam.EncodedFunc{Fn: doFn}
	emitType := emitFn.In(0)
	emitTypeDef := beam.TypeDefinition{Var: beam.YType, T: emitType}
	kvType := reflect.TypeOf(kv.Pair{})
	kvTypeDef := beam.TypeDefinition{Var: beam.ZType, T: kvType}
	// We switch on possible values of transform{hasEmit, hasKVInput, hasKVOutput, hasErrOutput, hasCtxInput}.
	switch t {
	// The following cases are for 1x1 transforms. Here, the doFn emits a single
	// value, so we will use the return type of the function.
	case transform{true, false, false, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn1x1Emit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  encodedDoFn},
			typeDef: emitTypeDef,
		}, nil
	case transform{true, false, false, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx1x1Emit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  encodedDoFn},
			typeDef: emitTypeDef,
		}, nil
	case transform{true, false, false, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn1x1ErrEmit{EmitFnType: beam.EncodedType{
				T: emitFn},
				Transform: encodedDoFn},
			typeDef: emitTypeDef,
		}, nil
	case transform{true, false, false, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx1x1ErrEmit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  encodedDoFn},
			typeDef: emitTypeDef,
		}, nil
	// The following cases are for 1x2 transforms. Here, the doFn emits two
	// values (<K,V>), so they will be encoded as a kv.Pair.
	case transform{true, false, true, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn1x2Emit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   encodedDoFn,
				OutputCodec: outputCodecEmit(emitFn)},
			typeDef: kvTypeDef,
			codec:   outputCodecEmit(emitFn),
		}, nil
	case transform{true, false, true, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx1x2Emit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   encodedDoFn,
				OutputCodec: outputCodecEmit(emitFn)},
			typeDef: kvTypeDef,
			codec:   outputCodecEmit(emitFn),
		}, nil
	case transform{true, false, true, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn1x2ErrEmit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   encodedDoFn,
				OutputCodec: outputCodecEmit(emitFn)},
			typeDef: kvTypeDef,
			codec:   outputCodecEmit(emitFn),
		}, nil
	case transform{true, false, true, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx1x2ErrEmit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   encodedDoFn,
				OutputCodec: outputCodecEmit(emitFn)},
			typeDef: kvTypeDef,
			codec:   outputCodecEmit(emitFn),
		}, nil
	// The following cases are for 2x1 transforms. Here, the doFn emits a single
	// value, so we will use the return type of the function.
	case transform{true, true, false, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x1Emit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  beam.EncodedFunc{Fn: doFn},
				InputCodec: inputCodec(fn),
			},
			typeDef: emitTypeDef,
		}, nil
	case transform{true, true, false, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x1Emit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  beam.EncodedFunc{Fn: doFn},
				InputCodec: inputCodec(fn),
			},
			typeDef: emitTypeDef,
		}, nil
	case transform{true, true, false, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x1ErrEmit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  beam.EncodedFunc{Fn: doFn},
				InputCodec: inputCodec(fn),
			},
			typeDef: emitTypeDef,
		}, nil
	case transform{true, true, false, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x1ErrEmit{
				EmitFnType: beam.EncodedType{T: emitFn},
				Transform:  beam.EncodedFunc{Fn: doFn},
				InputCodec: inputCodec(fn),
			},
			typeDef: emitTypeDef,
		}, nil
	// The following cases are for 2x2 transforms. Here, the doFn emits two
	// values, so they will be encoded as a kv.Pair. But we do not need to
	// supply the type definition in these cases because kv.Pair is also
	// the type of the input.
	case transform{true, true, true, false, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x2Emit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   beam.EncodedFunc{Fn: doFn},
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodecEmit(emitFn),
			},
			codec: outputCodecEmit(emitFn),
		}, nil
	case transform{true, true, true, false, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x2Emit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   beam.EncodedFunc{Fn: doFn},
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodecEmit(emitFn),
			},
			codec: outputCodecEmit(emitFn),
		}, nil
	case transform{true, true, true, true, false}:
		return &anonDoFn{
			fn: &generated.TransformFn2x2ErrEmit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   beam.EncodedFunc{Fn: doFn},
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodecEmit(emitFn),
			},
			codec: outputCodecEmit(emitFn),
		}, nil
	case transform{true, true, true, true, true}:
		return &anonDoFn{
			fn: &generated.TransformFnCtx2x2ErrEmit{
				EmitFnType:  beam.EncodedType{T: emitFn},
				Transform:   beam.EncodedFunc{Fn: doFn},
				InputCodec:  inputCodec(fn),
				OutputCodec: outputCodecEmit(emitFn),
			},
			codec: outputCodecEmit(emitFn),
		}, nil
	default:
		// TODO Add instructions for filing bugs.
		return nil, fmt.Errorf("this case should be unreachable because emitter based doFns must have an emitter input")
	}
}

func outputCodecEmit(fn reflect.Type) *kv.Codec {
	return kv.NewCodec(fn.In(0), fn.In(1))
}

func getEmitFn(doFn reflectx.Func) reflect.Type {
	n := doFn.Type().NumIn() - 1
	fn := doFn.Type().In(n)
	if fn.Kind() == reflect.Func {
		return fn
	}
	return nil
}

// kind: ParamKind and ReturnKind are both ints
// this type makes the validOrder function easier to read
type kind int

// this logic is reused for validateArgOrder and validateReturnOrder
func validOrder(wantOrder []kind, haveOrder []kind) (valid bool, badIndex int) {
	// this will create a map like:
	//	firstElement: 0,
	//	secondElement: 1,
	//	...
	//	lastElement: <length-1>
	indices := make(map[kind]int)
	for i, w := range wantOrder {
		indices[w] = i
	}

	// Keep track of the index of the previous element
	prev := -1
	for i, h := range haveOrder {
		cur := indices[h]
		// If we ever "jump backwards", the order is wrong
		if prev > cur {
			return false, i
		}
		prev = cur
	}
	return true, 0
}

func validateArgOrder(fn *funcx.Fn) error {
	order := []kind{
		kind(funcx.FnContext),
		kind(funcx.FnEventTime),
		kind(funcx.FnValue),
		kind(funcx.FnEmit),
	}

	fnOrder := make([]kind, len(fn.Param))
	for i, p := range fn.Param {
		fnOrder[i] = kind(p.Kind)
	}

	if valid, badIndex := validOrder(order, fnOrder); !valid {
		return fmt.Errorf("doFn's parameter number %d is in the wrong order (should be earlier)", badIndex+1)
	}
	return nil
}

func validateRetOrder(fn *funcx.Fn) error {
	order := []kind{
		kind(funcx.RetEventTime),
		kind(funcx.RetValue),
		kind(funcx.RetError),
	}

	fnOrder := make([]kind, len(fn.Ret))
	for i, r := range fn.Ret {
		fnOrder[i] = kind(r.Kind)
	}

	if valid, badIndex := validOrder(order, fnOrder); !valid {
		return fmt.Errorf("doFn's return number %d is in the wrong order (should be earlier)", badIndex+1)
	}
	return nil
}
