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

// Package kv contains Pair for holding <K,V> pairs as byte slices,
// and Codec for transforming <K,V> pairs into Pair and vice versa.
package kv

import (
	"bytes"
	"fmt"
	"reflect"

	log "github.com/golang/glog"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
)

func init() {
	register.DoFn2x1[beam.T, beam.V, Pair](&EncodeFn{})
	register.DoFn1x2[Pair, beam.T, beam.V](&DecodeFn{})
}

// Codec provides functions for encoding a <K,V> pair into a Pair and
// decoding a Pair into a <K,V> pair. It can be used for performing <K,V>
// transforms. Whenever a Codec is used inside a structural doFn, it should
// be an exported field in order to be serialized.
//
// After initialization, Setup function should be called before any calls are
// made to Encode/Decode.
type Codec struct {
	KType beam.EncodedType
	VType beam.EncodedType

	kEnc beam.ElementEncoder
	vEnc beam.ElementEncoder

	kDec beam.ElementDecoder
	vDec beam.ElementDecoder
}

// NewCodec returns a new Codec with specified <K,V> type.
func NewCodec(kType, vType reflect.Type) *Codec {
	if kType == nil || vType == nil {
		return nil
	}
	return &Codec{
		KType: beam.EncodedType{kType},
		VType: beam.EncodedType{vType},
	}
}

// Setup initializes the encoders and decoders. This functions needs to be
// called once before any calls are made to Encode/Decode.
func (codec *Codec) Setup() error {
	if codec.KType.T == nil || codec.VType.T == nil {
		return fmt.Errorf("Codec should be initialized using the NewCodec method")
	}
	codec.kEnc = beam.NewElementEncoder(codec.KType.T)
	codec.vEnc = beam.NewElementEncoder(codec.VType.T)
	codec.kDec = beam.NewElementDecoder(codec.KType.T)
	codec.vDec = beam.NewElementDecoder(codec.VType.T)
	return nil
}

// Pair contains a KV<K,V> pair, both values encoded as []byte.
type Pair struct {
	K []byte
	V []byte
}

// Encode transforms a <K,V> pair into a Pair.
func (codec *Codec) Encode(k, v any) (Pair, error) {
	var bufK, bufV bytes.Buffer
	if err := codec.kEnc.Encode(k, &bufK); err != nil {
		return Pair{}, fmt.Errorf("kv.Codec.Encode: couldn't Encode key %v: %v", k, err)
	}
	if err := codec.vEnc.Encode(v, &bufV); err != nil {
		return Pair{}, fmt.Errorf("kv.Codec.Encode: couldn't Encode value %v: %v", v, err)
	}
	return Pair{
		K: bufK.Bytes(),
		V: bufV.Bytes(),
	}, nil
}

// Decode transforms a Pair into a <K,V> pair.
func (codec *Codec) Decode(p Pair) (k, v any, err error) {
	k, err = codec.kDec.Decode(bytes.NewBuffer(p.K))
	if err != nil {
		return k, v, fmt.Errorf("kv.Codec.Decode: couldn't Decode key %v: %v", k, err)
	}
	v, err = codec.vDec.Decode(bytes.NewBuffer(p.V))
	if err != nil {
		return k, v, fmt.Errorf("kv.Codec.Decode: couldn't Decode value %v: %v", v, err)
	}
	return k, v, nil
}

// EncodeFn transforms a PCollection<K,V> into a PCollection<Pair>.
type EncodeFn struct {
	KType beam.EncodedType
	VType beam.EncodedType

	kEnc beam.ElementEncoder
	vEnc beam.ElementEncoder
}

// NewEncodeFn returns an EncodeFn from given types.
func NewEncodeFn(kT, vT typex.FullType) *EncodeFn {
	return &EncodeFn{
		KType: beam.EncodedType{kT.Type()},
		VType: beam.EncodedType{vT.Type()},
	}
}

// Setup initializes the encoders.
func (fn *EncodeFn) Setup() {
	fn.kEnc = beam.NewElementEncoder(fn.KType.T)
	fn.vEnc = beam.NewElementEncoder(fn.VType.T)
}

// ProcessElement encodes a <K,V> as a Pair.
func (fn *EncodeFn) ProcessElement(k beam.T, v beam.V) Pair {
	var bufK bytes.Buffer
	if err := fn.kEnc.Encode(k, &bufK); err != nil {
		log.Exitf("kv.EncodeFn.ProcessElement: couldn't encode key %v: %v", k, err)
	}
	var bufV bytes.Buffer
	if err := fn.vEnc.Encode(v, &bufV); err != nil {
		log.Exitf("kv.EncodeFn.ProcessElement: couldn't encode value %v: %v", v, err)
	}
	return Pair{
		K: bufK.Bytes(),
		V: bufV.Bytes(),
	}
}

// DecodeFn transforms a PCollection<codedKV> into a PCollection<K,V>.
type DecodeFn struct {
	KType beam.EncodedType
	VType beam.EncodedType

	kDec beam.ElementDecoder
	vDec beam.ElementDecoder
}

// NewDecodeFn returns a DecodeFn from given types.
func NewDecodeFn(kT, vT typex.FullType) *DecodeFn {
	return &DecodeFn{
		KType: beam.EncodedType{kT.Type()},
		VType: beam.EncodedType{vT.Type()},
	}
}

// Setup initializes the decoders.
func (fn *DecodeFn) Setup() {
	fn.kDec = beam.NewElementDecoder(fn.KType.T)
	fn.vDec = beam.NewElementDecoder(fn.VType.T)
}

// ProcessElement decodes a Pair into a <K,V>.
func (fn *DecodeFn) ProcessElement(c Pair) (beam.T, beam.V) {
	k, err := fn.kDec.Decode(bytes.NewBuffer(c.K))
	if err != nil {
		log.Exitf("kv.DecodeFn.ProcessElement: couldn't decode key %v: %v", k, err)
	}
	v, err := fn.vDec.Decode(bytes.NewBuffer(c.V))
	if err != nil {
		log.Exitf("kv.DecodeFn.ProcessElement: couldn't decode value %v: %v", v, err)
	}
	return k, v
}
