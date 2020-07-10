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

package rand

import (
	"bytes"
	"testing"
)

func TestBooleanBufIsShifting(t *testing.T) {
	randBuf = bytes.NewReader([]byte{
		0b00100100,
		0b10010000,
	})
	for pos, want := range []bool{
		// first byte
		false,
		false,
		true,
		false,
		false,
		true,
		false,
		false,
		// second byte
		false,
		false,
		false,
		false,
		true,
		false,
		false,
		true,
	} {
		if got := Boolean(); got != want {
			t.Errorf("Boolean: got %v, want %v in %v-th iteration", got, want, pos)
		}
	}
}
