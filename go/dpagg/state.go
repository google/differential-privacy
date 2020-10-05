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

package dpagg

type state int

var errorMessages = []string{"", " object has been already merged.", " object has been already serialized.", "Noised result was already computed and returned." }
var stateName = []string{"Default", "Merged", "Serialized", "ResultReturned"}
const (
	Default = iota
	Merged
	Serialized
	ResultReturned
)

func (s state) errorMessage(label string) string {
	if(s == Merged || s == Serialized){
		return label + errorMessages[s]
	}
	return errorMessages[s]
}

func (s state) String() string{
	return stateName[s]
}