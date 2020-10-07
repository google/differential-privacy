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

type aggregationState int

var errorMessages = map[int]string{
	Default:        "",
	Merged:         "Object has been already merged",
	Serialized:     "Object has been already serialized",
	ResultReturned: "Noised result is already computed and returned",
}
var stateName = map[int]string{
	Default:        "Default",
	Merged:         "Merged",
	Serialized:     "Serialized",
	ResultReturned: "ResultReturned",
}

const (
	Default = iota
	Merged
	Serialized
	ResultReturned
)

func (s aggregationState) errorMessage() string {
	return errorMessages[int(s)]
}

func (s aggregationState) String() string {
	return stateName[int(s)]
}
