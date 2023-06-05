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

package codelab

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
)

func init() {
	register.Function2x1[string, func(Visit), error](CreateVisitsFn)
	register.Emitter1[Visit]()
}

// Visit represents a visit from a user to the restaurant.
type Visit struct {
	VisitorID    string
	TimeEntered  time.Time
	MinutesSpent int
	MoneySpent   int
}

// CreateVisitsFn creates and emits a Visit struct from a line that holds visit information.
func CreateVisitsFn(line string, emit func(Visit)) error {
	// Skip the column headers line
	notHeader, err := regexp.MatchString("[0-9]", line)
	if err != nil {
		return err
	}
	if !notHeader {
		return nil
	}

	cols := strings.Split(line, ",")
	if len(cols) != 4 {
		return fmt.Errorf("got %d number of columns in line %q, expected 4", len(cols), line)
	}
	visitorID := cols[0]
	timeEntered, err := time.Parse(time.Kitchen, cols[1])
	if err != nil {
		return err
	}
	timeSpent, err := strconv.Atoi(cols[2])
	if err != nil {
		return err
	}
	moneySpent, err := strconv.Atoi(cols[3])
	if err != nil {
		return err
	}
	emit(Visit{
		VisitorID:    visitorID,
		TimeEntered:  timeEntered,
		MinutesSpent: timeSpent,
		MoneySpent:   moneySpent,
	})
	return nil
}
