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

package main

import (
	"fmt"
	"io/ioutil"
	"sort"
	"strconv"
	"strings"

	"github.com/apache/beam/sdks/go/pkg/beam/io/textio"
	"github.com/google/differential-privacy/privacy-on-beam/codelab"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const (
	// Constants to differentiate between examples.
	count = "count"
	mean  = "mean"
	sum   = "sum"
)

func drawPlot(hourToValue map[int]float64, dp bool, example, output string) error {
	keys := make([]int, 0)
	for k := range hourToValue {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	points := make([]float64, 0)
	for _, k := range keys {
		points = append(points, hourToValue[k])
	}

	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("could not create plot: %v", err)
	}

	p.X.Label.Text = "Hour"
	switch example {
	case count:
		p.Y.Label.Text = "Visits"
		if dp {
			p.Title.Text = "Private Visits Per Hour"
		} else {
			p.Title.Text = "Non-Private Visits Per Hour"
		}
	case mean:
		p.Y.Label.Text = "Time Spent"
		if dp {
			p.Title.Text = "Private Mean Time Spent"
		} else {
			p.Title.Text = "Non-Private Mean Time Spent"
		}
	case sum:
		p.Y.Label.Text = "Revenue"
		if dp {
			p.Title.Text = "Private Revenue Per Hour"
		} else {
			p.Title.Text = "Non-Private Revenue Per Hour"
		}
	default:
		return fmt.Errorf("unknown example %q specified, please use one of 'count', 'sum', 'mean'", example)
	}

	w := vg.Points(20)

	bars, err := plotter.NewBarChart(plotter.Values(points), w)
	if err != nil {
		return fmt.Errorf("could not create bars from points %v: %v", plotter.Values(points), err)
	}
	bars.LineStyle.Width = vg.Length(0)
	bars.Color = plotutil.Color(0)

	p.Add(bars)
	p.NominalX("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23")

	if err := p.Save(10*vg.Inch, 5*vg.Inch, output); err != nil {
		return fmt.Errorf("Could not save plot: %v", err)
	}
	return nil
}

// readInput reads from a .csv file detailing visits to a restaurant in the form
// of "visitor_id, visit time, minutes spent, money spent" and returns a
// PCollection of Visit structs.
func readInput(s beam.Scope, input string) beam.PCollection {
	s = s.Scope("readInput")
	lines := textio.Read(s, input)
	return beam.ParDo(s, codelab.CreateVisitsFn, lines)
}

// readOutput reads from a .txt file where each line has an hour (int) associated with
// a value (float64) separated by a whitespace and returns a map of these hour to value
// pairs.
// Returns an error if there is an error reading the output file.
func readOutput(output string) (map[int]float64, error) {
	hourToValue := make(map[int]float64)
	contents, err := ioutil.ReadFile(output)
	if err != nil {
		return nil, fmt.Errorf("could not read output file %s", output)
	}
	lines := strings.Split(string(contents), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		elements := strings.Split(line, " ")
		if len(elements) != 2 {
			return nil, fmt.Errorf("got %d number of elements in line %q, expected 2", len(elements), line)
		}
		hour, err := strconv.Atoi(elements[0])
		if err != nil {
			return nil, fmt.Errorf("could not convert hour %s to int: %v", elements[0], err)
		}
		value, err := strconv.ParseFloat(elements[1], 64)
		if err != nil {
			return nil, fmt.Errorf("could not convert value %s to float64: %v", elements[1], err)
		}
		hourToValue[hour] = value
	}
	return hourToValue, nil
}
