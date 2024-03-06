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

	"github.com/apache/beam/sdks/v2/go/pkg/beam/io/textio"
	"github.com/google/differential-privacy/privacy-on-beam/v3/codelab"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const (
	// Constants to differentiate between examples.
	count            = "count"
	mean             = "mean"
	sum              = "sum"
	publicPartitions = "public_partitions"
)

func drawPlot(hourToValue, dpHourToValue map[int]float64, example, nonDPOutput, dpOutput string) error {
	// Sort dp and non-dp points.
	keys := make([]int, 0)
	for k := range hourToValue {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	points := make([]float64, 0)
	for _, k := range keys {
		points = append(points, hourToValue[k])
	}

	dpKeys := make([]int, 0)
	for k := range dpHourToValue {
		dpKeys = append(dpKeys, k)
	}
	sort.Ints(dpKeys)
	dpPoints := make([]float64, 0)
	for _, k := range dpKeys {
		dpPoints = append(dpPoints, dpHourToValue[k])
	}

	p := plot.New()

	p.X.Label.Text = "Hour"
	switch example {
	case count, publicPartitions: // count & publicPartitions both compute visits per hour.
		p.Y.Label.Text = "Visits"
		p.Title.Text = "Visits Per Hour"
	case mean:
		p.Y.Label.Text = "Time Spent"
		p.Title.Text = "Mean Time Spent"
	case sum:
		p.Y.Label.Text = "Revenue"
		p.Title.Text = "Revenue Per Hour"
	default:
		return fmt.Errorf("unknown example %q specified, please use one of 'count', 'sum', 'mean', 'public_partitions'", example)
	}

	w := vg.Points(20)

	// Non-DP Plot
	bars, err := plotter.NewBarChart(plotter.Values(points), w)
	if err != nil {
		return fmt.Errorf("could not create bars from points %v: %v", plotter.Values(points), err)
	}
	bars.LineStyle.Width = vg.Length(0)
	bars.Color = plotutil.Color(2)

	p.Add(bars)
	p.NominalX("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23")

	// Save non-dp plot.
	if err := p.Save(10*vg.Inch, 5*vg.Inch, nonDPOutput); err != nil {
		return fmt.Errorf("Could not save plot: %v", err)
	}

	// DP Plot
	dpBars, err := plotter.NewBarChart(plotter.Values(dpPoints), w)
	if err != nil {
		return fmt.Errorf("could not create bars from points %v: %v", plotter.Values(dpPoints), err)
	}
	dpBars.LineStyle.Width = vg.Length(0)
	dpBars.Color = plotutil.Color(3)
	dpBars.Offset = w

	p.Add(dpBars)
	p.Legend.Add("Raw", bars)
	p.Legend.Add("Private", dpBars)
	p.Legend.Top = true

	// Save dp plot.
	if err := p.Save(15*vg.Inch, 5*vg.Inch, dpOutput); err != nil {
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

func writeOutput(s beam.Scope, output beam.PCollection, outputTextName string) {
	s = s.Scope("writeOutput")
	output = beam.ParDo(s, convertToPairFn, output)
	formattedOutput := beam.Combine(s, &normalizeOutputCombineFn{}, output)
	textio.Write(s, outputTextName, formattedOutput)
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
