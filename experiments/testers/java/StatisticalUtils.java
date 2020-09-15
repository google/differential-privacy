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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import java.io.*;
import java.lang.*;
import java.util.*;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Date;

public class StatisticalUtils {
// Reads in samples of Count algorithms with insufficient noise.
	protected static Long[] getData(String filepath) {

	  File file = new File(filepath);
	  int counter = 0;
	  ArrayList<Long> sample = new ArrayList<Long>();

	  try (BufferedReader br = new BufferedReader(new FileReader(file))) {
	    String line;
	    while ((line = br.readLine()) != null) {
	    	counter++;
				int i = Double.valueOf(line).intValue();
	      Long dpcount = Long.valueOf(i);
	      sample.add(dpcount);
	    }
	  } catch (IOException e) {
	      e.printStackTrace();
	  }
	    Long sample_array[] = new Long[counter];
	    sample_array = sample.toArray(sample_array);
	    return sample_array;
	}

// Runs the Statistical Tester on a given pair of differentially protected samples.
// Desired outcome is 0 or false since the samples are known to violate DP.
	protected static boolean getOutcome(
		String fileA,
		String fileB,
	  int numberOfSamples,
	  double epsilon,
	  double delta,
	  double l2Tolerance) {

		Long[] samplesA = getData(fileA);
		Long[] samplesB = getData(fileB);

	  return StatisticalTestsUtil.verifyApproximateDp(samplesA, samplesB, epsilon,
	  	delta, l2Tolerance);
	}

// Determines overall outcome of the test by taking a majority vote. 
	protected static int getMajorityVote(
		String subfolder,
		int numberOfSamples,
		double epsilon,
		double delta,
		double l2Tolerance,
		int numberOfVotes) {

		int numPassed = 0;
		
		for (int i = 0; i < numberOfVotes; i++) {
  		String fileA = subfolder+"TestCase"+Integer.toString(i)+"A.txt";
  		String fileB = subfolder+"TestCase"+Integer.toString(i)+"B.txt";
  		boolean Outcome = getOutcome(fileA,fileB,numberOfSamples,epsilon,delta,l2Tolerance);
  		if (Outcome == true) {
  			numPassed++;
  		}
		}

		System.out.println("The algorithm passed "+Integer.toString(numPassed)
			+" out of "+Integer.toString(numberOfVotes)+" test runs.");

		long majorityVote = Math.round(numberOfVotes*0.5);
		if (numPassed >= majorityVote) {
			return 1;
		}
		else {
			return 0;
		}
	}
}