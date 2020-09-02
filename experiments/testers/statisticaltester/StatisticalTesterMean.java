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

/** Runs the Statistical Tester on samples of differentially private means with
insufficient noise. Records the results in an output text file.
*/
public class StatisticalTesterMean {

	private static final String homedir = "boundedmeansamples/";

// Reads in samples of Bounded Mean algorithms with insufficient noise.
	public static Long[] getData(String filepath) {

	  File file = new File(filepath);
	  int counter = 0;
	  ArrayList<Long> sample = new ArrayList<Long>();

	  try (BufferedReader br = new BufferedReader(new FileReader(file))) {
	    String line;
	    while ((line = br.readLine()) != null) {
	    	counter++;
				int i = Double.valueOf(line).intValue();
	    	Long dpmean = Long.valueOf(i);
	      sample.add(dpmean);
	    }
	  } catch (IOException e) {
	      e.printStackTrace();
	  }

	  Long sample_array[] = new Long[counter];
	  sample_array = sample.toArray(sample_array);
	  return sample_array;
	}

// Runs the Statistical Tester on a given pair of differentially private samples.
// Desired outcome is 0 or false since the samples are known to violate DP.
	public static boolean getOutcome(
		String fileA,
		String fileB,
	  int numberOfSamples,
	  double epsilon,
	  double delta,
	  double l2Tolerance,
	  int numberOfVotes) {

		Long[] samplesA = getData(fileA);
		Long[] samplesB = getData(fileB);

	    return StatisticalTestsUtil.verifyApproximateDp(
	        samplesA, samplesB, epsilon, delta, l2Tolerance);
	}

// Uses majority-vote protocol to determine overall outcome of the test. 
	public static int getMajorityVote(
		String subfolder,
		int numberOfSamples,
		double epsilon,
		double delta,
		double l2Tolerance,
		int numberOfVotes) {

		String fullpath = homedir+subfolder;
		int num_passed = 0;
		
		for (int i = 0; i < 7; i++) {
  			String fileA = fullpath+"TestCase"+Integer.toString(i)+"A.txt";
  			String fileB = fullpath+"TestCase"+Integer.toString(i)+"B.txt";
  			boolean Outcome = getOutcome(fileA,fileB,numberOfSamples,epsilon,delta,l2Tolerance,numberOfVotes);
  			if (Outcome == true) {
  				num_passed++;
  			}
		}

		System.out.println("The algorithm passed "+Integer.toString(num_passed)
			+" out of "+Integer.toString(numberOfVotes)+" test runs.");

		long majority_vote = Math.round(numberOfVotes*0.5);
		if (num_passed >= majority_vote) {
			return 1;
		}
		else {
			return 0;
		}
	}

// Run each test case according to parameters specified by MeanDpTest.java.
// If any one test fails to satisfy DP, the algorithm is considered not DP.
	public static int getOverallOutcome(int numberOfSamples, String ratio,
		int numberOfVotes) {

		int counter = 0;
		double small_epsilon = 0.1;
		double medium_epsilon = Math.log(3);
		double large_epsilon = 2*Math.log(3);
		final int delta = 0;

		int testcase1 = getMajorityVote("R"+ratio+"/Scenario1/",numberOfSamples,
			medium_epsilon,0,0.003,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase2 = getMajorityVote("R"+ratio+"/Scenario2/",numberOfSamples,
			medium_epsilon,0,0.0075,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase3 = getMajorityVote("R"+ratio+"/Scenario3/",numberOfSamples,
			medium_epsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase4 = getMajorityVote("R"+ratio+"/Scenario4/",numberOfSamples,
			medium_epsilon,0,0.007,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase5 = getMajorityVote("R"+ratio+"/Scenario5/",numberOfSamples,
			small_epsilon,0,0.008,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase6 = getMajorityVote("R"+ratio+"/Scenario6/",numberOfSamples,
			large_epsilon,0,0.0009,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase7 = getMajorityVote("R"+ratio+"/Scenario7/",numberOfSamples,
			medium_epsilon,0,0.003,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase8 = getMajorityVote("R"+ratio+"/Scenario8/",numberOfSamples,
			medium_epsilon,0,0.0075,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase9 = getMajorityVote("R"+ratio+"/Scenario9/",numberOfSamples,
			medium_epsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase10 = getMajorityVote("R"+ratio+"/Scenario10/",numberOfSamples,
			small_epsilon,0,0.008,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase11 = getMajorityVote("R"+ratio+"/Scenario11/",numberOfSamples,
			large_epsilon,0,0.0009,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase12 = getMajorityVote("R"+ratio+"/Scenario12/",numberOfSamples,
			medium_epsilon,0,0.003,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase13 = getMajorityVote("R"+ratio+"/Scenario13/",numberOfSamples,
			medium_epsilon,0,0.004,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase14 = getMajorityVote("R"+ratio+"/Scenario14/",numberOfSamples,
			medium_epsilon,0,0.025,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase15 = getMajorityVote("R"+ratio+"/Scenario15/",numberOfSamples,
			medium_epsilon,0,0.014,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase16 = getMajorityVote("R"+ratio+"/Scenario16/",numberOfSamples,
			small_epsilon,0,0.0105,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase17 = getMajorityVote("R"+ratio+"/Scenario17/",numberOfSamples,
			large_epsilon,0,0.002,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase18 = getMajorityVote("R"+ratio+"/Scenario18/",numberOfSamples,
			medium_epsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase19 = getMajorityVote("R"+ratio+"/Scenario19/",numberOfSamples,
			medium_epsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase20 = getMajorityVote("R"+ratio+"/Scenario20/",numberOfSamples,
			medium_epsilon,0,0.026,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase21 = getMajorityVote("R"+ratio+"/Scenario21/",numberOfSamples,
			medium_epsilon,0,0.014,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase22 = getMajorityVote("R"+ratio+"/Scenario22/",numberOfSamples,
			medium_epsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);


		int all_cases = testcase1+testcase2+testcase3+testcase4+testcase5+testcase6
		+testcase7+testcase8+testcase9+testcase10+testcase11+testcase12+testcase13
		+testcase14+testcase15+testcase16+testcase17+testcase18+testcase19
		+testcase20+testcase21+testcase22;
		System.out.println("The algorithm incorrectly passed "+Integer.toString(all_cases)+" out of 22 tests.");

		if (all_cases < 22) {
			System.out.println("The algorithm fails according to this Tester.");
			return 0;
		}

		else { 
			System.out.println("The algorithm passes according to this Tester.");
			return 1;
		}
	}

	public static void collectData(int numberOfVotes, int numberOfSamples, int ratio_min,
		int ratio_max, String filename) {
		String test_name = "insufficient_noise";
		String algorithm = "bounded_mean";
		String expected = "0";
		String num_datasets = "22";
		String num_samples = Integer.toString(numberOfSamples);
		String time = "null";
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss");

    PrintWriter pw = null;
    try {
      pw = new PrintWriter(new File("../results/"+filename));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    StringBuilder builder = new StringBuilder();
    Timestamp timestamp = new Timestamp(System.currentTimeMillis());
    builder.append("Results run on: "+timestamp + "\n");
    String columnNamesList = "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)";
    builder.append(columnNamesList +"\n");

		for (int i = ratio_min; i <= ratio_max; i++) {
			String r = Integer.toString(i);
			String Outcome = Integer.toString(getOverallOutcome(numberOfSamples,r,numberOfVotes));
	    builder.append(test_name+","+algorithm+","+expected+","+Outcome+","+r+","+num_datasets+","+num_samples+","+time);
	    builder.append('\n');
    }
    pw.write(builder.toString());
 		pw.close();
	}
}