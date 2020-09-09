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

// Run each test case according to parameters specified by MeanDpTest.java.
// If any one test fails to satisfy DP, the algorithm is considered not DP.
	private static int getOverallOutcome(int numberOfSamples, String ratio,
		int numberOfVotes) {

		int counter = 0;
		double smallEpsilon = 0.1;
		double mediumEpsilon = Math.log(3);
		double largeEpsilon = 2*Math.log(3);
		final int delta = 0;

		int testCase1 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario1/",numberOfSamples,
			mediumEpsilon,0,0.003,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase2 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario2/",numberOfSamples,
			mediumEpsilon,0,0.0075,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase3 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario3/",numberOfSamples,
			mediumEpsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase4 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario4/",numberOfSamples,
			mediumEpsilon,0,0.007,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase5 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario5/",numberOfSamples,
			smallEpsilon,0,0.008,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase6 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario6/",numberOfSamples,
			largeEpsilon,0,0.0009,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase7 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario7/",numberOfSamples,
			mediumEpsilon,0,0.003,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase8 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario8/",numberOfSamples,
			mediumEpsilon,0,0.0075,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase9 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario9/",numberOfSamples,
			mediumEpsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase10 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario10/",numberOfSamples,
			smallEpsilon,0,0.008,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase11 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario11/",numberOfSamples,
			largeEpsilon,0,0.0009,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase12 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario12/",numberOfSamples,
			mediumEpsilon,0,0.003,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase13 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario13/",numberOfSamples,
			mediumEpsilon,0,0.004,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase14 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario14/",numberOfSamples,
			mediumEpsilon,0,0.025,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase15 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario15/",numberOfSamples,
			mediumEpsilon,0,0.014,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase16 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario16/",numberOfSamples,
			smallEpsilon,0,0.0105,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase17 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario17/",numberOfSamples,
			largeEpsilon,0,0.002,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase18 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario18/",numberOfSamples,
			mediumEpsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase19 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario19/",numberOfSamples,
			mediumEpsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase20 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario20/",numberOfSamples,
			mediumEpsilon,0,0.026,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase21 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario21/",numberOfSamples,
			mediumEpsilon,0,0.014,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase22 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario22/",numberOfSamples,
			mediumEpsilon,0,0.004,numberOfVotes);
		counter++;
		System.out.println(counter);


		int allCases = testCase1+testCase2+testCase3+testCase4+testCase5+testCase6
		+testCase7+testCase8+testCase9+testCase10+testCase11+testCase12+testCase13
		+testCase14+testCase15+testCase16+testCase17+testCase18+testCase19
		+testCase20+testCase21+testCase22;
		System.out.println("The algorithm incorrectly passed "+Integer.toString(allCases)+" out of 22 tests.");

		if (allCases < 22) {
			System.out.println("The algorithm fails according to this Tester.");
			return 0;
		}

		else { 
			System.out.println("The algorithm passes according to this Tester.");
			return 1;
		}
	}

	protected static void collectData(int numberOfVotes, int numberOfSamples, int ratioMin,
		int ratioMax, String fileName) {
		String testName = "insufficient_noise";
		String algorithmType = "bounded_mean";
		String expected = "0";
		String numDatasets = "22";
		String numSamples = Integer.toString(numberOfSamples);
		String time = "null";
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss");

    PrintWriter pw = null;
    try {
      pw = new PrintWriter(new File("../results/"+fileName));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    StringBuilder builder = new StringBuilder();
    Timestamp timestamp = new Timestamp(System.currentTimeMillis());
    builder.append("Results run on: "+timestamp + "\n");
    String columnNamesList = "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)";
    builder.append(columnNamesList +"\n");

		for (int i = ratioMin; i <= ratioMax; i++) {
			String r = Integer.toString(i);
			String Outcome = Integer.toString(getOverallOutcome(numberOfSamples,r,numberOfVotes));
	    builder.append(testName+","+algorithmType+","+expected+","+Outcome+","+r+","+numDatasets+","+numSamples+","+time);
	    builder.append('\n');
    }
    pw.write(builder.toString());
 		pw.close();
	}
}