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

/** Runs the Statistical Tester on differentially private sums with
insufficient noise. Records the results in an output text file.
*/
public class StatisticalTesterSum {

	private static final String homedir = "boundedsumsamples/";

// Run each test case according to parameters specified by SumDpTest.java.
// If any one test fails to satisfy DP, the algorithm is considered not DP.
	public static int getOverallOutcome(int numberOfSamples, String ratio,
		int numberOfVotes) {

		int counter = 0;
		double small_epsilon = 0.1;
		double medium_epsilon = Math.log(3);
		double large_epsilon = 2*Math.log(3);
		final int delta = 0;

		int testcase1 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario1/",numberOfSamples,
			medium_epsilon,delta,0.02,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase2 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario2/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase3 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario3/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase4 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario4/",numberOfSamples,
			small_epsilon,delta,0.0135,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase5 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario5/",numberOfSamples,
			large_epsilon,delta,0.0135,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase6 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario6/",numberOfSamples,
			medium_epsilon,delta,0.02,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase7 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario7/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase8 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario8/",numberOfSamples,
			small_epsilon,delta,0.0135,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase9 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario9/",numberOfSamples,
			large_epsilon,delta,0.0135,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase10 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario10/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase11 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario11/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase12 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario12/",numberOfSamples,
			small_epsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase13 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario13/",numberOfSamples,
			large_epsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase14 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario14/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase15 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario15/",numberOfSamples,
			medium_epsilon,delta,0.04,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testcase16 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario16/",numberOfSamples,
				small_epsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testcase17 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario17/",numberOfSamples,
			large_epsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int all_cases = testcase1+testcase2+testcase3+testcase4+testcase5+testcase6
		+testcase7+testcase8+testcase9+testcase10+testcase11+testcase12+testcase13
		+testcase14+testcase15+testcase16+testcase17;
		System.out.println("The algorithm incorrectly passed "+Integer.toString(all_cases)+" out of 17 tests.");

		if (all_cases < 17) {
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
		String algorithm_type = "bounded_sum";
		String expected = "0";
		String num_datasets = "17";
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
	    builder.append(test_name+","+algorithm_type+","+expected+","+Outcome+","+r+","+num_datasets+","+num_samples+","+time);
	    builder.append('\n');
    }
    pw.write(builder.toString());
 		pw.close();
	}
}
