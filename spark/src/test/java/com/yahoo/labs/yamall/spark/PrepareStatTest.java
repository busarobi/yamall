package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 11/6/17.
 */
public class PrepareStatTest {
    @Test
    public void testComputeStat() throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("compute stat");
        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");

        sparkConf.set("spark.myapp.inputdir", "/Users/busafekete/work/DistOpt/Clkb_data/");
        sparkConf.set("spark.myapp.output", "/Users/busafekete/work/DistOpt/report.txt");

        String inputDir = sparkConf.get("spark.myapp.inputdir");
        String output = sparkConf.get("spark.myapp.output");

        StringBuilder strb = new StringBuilder("");
        strb.append( "Input: " + inputDir + "\n" );
        strb.append( "Output: " + output + "\n" );

        System.out.println( strb.toString() + "\n" );

        FileWriterToHDFS.writeToHDFS( output, strb.toString());

        PrepareStat.computeStat(strb, sparkContext, inputDir);

        System.out.println( strb.toString());
        FileWriterToHDFS.writeToHDFS( output, strb.toString());
    }

}