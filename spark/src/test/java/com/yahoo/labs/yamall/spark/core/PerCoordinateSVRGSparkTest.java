package com.yahoo.labs.yamall.spark.core;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 9/16/17.
 */
public class PerCoordinateSVRGSparkTest {
    protected static String inputDir = "/Users/busafekete/work/DistOpt/Clkb_data/";
    protected static int bits = 22;
    @Test
    public void testTrain() throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (parallel training)");

        sparkConf.set("spark.myapp.outdir", "/Users/busafekete/work/DistOpt/tmp/");
        sparkConf.set("spark.myapp.iter", "10");

        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");

        JavaRDD<String> input = null;
        input = sparkContext.textFile(inputDir);
        StringBuilder strb = new StringBuilder("");

        PerCoordinateSVRGSpark learner = new PerCoordinateSVRGSpark(sparkConf,strb,bits);
        learner.train(input);

    }

}