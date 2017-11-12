package com.yahoo.labs.yamall.spark.helper;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 11/6/17.
 */
public class ConvertVWToLibSVMTest {
    @Test
    public void testConvert() throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("compute stat");
        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");

        sparkConf.set("spark.myapp.inputdir", "/Users/busafekete/work/DistOpt/Clkb_data/");
        sparkConf.set("spark.myapp.outputdir", "/Users/busafekete/work/DistOpt/Clkb_data_libsvm/");

        String inputDir = sparkConf.get("spark.myapp.inputdir");
        String outputDir = sparkConf.get("spark.myapp.outputdir");
        ConvertVWToLibSVM.bitsHash = Integer.parseInt(sparkConf.get("spark.myapp.bitshash", "23"));

        ConvertVWToLibSVM.convert(inputDir, outputDir, sparkContext);

    }


}