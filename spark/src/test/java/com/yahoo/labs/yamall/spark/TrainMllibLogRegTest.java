package com.yahoo.labs.yamall.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 11/6/17.
 */
public class TrainMllibLogRegTest {
    @Test
    public void testRun() throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("compute stat");
        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");

        sparkConf.set("executor-memory", "14G");
        sparkConf.set("driver-memory", "14G");

        sparkConf.set("spark.myapp.input", "/Users/busafekete/work/DistOpt/Clkb_data_libsvm/");
        sparkConf.set("spark.myapp.test", "/Users/busafekete/work/DistOpt/Clkb_data_libsvm/");
        sparkConf.set("spark.myapp.outdir", "/Users/busafekete/work/DistOpt/MllibResult/");
        sparkConf.set("spark.myapp.method", "LogisticRegressionWithLBFGS" );
//        sparkConf.set("spark.myapp.input",
//                "/Users/busafekete/work/Clkb/Clkb_variants/FeaturesData/20171030.20170925.20171022.train_lineprint_exp.txt" );
//        sparkConf.set("spark.myapp.test",
//                "/Users/busafekete/work/Clkb/Clkb_variants/FeaturesData/20171030.20170925.20171022.test_lineprint_exp.txt" );

        TrainMllibLogReg.run(sparkConf,sparkContext);
    }

}