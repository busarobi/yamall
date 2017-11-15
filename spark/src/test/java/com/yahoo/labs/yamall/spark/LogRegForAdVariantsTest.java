package com.yahoo.labs.yamall.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 11/15/17.
 */
public class LogRegForAdVariantsTest {
    @Test
    public void testMain() throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("compute stat");
        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");

        sparkConf.set("executor-memory", "14G");
        sparkConf.set("driver-memory", "14G");

//        sparkConf.set("spark.myapp.input", "/Users/busafekete/work/DistOpt/Clkb_data_libsvm/");
//        sparkConf.set("spark.myapp.test", "/Users/busafekete/work/DistOpt/Clkb_data_libsvm/");

        sparkConf.set("spark.myapp.reg", "0.0000000000001");
        sparkConf.set("spark.myapp.iter", "10");
        sparkConf.set("spark.myapp.method", "LogisticRegressionWithLBFGS" );
        sparkConf.set("spark.myapp.input",
                "/Users/busafekete/work/Clkb/Clkb_variants/FeaturesData/20171112.20171008.20171104.train_lineprint_exp_comb_nodomain.txt" );
        sparkConf.set("spark.myapp.test",
                "/Users/busafekete/work/Clkb/Clkb_variants/FeaturesData/20171112.20171008.20171104.test_lineprint_exp_comb_nodomain.txt" );

        sparkConf.set("spark.myapp.outdir", "/Users/busafekete/work/DistOpt/MllibResult/");

        LogRegForAdVariants.run(sparkConf,sparkContext);
    }

}