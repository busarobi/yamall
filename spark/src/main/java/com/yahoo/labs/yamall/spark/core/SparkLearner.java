package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.ml.Learner;
import org.apache.spark.api.java.JavaRDD;

import java.io.IOException;

/**
 * Created by busafekete on 8/29/17.
 */
public interface SparkLearner extends Learner {
    void train(JavaRDD<String> data) throws IOException;
}
