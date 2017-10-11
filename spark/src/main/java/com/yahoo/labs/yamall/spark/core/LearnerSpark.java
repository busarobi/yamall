package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.ml.Learner;
import org.apache.spark.api.java.JavaRDD;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * Created by busafekete on 8/29/17.
 */
public interface LearnerSpark extends Learner {
    public void train(JavaRDD<String> data) throws IOException, ExecutionException, InterruptedException;
    public void setTestRDD(JavaRDD<String> input );
}
