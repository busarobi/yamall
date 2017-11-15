package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.ml.Learner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * Created by busafekete on 8/29/17.
 */
public interface LearnerSpark extends Learner {
    public void train(JavaRDD<String> data) throws IOException, ExecutionException, InterruptedException;
    public void setTestRDD(JavaRDD<String> input );
    public JavaPairRDD<Object, Object> getPosteriors(JavaSparkContext sparkContext, String inputDir);
    public void saveModel( String path) throws IOException;
    public static LearnerSpark loadModel( String path) throws IOException { return null; };
}
