package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.ml.Learner;
import com.yahoo.labs.yamall.synthetic.helper.LearnerFactory;
import org.apache.spark.SparkConf;
import scala.Tuple2;

import java.util.Properties;

/**
 * Created by busafekete on 8/28/17.
 */
public class SparkLearnerFactory {
    public static String PREFIX = "spark.myapp.";

    public static Learner getLearnerForSpark(SparkConf sparkConf ){
        Properties properties = new Properties();

        for( Tuple2<String,String> prop : sparkConf.getAll() ){
            if (prop._1().contains(PREFIX) ) {
                properties.put( prop._1().replace(PREFIX, ""), prop._2());
            }
        }

        return LearnerFactory.getLearner(properties);
    }

}
