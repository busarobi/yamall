package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by busafekete on 11/6/17.
 */
public class PrepareStat {
    static class InstanceToNumberOfFeatures implements DoubleFunction<String> {
        protected int getNumberOfFeatures(String namespace){
            int numberOfFeatures = 0;
            String[] features = namespace.split(" ");
            for(String f : features){
                f = f.trim();
                if (f.length()>0) {
                    if (f.charAt(0) != '|')
                        numberOfFeatures++;
                }
            }
            return numberOfFeatures;
        }

        @Override
        public double call(String v1) throws Exception {
            String[] namespaces = v1.split("\\|");
            int numberOfFeatures = 0;
            for(int i=1; i < namespaces.length; i++ ){
                numberOfFeatures += getNumberOfFeatures(namespaces[i]);
            }
            return (double) numberOfFeatures;
        }
    }

    static class InstanceToFeatures implements FlatMapFunction<String,String> {
        protected ArrayList<String> getFeatures(String namespace){
            ArrayList<String> featureNames = new ArrayList<String>();
            String[] features = namespace.split(" ");
            for(String f : features){
                f = f.trim();
                if (f.length()>0) {
                    if (f.charAt(0) != '|') {
                        String fname = f.split(":")[0];
                        featureNames.add(fname);
                    }
                }
            }
            return featureNames;
        }

        @Override
        public Iterator<String> call(String s) throws Exception {
            String[] namespaces = s.split("\\|");
            ArrayList<String> features = new ArrayList<String>();
            for(int i=1; i < namespaces.length; i++ ){
                features.addAll(getFeatures(namespaces[i]));
            }
            return features.iterator();
        }
    }


    public static void computeStat( StringBuilder strb, JavaSparkContext sparkContext, String inputDir ){
        JavaRDD<String> input = sparkContext.textFile(inputDir );

        long n = input.count();
        strb.append( "Number of instances: " + n + "\n");

        // numbr of features
        JavaRDD<String> features = input.flatMap(new InstanceToFeatures());
        JavaPairRDD<String,Long> featureswithoccurence = features.mapToPair(s -> new Tuple2<>(s,1L))
                .reduceByKey( ( occ1, occ2) -> occ1 + occ2 );

        long numberOfFeatures = featureswithoccurence.count();
        strb.append( "Number of features: " + numberOfFeatures + "\n");


        // compute label distribution
        JavaDoubleRDD labels = input.mapToDouble(s -> Double.parseDouble(s.split(" ")[0]));
        Tuple2<double[], long[]> buckets = labels.histogram(2);

        strb.append( "Number of negatives: " + buckets._2()[0] + "\n");
        strb.append( "Number of positives: " + buckets._2()[1] + "\n");

        strb.append( "Percentage of negatives: " + (100.0*(buckets._2()[0]/(double) n)) + "\n");
        strb.append( "Percentage of positives: " + (100.0*(buckets._2()[1]/(double) n)) + "\n");


        // average number of features
        JavaDoubleRDD numOfFeatures = input.mapToDouble(new InstanceToNumberOfFeatures());
        double avgFeatureNumber = numOfFeatures.mean();
        strb.append( "Avg. number of features: " + avgFeatureNumber + "\n");

    }

    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("compute stat");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        String inputDir = sparkConf.get("spark.myapp.inputdir");
        String output = sparkConf.get("spark.myapp.output");

        StringBuilder strb = new StringBuilder("");
        strb.append( "Input: " + inputDir + "\n" );
        strb.append( "Output: " + output + "\n" );

        System.out.println( strb.toString() + "\n" );

        FileWriterToHDFS.writeToHDFS( output, strb.toString());

        computeStat(strb, sparkContext, inputDir);

        System.out.println( strb.toString());
        FileWriterToHDFS.writeToHDFS( output, strb.toString());
    }

}
