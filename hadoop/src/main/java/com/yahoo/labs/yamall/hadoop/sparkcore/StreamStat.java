package com.yahoo.labs.yamall.hadoop.sparkcore;

import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by busafekete on 7/21/17.
 */
public class StreamStat {
    public final static String NUM_OF_FEATURES = "NUMOFFEATURES_";
    public static class Extractor implements FlatMapFunction<String, String> {

        @Override
        public Iterator<String> call(String s) {
            ArrayList<String> array = new ArrayList<>();
            String[] tokens = s.split(" ");
            int numOfFeatures = 0;
            array.add( "LABEL_" + tokens[0]);

            String namespace = "";
            for( int i = 1; i < tokens.length; i++ ){ // skip label
                String tok = tokens[i];

                if ((tok.length()>0) && (tok.charAt(0) == '|')) { // namespace
                    namespace = tok.substring(1).split(":")[0];
                    continue;
                }

                if (namespace=="") continue;  // because of the weight

                String featName = tok.split(":")[0];
                array.add(namespace + "_" + featName);
                numOfFeatures++;
            }
            array.add( NUM_OF_FEATURES + Integer.toString(numOfFeatures));
            return array.iterator();
        }
    }

    public static class FinalFilter implements Function<Tuple2<Integer, String>, Boolean> {
        protected String query1 = "LABEL_1.0";
        protected String query2 = "LABEL_-1.0";

        @Override
        public Boolean call(Tuple2<Integer, String> v) throws Exception {
            String key = v._2();

            if (query1.compareTo(key) == 0) {
                return false;
            } else if ( query2.compareTo(key) == 0 ) {
                return false;
            } else if ( key.contains(NUM_OF_FEATURES) ) {
                return false;
            } else {
                return true;
            }
        }
    }

    public static class FinalFilter2 implements Function<Tuple2<Integer, String>, Boolean> {
        protected String query1 = "LABEL_1.0";
        protected String query2 = "LABEL_-1.0";

        @Override
        public Boolean call(Tuple2<Integer, String> v) throws Exception {
            String key = v._2();

            if (query1.compareTo(key) == 0) {
                return true;
            } else if ( query2.compareTo(key) == 0 ) {
                return true;
            } else {
                return false;
            }
        }
    }


    public static class FeatureFilter implements Function<Tuple2<Integer, String>, Boolean> {

        @Override
        public Boolean call(Tuple2<Integer, String> v) throws Exception {
            String key = v._2();
            return key.contains(NUM_OF_FEATURES);
        }
    }


    public static class FinalFilterNeg extends FinalFilter {
        @Override
        public Boolean call(Tuple2<Integer, String> v) throws Exception {
            return (! super.call(v));
        }
    }


    public static class Stat implements Serializable {
        public void run() throws IOException {
            SparkConf sparkConf = new SparkConf().setAppName("compute basic stat");
            String outDir = sparkConf.get("spark.myapp.outdir");
            JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

            // Read the source file
            String inputString = sparkConf.get("spark.myapp.input");

            JavaRDD<String> input = sparkContext.textFile(inputString);
            StringBuilder result = new StringBuilder("");

            // Gets the number of entries in the RDD
            // long count = input.count();
            // result.append(String.format("Total lines in %s is %d\n", inputString, count));
            // System.out.println(result.toString());


            JavaRDD<String> labels = input.flatMap(new Extractor());
            JavaPairRDD<String, Integer> counts = labels.mapToPair(
                    new PairFunction<String, String, Integer>() {
                        public Tuple2<String, Integer> call(String x) {
                            return new Tuple2(x, 1);
                        }
                    }).reduceByKey(new Function2<Integer, Integer, Integer>() {
                public Integer call(Integer x, Integer y) {
                    return x + y;
                }
            });



            JavaPairRDD<Integer, String> swappedPair = counts.mapToPair(new PairFunction<Tuple2<String, Integer>, Integer, String>() {
                @Override
                public Tuple2<Integer, String> call(Tuple2<String, Integer> item) throws Exception {
                    return item.swap();
                }

            });

            // sort
            JavaPairRDD<Integer, String> sortedPair = swappedPair.sortByKey(false);

            FinalFilter finalfilter = new FinalFilter();
            JavaPairRDD<Integer,String> sortedFilteredPairs = sortedPair.filter(finalfilter);


            long numFeatures = sortedFilteredPairs.count();
            //long numFeatures = counts.count();

            // pos/neg
            FinalFilterNeg finalfilterneg = new FinalFilterNeg();
            JavaPairRDD<Integer,String> posnegandnumfeatures = sortedPair.filter(finalfilterneg);


            JavaPairRDD<Integer,String> features = sortedPair.filter(new FeatureFilter());
            Double avg = this.averageOfOccurence(features);

            //String posnegOutDir = outDir + "posneg";
            //FileDeleter.directoryDeleter(posnegOutDir);
            //posneg.saveAsTextFile(posnegOutDir);
            JavaPairRDD<Integer,String> posneg = posnegandnumfeatures.filter(new FinalFilter2());
            List<Tuple2<Integer,String>> arr = posneg.take(2);
            long pos = 0;
            long neg = 0;
            for(Tuple2<Integer,String> t : arr){
                if(t._2().compareTo("LABEL_1.0")==0) pos = t._1();
                if(t._2().compareTo("LABEL_-1.0")==0) neg = t._1();
            }

            long count = pos + neg;

            result.append(String.format("Total lines in %s is %d\n", inputString, count));
            result.append( String.format( "Number of positives: %d\n", pos ) );
            result.append( String.format( "Number of negatives: %d\n", neg ) );
            result.append( String.format( "Number of features: %d\n", numFeatures ) );
            result.append( String.format( "Average number of features: %f\n", avg ) );

            // output result
            String resultFile = outDir + "results.txt";
            ResultWriter.writeToHDFS(resultFile, result.toString());

            // write the features
            String featOutDir = outDir + "features";
            FileDeleter.directoryDeleter(featOutDir);

            sortedFilteredPairs = sortedFilteredPairs.filter(new Function<Tuple2<Integer, String>, Boolean>() {
                public Boolean call(Tuple2<Integer, String> x) { return (x._1() > 100000); }
            } );

            sortedFilteredPairs.coalesce(5).saveAsTextFile(featOutDir, GzipCodec.class);
            //sortedFilteredPairs.saveAsTextFile(featOutDir);

        }

        public Double averageOfOccurence( JavaPairRDD<Integer,String> features ) {

            JavaRDD<Tuple2<Integer,Double>> values = features.map(new Function<Tuple2<Integer, String>, Tuple2<Integer,Double> >() {
                @Override
                public Tuple2<Integer,Double> call(Tuple2<Integer, String> obj) throws Exception {
                    String numString = obj._2().replace(NUM_OF_FEATURES,"");
                    Integer numOfFeature = Integer.parseInt(numString);
                    Tuple2<Integer,Double> retval = new Tuple2<>(obj._1(), (double)numOfFeature);
                    return retval;
                }
            });
            //values.saveAsTextFile("./resources/spar");
            Tuple2<Integer,Double> average= values.reduce(new Average());
            return average._2();
        }

    }

    public static class Average implements Function2< Tuple2<Integer,Double>, Tuple2<Integer,Double>, Tuple2<Integer,Double>> {
        @Override
        public Tuple2<Integer,Double> call(Tuple2<Integer,Double> a, Tuple2<Integer,Double> b) {
            Double value =  ((a._1() * a._2() + b._1() * b._2())/ ( (double) (a._1() + b._1())) );
            return new Tuple2<Integer,Double>(1, value);
        }
    }


    public static void main( String[] args) throws IOException {
        //testCutLabel();
        //testFinalFilter();


        Stat stat = new Stat();
        stat.run();
    }

    static void testCutLabel() {
        String file = "./resources/train.txt";
        try(BufferedReader br = new BufferedReader(new FileReader(file))) {
            Extractor cl = new Extractor();
            for(String line; (line = br.readLine()) != null; ) {


                // line is not visible here.


                Iterator<String> it = cl.call(line);
                while (it.hasNext()) {
                    System.out.println(it.next());
                }
            }
        } catch (IOException e ){

        }
    }

    static void testFinalFilter() {
        try {
            FinalFilter ff = new FinalFilter();
            String word = "LABEL_-1.0";
            ff.call(new Tuple2<>(100, word));

            word = "ADsdss_121";
            ff.call(new Tuple2<>(100, word));

        } catch ( Exception e ){

        }
    }

}
