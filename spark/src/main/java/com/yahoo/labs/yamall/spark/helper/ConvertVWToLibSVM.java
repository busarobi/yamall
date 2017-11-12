package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.parser.VWParser;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import org.apache.hadoop.io.compress.BZip2Codec;
import org.apache.commons.math3.util.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/**
 * Created by busafekete on 11/6/17.
 */
public class ConvertVWToLibSVM {
    static int bitsHash = 23;


    static class VWTOLibSVMConverter implements Function<String,String> {
        protected VWParser vwParser = null;
        public VWTOLibSVMConverter(int bit){
            vwParser = new VWParser(bit, null, false);
        }

        @Override
        public String call(String v1) throws Exception {
            Instance sample = this.vwParser.parse(v1);

            ArrayList<Pair<Integer,Double>> features = new ArrayList<>();

            for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                double x_i = entry.getDoubleValue();
                Pair<Integer,Double> p = new Pair<>(key, x_i);
                features.add(p);
            }

            Collections.sort(features, new Comparator<Pair<Integer, Double>>() {
                @Override
                public int compare(final Pair<Integer, Double> o1, final Pair<Integer, Double> o2) {
                    return o1.getKey().compareTo(o2.getKey());
                }
            });
            StringBuilder strb = new StringBuilder("");
            Integer label = new Integer((int)((sample.getLabel()+1.0)/2.0));
            strb.append(label.toString());

            for (Pair<Integer,Double> p : features){
                strb.append( " " + (p.getKey()+1) + ":" + p.getValue() );
            }

            return strb.toString();
        }
    }

    public static void convert(String inputDir, String outputDir, JavaSparkContext sparkContext){
        JavaRDD<String> input = sparkContext.textFile(inputDir);
        JavaRDD<String> outputLibSVM = input.map(new VWTOLibSVMConverter(bitsHash));

        FileDeleter.directoryDeleter(outputDir);
        outputLibSVM.saveAsTextFile(outputDir, BZip2Codec.class);
    }

    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("compute stat");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        String inputDir = sparkConf.get("spark.myapp.inputdir");
        String outputDir = sparkConf.get("spark.myapp.outputdir");
        bitsHash = Integer.parseInt(sparkConf.get("spark.myapp.bitshash", "23"));

        convert(inputDir, outputDir, sparkContext);
    }

}
