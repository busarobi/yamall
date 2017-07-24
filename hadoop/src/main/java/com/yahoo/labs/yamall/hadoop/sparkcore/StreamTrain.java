package com.yahoo.labs.yamall.hadoop.sparkcore;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.*;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Iterator;

/**
 * Created by busafekete on 7/21/17.
 */
public class StreamTrain {

    public static class Extractor implements Function<String, Instance> {
        VWParser vwparser = null;
        public Extractor( int bithash ){
            vwparser = new VWParser(bithash, null, false);
        }

        @Override
        public Instance call(String line) throws Exception {
            Instance sample = vwparser.parse(line);
            return sample;
        }
    }



    static class SparkLearner implements Serializable  {
        protected String inputDir;
        protected String outputDir;

        protected String logFile = "";

        public static int bitsHash = 22;
        public long N = 0;
        StringBuilder strb = new StringBuilder("");

        protected int mainloops = 10;
        protected int batchsize = 10000;
        protected int testsize = 1000;

        protected String method = null;
        protected Learner learner = null;

        public static double minPrediction = -50.0;
        public static double maxPrediction = 50.0;

        protected double cumLoss = 0.0;
        protected long numSamples = 0;


        public SparkLearner() {
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");

            this.outputDir = sparkConf.get("spark.myapp.outdir");
            this.inputDir = sparkConf.get("spark.myapp.input");

            this.mainloops = Integer.parseInt(sparkConf.get("spark.myapp.mainloops", "10"));
            this.batchsize = Integer.parseInt(sparkConf.get("spark.myapp.batchsize", "10000"));
            this.testsize = Integer.parseInt(sparkConf.get("spark.myapp.testsize", "1000"));

            this.logFile = this.outputDir + "log.txt";

            strb.append("Input: " + this.inputDir + "\n");
            strb.append("Output: " + this.outputDir + "\n");
            strb.append("main loops: " + this.mainloops + "\n");
            strb.append("batch size: " + this.batchsize + "\n");
            strb.append("test size: " + this.testsize + "\n");
        }

        protected void saveLog() throws IOException {
            ResultWriter.writeToHDFS(this.logFile, strb.toString());
        }

        public String getReport() {
            double trainLoss = cumLoss / (double) numSamples;
            String line = String.format("Train size, %d,Train loss, %f\n\n", numSamples, trainLoss );
            return line;
        }


        public void train() throws IOException {
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");
            JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
            JavaRDD<String> input = sparkContext.textFile(inputDir);

            // compute lines
            N = input.count();
            double samplingfraction = (double) batchsize / (double) N;
            double samplingfractiontest = (double) testsize / (double) N;
            strb.append( String.format("Sampling fraction: %f\n", samplingfraction ) );

            // create learner
            setLearner();

            JavaRDD<Instance> data = input.map(new Extractor(bitsHash));
            this.saveLog();

            for( int maini = 0; maini < this.mainloops; maini++){
                // train
                JavaRDD<Instance> batch = data.sample(false,samplingfraction);
                strb.append( "Fraction: " + batch.count() + "\n");
                Iterator<Instance> batchIter = batch.toLocalIterator();

                while(batchIter.hasNext()){
                    Instance instance = batchIter.next();
                    double score = learner.update(instance);
                    score = Math.min(Math.max(score, minPrediction), maxPrediction);
                    cumLoss += learner.getLoss().lossValue(score, instance.getLabel()) * instance.getWeight();
                    numSamples++;
                }

                strb.append(getReport());
                String modelFileName = String.format( "model_%d", maini );
                saveModel(this.outputDir, modelFileName );

                // test
                JavaRDD<Instance> batchtest = data.sample(false,samplingfractiontest);
                strb.append( "Fraction (test): " + batchtest.count() + "\n");
                Iterator<Instance> batchIterTest = batch.toLocalIterator();

                double testCumLoss = 0.0;
                int testNum = 0;
                while(batchIterTest.hasNext()){
                    Instance instance = batchIterTest.next();
                    double score = learner.predict(instance);
                    score = Math.min(Math.max(score, minPrediction), maxPrediction);

                    testCumLoss += learner.getLoss().lossValue(score, instance.getLabel()) * instance.getWeight();
                    testNum++;
                }

                double testLoss = testCumLoss / (double) testNum;
                String tmpline = String.format("Test size, %d,Test loss, %f\n", testNum, testLoss );
                strb.append(tmpline);


                this.saveLog();
            }

        }

        public void saveModel( String dir, String fname ) throws IOException {
            FileDeleter.delete(new File(dir + fname));
            IOLearner.saveLearner(learner, fname);

            // copy output to HDFS
            FileSystem fileSystem = FileSystem.get(new Configuration());
            fileSystem.moveFromLocalFile(new Path(fname), new Path(dir));

        }


        public void setLearner(){
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");
            this.method = sparkConf.get("spark.myapp.method", "sgd_vw");

            double learningRate = Double.parseDouble(sparkConf.get("spark.myapp.lr", "0.05"));
            double regPar = Double.parseDouble(sparkConf.get("spark.myapp.reg", "0.0"));
            int step = Integer.parseInt(sparkConf.get("spark.myapp.step", "500"));


            System.out.println( "----> Method: " + this.method + "\n" );

            if ( this.method.compareToIgnoreCase("SGD_VW") == 0) {
                strb.append( "SGD_VW learning rate: " + learningRate + "\n");

                learner = new SGD_VW(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method.compareToIgnoreCase("SVRG") == 0) {
                strb.append( "SVRG learning rate: " + learningRate + "\n");
                strb.append( "SVRG regularization param: " + regPar + "\n");
                strb.append( "SVRG step: " + step + "\n");

                SVRG svrg = new SVRG(bitsHash);
                svrg.setLearningRate(learningRate);
                svrg.setRegularizationParameter(regPar);
                svrg.setStep(step);
                //svrg.doAveraging();

                learner = svrg;
            } else if ( this.method.compareToIgnoreCase("SVRG_ADA") == 0) {
                strb.append( "SVRG_ADA learning rate: " + learningRate + "\n");
                strb.append( "SVRG_ADA regularization param: " + regPar + "\n");
                strb.append( "SVRG_ADA step: " + step + "\n");

                SVRG_ADA svrg = new SVRG_ADA(bitsHash);
                svrg.setLearningRate(learningRate);
                svrg.setRegularizationParameter(regPar);
                svrg.setStep(step);
                //svrg.doAveraging();

                learner = svrg;
            } else if ( this.method .compareToIgnoreCase("SGD") == 0) {
                strb.append( "SGD learning rate: " + learningRate + "\n");

                learner = new SGD(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method .compareToIgnoreCase("FREE_REX") == 0) {
                strb.append( "FREE REX learning rate: " + learningRate + "\n");

                learner = new PerCoordinateFreeRex(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method .compareToIgnoreCase("SOLO") == 0) {
                strb.append( "SOLO learning rate: " + learningRate + "\n");

                learner = new PerCoordinateSOLO(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method .compareToIgnoreCase("MB_SGD") == 0) {
                MiniBatchSGD mbsgd = new MiniBatchSGD(bitsHash);

                strb.append( "MB SGD learning rate: " + learningRate + "\n");
                strb.append( "MB SGD regularization param: " + regPar + "\n");
                strb.append( "MB SGD step: " + step + "\n");

                mbsgd.setLearningRate(learningRate);
                mbsgd.setRegularizationParameter(regPar);
                mbsgd.setStep(step);
                learner = mbsgd;
            }

            Loss lossFnc = new LogisticLoss();
            learner.setLoss(lossFnc);

        }

    }

    public static void main(String[] args) throws IOException {
        SparkLearner sl = new SparkLearner();
        sl.train();
    }

}
