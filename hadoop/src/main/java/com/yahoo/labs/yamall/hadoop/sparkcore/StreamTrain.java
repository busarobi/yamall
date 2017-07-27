package com.yahoo.labs.yamall.hadoop.sparkcore;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.*;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.zip.GZIPInputStream;

/**
 * Created by busafekete on 7/21/17.
 */
public class StreamTrain {

//    public static class Extractor implements Function<String, Instance> {
//        VWParser vwparser = null;
//        public Extractor( int bithash ){
//            vwparser = new VWParser(bithash, null, false);
//        }
//
//        @Override
//        public Instance call(String line) throws Exception {
//            Instance sample = vwparser.parse(line);
//            return sample;
//        }
//    }



    static class SparkLearner implements Serializable  {
        protected String inputDir;
        protected String outputDir;

        protected String logFile = "";

        public static int bitsHash = 22;
        StringBuilder strb = new StringBuilder("");

        protected String method = null;
        protected Learner learner = null;

        public static double minPrediction = -50.0;
        public static double maxPrediction = 50.0;

        protected double cumLoss = 0.0;
        protected long numSamples = 0;
        public static int evalPeriod = 5000000;

        public SparkLearner() {
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");

            this.outputDir = sparkConf.get("spark.myapp.outdir");
            this.inputDir = sparkConf.get("spark.myapp.input");

            this.evalPeriod = Integer.parseInt(sparkConf.get("spark.myapp.evalperiod", "5000000"));

            this.logFile = this.outputDir + "log.txt";

            strb.append("---Input: " + this.inputDir + "\n");
            strb.append("---Output: " + this.outputDir + "\n");
            strb.append("---eval period: " + this.evalPeriod + "\n");
        }

        protected void saveLog() throws IOException {
            this.logFile = this.outputDir + "log_" + this.numSamples + "_.txt";
            ResultWriter.writeToHDFS(this.logFile, strb.toString());
        }


        public void train() throws IOException {
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");
            JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

            VWParser vwparser = new VWParser(bitsHash, null, false);

            // create learner
            setLearner();

            long clusterStartTime = System.currentTimeMillis();
            FileSystem hdfs = FileSystem.get(sparkContext.hadoopConfiguration());

            // Get a list of all the files in the inputPath directory. We will read these files one at a time
            //the second boolean parameter here sets the recursion to true
            ArrayList<Path> featureFilePaths = new ArrayList<>();
            RemoteIterator<LocatedFileStatus> fileStatusListIterator = hdfs.listFiles(
                    new Path(this.inputDir ), true);

            while(fileStatusListIterator.hasNext()){
                LocatedFileStatus fileStatus = fileStatusListIterator.next();
                String fileName = fileStatus.getPath().getName();
                if ( fileName.contains(".gz") || fileName.contains(".txt") )
                    featureFilePaths.add(fileStatus.getPath());
            }

            // pick some file for testing
            Collections.shuffle(featureFilePaths);

            strb.append("---Number of files: " + featureFilePaths.size() + "\n");
            ArrayList<Path> featureFilePathsTest = new ArrayList<>();
            for(int i =0; i < 3; i++ ){
                featureFilePathsTest.add(featureFilePaths.remove(featureFilePaths.size()-1));
            }


            for (Path featureFile : featureFilePaths) {
                System.out.println("----- Starting file " + featureFile + " -----");
                strb.append("----- Starting file " + featureFile + " -----\n");

                BufferedReader br = null;
                if (featureFile.getName().contains(".gz"))
                    br = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
                else
                    br = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));

                for(;;) { // forever
                    String strLine = br.readLine();

                    Instance sample;


                    if (strLine != null) {
                        sample = vwparser.parse(strLine);
                    } else
                        break;

                    double score = learner.update(sample);
                    score = Math.min(Math.max(score, minPrediction), maxPrediction);

                    cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();

                    numSamples++;

                    if (numSamples % evalPeriod == 0 ){
                        double trainLoss = cumLoss / (double) numSamples;
                        double testLoss = eval(featureFilePathsTest, hdfs, vwparser);
                        long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
                        double elapsedTime = clusteringRuntime/1000.0;
                        double elapsedTimeInhours = elapsedTime/3600.0;

                        String line = String.format("%d %f %f %f\n", numSamples, trainLoss, testLoss, elapsedTimeInhours);
                        strb.append(line);
                        System.out.print(this.method + " " + line);
                        this.saveLog();

                        String modelFile = "model_" + numSamples;
                        saveModel(this.outputDir, modelFile);
                    }

                }
            }



        }

        public double eval( ArrayList<Path> files, FileSystem hdfs, VWParser vwparser ) throws  IOException {
            int numSamples = 0;
            double score;
            double cumLoss = 0.0;

            for (Path featureFile : files) {
                BufferedReader br = null;
                if (featureFile.getName().contains(".gz"))
                    br = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
                else
                    br = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));

                for(;;) { // forever
                    String strLine = br.readLine();

                    Instance sample;

                    if (strLine != null) {
                        sample = vwparser.parse(strLine);
                    } else
                        break;


                    score = learner.predict(sample);
                    score = Math.min(Math.max(score, minPrediction), maxPrediction);

                    cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();

                    numSamples++;
                }
            }

            return cumLoss / (double) numSamples;
        }

        protected void saveModel( String dir, String fname ) throws IOException {
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
                strb.append( "---SGD_VW learning rate: " + learningRate + "\n");

                learner = new SGD_VW(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method.compareToIgnoreCase("SVRG") == 0) {
                strb.append( "---SVRG learning rate: " + learningRate + "\n");
                strb.append( "---SVRG regularization param: " + regPar + "\n");
                strb.append( "---SVRG step: " + step + "\n");

                SVRG svrg = new SVRG(bitsHash);
                svrg.setLearningRate(learningRate);
                svrg.setRegularizationParameter(regPar);
                svrg.setStep(step);
                //svrg.doAveraging();

                learner = svrg;

            } else if ( this.method.compareToIgnoreCase("SVRG_ADA") == 0) {
                strb.append("---SVRG_ADA learning rate: " + learningRate + "\n");
                strb.append("---SVRG_ADA regularization param: " + regPar + "\n");
                strb.append("---SVRG_ADA step: " + step + "\n");

                SVRG_ADA svrg = new SVRG_ADA(bitsHash);
                svrg.setLearningRate(learningRate);
                svrg.setRegularizationParameter(regPar);
                svrg.setStep(step);
                //svrg.doAveraging();

                learner = svrg;
            } else if ( this.method.compareToIgnoreCase("SVRG_FR") == 0) {
                    strb.append( "---SVRG_FR learning rate: " + learningRate + "\n");
                    strb.append( "---SVRG_FR regularization param: " + regPar + "\n");
                    strb.append( "---SVRG_FR step: " + step + "\n");

                    SVRG_FR svrg = new SVRG_FR(bitsHash);
                    svrg.setLearningRate(learningRate);
                    svrg.setRegularizationParameter(regPar);
                    svrg.setStep(step);
                    //svrg.doAveraging();

                    learner = svrg;
                } else if ( this.method .compareToIgnoreCase("SGD") == 0) {
                strb.append( "---SGD learning rate: " + learningRate + "\n");

                learner = new SGD(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method .compareToIgnoreCase("FREE_REX") == 0) {
                strb.append( "---FREE REX learning rate: " + learningRate + "\n");

                learner = new PerCoordinateFreeRex(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method .compareToIgnoreCase("SOLO") == 0) {
                strb.append( "---SOLO learning rate: " + learningRate + "\n");

                learner = new PerCoordinateSOLO(bitsHash);
                learner.setLearningRate(learningRate);
            } else if ( this.method .compareToIgnoreCase("MB_SGD") == 0) {
                MiniBatchSGD mbsgd = new MiniBatchSGD(bitsHash);

                strb.append( "---MB SGD learning rate: " + learningRate + "\n");
                strb.append( "---MB SGD regularization param: " + regPar + "\n");
                strb.append( "---MB SGD step: " + step + "\n");

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
