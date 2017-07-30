package com.yahoo.labs.yamall.hadoop.sparkcore;

/**
 * Created by busafekete on 7/27/17.
 */

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.*;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.Partitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;


/**
 * Created by busafekete on 7/25/17.
 */
public class MiniBatchSGDSpark implements Serializable {

    protected String inputDir;
    protected String outputDir;

    protected String logFile = "";

    public static int bitsHash = 22;
    StringBuilder strb = new StringBuilder("");

    protected String method = null;
    protected MBSGD learner = null;

    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;

    protected double cumLoss = 0.0;
    protected long numSamples = 0;
    public static int iter = 100;

    class MBSGD extends SGD {
        public MBSGD(int bits) {
            super(bits);
        }

        public void update(double[] batchGrad) {
            iter++;
            double mul = eta * (1.0 / Math.sqrt(iter));
            for (int i = 0; i < size_hash; i++) w[i] += (mul * batchGrad[i]);
        }

        public double[] getDenseWeights() {
            // useful for extracting w_prev in SVRG
            return w;
        }

    }

    //FileSystem hdfs = null;
    //ArrayList<Path> featureFilePaths = null;
    public MiniBatchSGDSpark() {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");

        this.outputDir = sparkConf.get("spark.myapp.outdir");
        this.inputDir = sparkConf.get("spark.myapp.input");

        this.iter = Integer.parseInt(sparkConf.get("spark.myapp.iter", "10"));

        this.logFile = this.outputDir + "log.txt";

        strb.append("---Input: " + this.inputDir + "\n");
        strb.append("---Output: " + this.outputDir + "\n");
        strb.append("---Iter: " + this.iter + "\n");
    }

    protected void saveLog(int i) throws IOException {
        this.logFile = this.outputDir + "log_" + i + ".txt";
        ResultWriter.writeToHDFS(this.logFile, strb.toString());
    }

    class CustomPartitioner extends Partitioner {

        private int numParts;

        public CustomPartitioner(int i) {
            numParts = i;
        }

        @Override
        public int numPartitions() {
            return numParts;
        }

        @Override
        public int getPartition(Object key) {

            //partition based on the first character of the key...you can have your logic here !!
            return ((String) key).charAt(0) % numParts;

        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof CustomPartitioner) {
                CustomPartitioner partitionerObject = (CustomPartitioner) obj;
                if (partitionerObject.numParts == this.numParts)
                    return true;
            }

            return false;
        }
    }

    class BatchGradObject implements Serializable {
        protected double[] localGbatch;
        protected int size_hash;
        protected double[] localw;
        protected VWParser vwparser = null;
        protected Loss lossFnc = new LogisticLoss();
        public long gatherGradIter = 0;
        public double cumLoss = 0.0;
        protected int bits = 0;
        protected boolean normalizationFlag = false;

//            BatchGradObject( BatchGradObject o ) {
//                bits = o.bits;
//                size_hash = o.size_hash;
//                w = new double[size_hash];
//                localGbatch = new double[size_hash];
//                for (int i=0; i < size_hash; i++ ) {
//                    w[i] = o.w[i];
//                    localGbatch[i]=o.localGbatch[i];
//                }
//
//                vwparser = new VWParser(bits, null, false);
//            }


        BatchGradObject(int b, double[] weights, VWParser p) {
            bits = b;
            size_hash = 1 << bits;
            localw = new double[size_hash];
            for (int i = 0; i < size_hash; i++) localw[i] = weights[i];
            localGbatch = new double[size_hash];
            vwparser = p;
        }

        public double accumulateGradient(String sampleString) {
            gatherGradIter++;
            Instance sample = vwparser.parse(sampleString);

            double pred = predict(sample);

            final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

            pred = Math.min(Math.max(pred, minPrediction), maxPrediction);

            if (Math.abs(grad) > 1e-8) {
                sample.getVector().addScaledSparseVectorToDenseVector(localGbatch, grad);
            }
            cumLoss += lossFnc.lossValue(pred, sample.getLabel()) * sample.getWeight();
            return pred;
        }

        public double predict(Instance sample) {
            return sample.getVector().dot(localw);
        }

        public void aggregate(BatchGradObject obj2) {
            System.out.println("Before Cum loss obj1: " + cumLoss);
            System.out.println("Before Cum loss obj2: " + obj2.cumLoss);

            this.normalizeBatchGradient();
            obj2.normalizeBatchGradient();

            System.out.println("After Cum loss obj1: " + cumLoss);
            System.out.println("After Cum loss obj2: " + obj2.cumLoss);

            double sum = (double) (gatherGradIter + obj2.gatherGradIter);
            if (sum > 0.0) {
                for (int i = 0; i < size_hash; i++)
                    localGbatch[i] = (gatherGradIter * localGbatch[i] + obj2.gatherGradIter * obj2.localGbatch[i]) / sum;
                cumLoss = (gatherGradIter * cumLoss + obj2.gatherGradIter * obj2.cumLoss) / sum;
                gatherGradIter += obj2.gatherGradIter;
            }

            System.out.println("After aggregation Cum loss obj1: " + cumLoss);

        }

        protected void normalizeBatchGradient() {
            if (normalizationFlag == false) {
                if (gatherGradIter > 0) {
                    for (int i = 0; i < size_hash; i++) localGbatch[i] /= (double) gatherGradIter;
                    cumLoss /= (double) gatherGradIter;
                    normalizationFlag = true;
                }
            }
        }

        public double[] getGbatch() {
            return localGbatch;
        }

        public long getNum() {
            return gatherGradIter;
        }

    }

    class CombOp implements Function2<BatchGradObject, BatchGradObject, BatchGradObject> {

        @Override
        public BatchGradObject call(BatchGradObject v1, BatchGradObject v2) throws Exception {
            v1.aggregate(v2);
            v2 = null;
            return v1;
        }
    }

    class SeqOp implements Function2<BatchGradObject, String, BatchGradObject> {

        @Override
        public BatchGradObject call(BatchGradObject v1, String v2) throws Exception {
            v1.accumulateGradient(v2);
            return v1;
        }
    }


    public void train() throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);


        String line = "";

        JavaRDD<String> input = sparkContext.textFile(inputDir);

        double fraction = 1.0 / ((double) iter);
        System.out.println("--Fraction: " + fraction);


        // save example to hdfs
        JavaRDD<String> subsampTrain = input.sample(false, fraction);


        long lineNumGrad = subsampTrain.count();
        line = "--- Number of instances for the gradient step: " + lineNumGrad + "\n";
        strb.append(line);

        long clusterStartTime = System.currentTimeMillis();


        // create learner
        double learningRate = Double.parseDouble(sparkConf.get("spark.myapp.lr", "0.05"));
        double regPar = Double.parseDouble(sparkConf.get("spark.myapp.reg", "0.0"));
        int step = Integer.parseInt(sparkConf.get("spark.myapp.step", "500"));


        strb.append("---SVRG_FR learning rate: " + learningRate + "\n");
        strb.append("---SVRG_FR regularization param: " + regPar + "\n");


        learner = new MBSGD(bitsHash);
        learner.setLearningRate(learningRate);


        Loss lossFnc = new LogisticLoss();
        learner.setLoss(lossFnc);

        saveLog(0);

        int gradSteps = 0;

        VWParser vwparser = new VWParser(bitsHash, null, false);
        //open();

        // optimization
        for (int i = 0; i < iter; i++) {
            line = "--------------------------------------------------------------------\n---> Iter: " + i + "\n";
            strb.append(line);
            saveLog(0);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            // compute gradient
            JavaRDD<String> subsamp = input.sample(false, fraction);

            // compute batch gradient
            double[] prev_w = learner.getDenseWeights();
            BatchGradObject batchgradient = subsamp.treeAggregate(new BatchGradObject(bitsHash, prev_w, vwparser), new SeqOp(), new CombOp(), 11);
            batchgradient.normalizeBatchGradient();

            numSamples += batchgradient.getNum();
            line = "--- Gbatch step: " + batchgradient.gatherGradIter + " Cum loss: " + batchgradient.cumLoss + "\n";
            strb.append(line);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            // update gradient
            learner.update(batchgradient.getGbatch());
            String modelFile = "model_" + i;
            saveModel(this.outputDir, modelFile);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            // info
            long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
            double elapsedTime = clusteringRuntime / 1000.0;
            double elapsedTimeInhours = elapsedTime / 3600.0;

            line = String.format("%d %f %f %f\n", numSamples, 0.0, batchgradient.cumLoss, elapsedTimeInhours);
            strb.append(line);
            System.out.print(this.method + " " + line);

            saveLog(0);
            saveLog(i);
        }
    }

    protected void saveModel(String dir, String fname) throws IOException {
        FileDeleter.delete(new File(dir + fname));
        IOLearner.saveLearner(learner, fname);

        // copy output to HDFS
        FileSystem fileSystem = FileSystem.get(new Configuration());
        fileSystem.moveFromLocalFile(new Path(fname), new Path(dir));

    }



    public static void main(String[] args) throws IOException {
        MiniBatchSGDSpark sl = new MiniBatchSGDSpark();
        sl.train();
    }

}
