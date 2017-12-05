package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.Loss;
import com.yahoo.labs.yamall.ml.PerCoordinateSVRG;
import com.yahoo.labs.yamall.spark.gradient.BatchGradient;
import com.yahoo.labs.yamall.spark.helper.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaFutureAction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Created by busafekete on 8/29/17.
 */
public class PerCoordinateSVRGSparkBoosted extends PerCoordinateSVRG implements LearnerSpark {

    protected int bitsHash = 23;
    protected int sparkIter = 1;
    protected String logFile = "log.txt";
    protected StringBuilder strb = new StringBuilder("");
    protected boolean miniBatchSGD = false;
    protected String outputDir = "";
    protected int batchSize = 100000;
    protected int numSGDPartitions = 10;
    protected JavaRDD<String> testRDD = null;
    protected JavaRDD<Instance> data = null;

    public void init(SparkConf sparkConf) {
        outputDir = sparkConf.get("spark.myapp.outdir");
        // obsolete parameter, it gets updated based on batchSize
        sparkIter = Integer.parseInt(sparkConf.get("spark.myapp.iter", "10"));
        numSGDPartitions = Integer.parseInt(sparkConf.get("spark.myapp.sgdpartition", "10"));
        logFile = outputDir + "/log.txt";

        strb.append("---++++++++ Learner report\n");
        strb.append("--- SVRG_FR\n");
        strb.append("--- Output: " + outputDir + "\n");
        strb.append("--- Log file: " + logFile + "\n");

        strb.append("--- Iter: " + sparkIter + "\n");
        strb.append("--- Batch size: " + batchSize + "\n");
        strb.append("--- Bits hash: " + bitsHash + "\n");
        strb.append("--- SVRG_FR learning rate: " + this.eta + "\n");

        Loss lossFnc = new LogisticLoss();
        this.setLoss(lossFnc);


        System.out.println(strb.toString());
    }

    public void setTestRDD(JavaRDD<String> inputTest ){
            this.testRDD = inputTest;
    }

    @Override
    public JavaPairRDD<Object, Object> getPosteriors(JavaSparkContext sparkContext, String inputDir) {
        JavaRDD<String> input = sparkContext.textFile(inputDir );
        JavaPairRDD<String, Tuple2> posteriorsAndLables = input.mapToPair(new PosteriorComputer(this, bitsHash));
        JavaPairRDD<Object, Object> predictionAndLabels = posteriorsAndLables.values().mapToPair((PairFunction<Tuple2, Object, Object>) tup -> new Tuple2<>(tup._1(),tup._2()));
        return predictionAndLabels;
    }

    @Override
    public void saveModel(String path) throws IOException {
        ModelSerializationToHDFS.saveModel(outputDir, this);
    }

    void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }

    public void useMiniBatchSGD() {
        this.miniBatchSGD = true;
    }

    public PerCoordinateSVRGSparkBoosted(SparkConf sparkConf, StringBuilder strb, int bitsHash) {
        super(bitsHash);

        this.bitsHash = bitsHash;
        batchSize = Integer.parseInt(sparkConf.get("spark.myapp.batchsize", "1000"));
        this.setSGDSize(batchSize);

        this.setLearningRate(Double.parseDouble(sparkConf.get("spark.myapp.lr", "0.5")));
        this.strb = strb;

        init(sparkConf);
    }

    private void endBatchPhaseSpark(BatchGradient.BatchGradientData batchgradient) {
        gatherGradientIter = batchgradient.gatherGradIter;
        totalSamplesSeen += batchgradient.gatherGradIter;

        double[] refBatchGrad = batchgradient.getGbatch();
        long[] refFeatureCounts = batchgradient.getFeatureCounts();
        double[] refFeatureMax = batchgradient.getFeatureMax();

        for (int i=0; i<size_hash; i++) {
            lastUpdated[i] = gatherGradientIter;
            //negativeBatchGradient[i] = refBatchGrad[i]/gatherGradientIter;
            negativeBatchGradient[i] = refBatchGrad[i];

            featureScalings[i] = Math.max(featureScalings[i],refFeatureMax[i]);
            featureCounts[i] += refFeatureCounts[i];
        }
        SGDIter = 0;

        if (doUseReset) {
            baseLearner.setCenter(w_previous);
            baseLearner.reset();
        }

        // perform one full batch grad step. Also update feature scalings.
        baseLearner.updateFromNegativeGrad(SparseVector.dense2Sparse(featureScalings),  SparseVector.dense2Sparse(negativeBatchGradient));
    }

    @Override
    public void train(JavaRDD<String> input) throws IOException, ExecutionException, InterruptedException {
        String line ="";
        saveLog();

        long clusterStartTime = System.currentTimeMillis();

        //
        long sampleSize = input.count();
        strb.append("--- Input instances: " + sampleSize + "\n");
        data = input.map(new StringToYamallInstance(bitsHash));
        data.persist(StorageLevel.MEMORY_AND_DISK());
        double sgdsamplingRatio = (double) batchSize / (double)sampleSize;


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // burn in
        strb.append("--- Burn-in starts (" +getBurnInLength() + ")\n");
        int burninSampleSize = 0;
        double burninCumLoss = 0.0;
        List<Instance> inMemorySamples = null;
        JavaRDD<Instance>[] sgdSplit = null;
        JavaFutureAction<List<Instance>> faction = null;

        inMemorySamples = data.sample(false,sgdsamplingRatio).collect();
        for (Instance sample : inMemorySamples) {
            updateFeatureCounts(sample);
            double score = this.updateBurnIn(sample);
            burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
            burninSampleSize++;
        }


        endBurnInPhase();

        double trainLoss = (burninCumLoss / (double) burninSampleSize);
        long numSamples = burninSampleSize;
        strb.append("--- Burn-in is ready, sample size: " + burninSampleSize + "\n--- cummulative loss: " + trainLoss  + "\n" );
        saveLog();
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        for (int i = 0; i < sparkIter; i++) {
            line = "--------------------------------------------------------------------\n---> Iter: " + i + "\n";
            strb.append(line);
            System.out.println(line);
            saveLog();

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // compute gradient
            double[] prev_w = this.baseLearner.getDenseWeights();

            // JavaRDD<Instance> subsamp = inputInstancesSplit[2*i+1];
            // BatchGradient.BatchGradientData batchgradient = BatchGradient.computeGradient(subsamp,bitsHash,prev_w);
            BatchGradient.BatchGradientData batchgradient = BatchGradient.computeGradient(data,bitsHash,prev_w);

            endBatchPhaseSpark(batchgradient);

            int ind = checkIsInf(batchgradient.getGbatch());
            if (ind >= 0) {
                line = "--- Infinite value in batch grad vector \n";
                strb.append(line);
                saveLog();
                System.exit(0);
            }
            trainLoss = ( numSamples * trainLoss + batchgradient.cumLoss * batchgradient.gatherGradIter ) / ((double) numSamples + batchgradient.gatherGradIter);
            numSamples += batchgradient.gatherGradIter;
            line = "--- Batch step     -- Sample size: " + batchgradient.gatherGradIter + " Cum. loss: " + batchgradient.cumLoss + "\n";
            System.out.println(line);
            strb.append(line);
            saveLog();

            double sgdTrainLoss = 0.0;
            int gradientSampleSize = 0;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // grad step

            inMemorySamples = data.sample(false,sgdsamplingRatio).collect();
            for (Instance sample : inMemorySamples) {
                updateFeatureCounts(sample);
                double score = this.updateSGDStep(sample);
                sgdTrainLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
                gradientSampleSize++;
            }
            endSGDPhase();
            strb.append("--- Gradient phase -- Sample size: " + gradientSampleSize + " Cum. loss: " + (sgdTrainLoss/(double)gradientSampleSize) + "\n");


            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // info
            trainLoss = (numSamples*trainLoss+sgdTrainLoss)/((double)numSamples+gradientSampleSize);
            numSamples += gradientSampleSize;

            long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
            double elapsedTime = clusteringRuntime / 1000.0;
            double elapsedTimeInhours = elapsedTime / 3600.0;

            line = String.format("--- Num of samples: %d\tTrain loss: %f\tElapsed time: %f\n", numSamples, trainLoss, elapsedTimeInhours);
            strb.append(line);
            System.out.print(line);
            saveLog();

            if (testRDD != null ) {
                double testLoss = Evaluate.getLoss(this.testRDD, this.baseLearner, this.bitsHash);
                line = String.format("%d %f %f %f\n", numSamples, trainLoss, testLoss, elapsedTimeInhours);
                strb.append(line);
                saveLog();
            }


            //////////////////////////////////////////////////////////////////////////////////////////////////////////
        }

    }


    protected static int checkIsInf(double[] arr) {
        int retVal = -1;
        for (int i = 0; i < arr.length; i++) {
            if (Double.isInfinite(arr[i])) {
                retVal = i;
                break;
            }
        }
        return retVal;
    }
}
