package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.Loss;
import com.yahoo.labs.yamall.ml.PerCoordinateSVRG;
import com.yahoo.labs.yamall.spark.gradient.BatchGradient;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import com.yahoo.labs.yamall.spark.helper.StringToYamallInstance;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;

import java.io.IOException;
import java.util.List;

/**
 * Created by busafekete on 8/29/17.
 */
public class PerCoordinateSVRGSpark extends PerCoordinateSVRG implements SparkLearner {

    protected int bitsHash = 22;
    protected int sparkIter = 1;
    protected String logFile = "log.txt";
    protected StringBuilder strb = new StringBuilder("");
    protected boolean miniBatchSGD = false;
    protected String outputDir = "";
    protected int batchSize = 10000;

    public void init(SparkConf sparkConf) {
        outputDir = sparkConf.get("spark.myapp.outdir");
        sparkIter = Integer.parseInt(sparkConf.get("spark.myapp.iter"));
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



    void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }

    public void useMiniBatchSGD() {
        this.miniBatchSGD = true;
    }

    public PerCoordinateSVRGSpark(SparkConf sparkConf, StringBuilder strb, int bitsHash) {
        super(bitsHash);

        this.bitsHash = bitsHash;
        batchSize = Integer.parseInt(sparkConf.get("spark.myapp.batchsize", "1000"));
        this.setSGDSize(batchSize);

        this.setLearningRate(Double.parseDouble(sparkConf.get("spark.myapp.lr", "0.5")));
        this.strb = strb;

        init(sparkConf);
    }

    private void endBatchPhase(BatchGradient.BatchGradientData batchgradient) {
        gatherGradientIter = batchgradient.gatherGradIter;
        totalSamplesSeen += batchgradient.gatherGradIter;

        double[] refBatchGrad = batchgradient.getGbatch();
        long[] refFeatureCounts = batchgradient.getFeatureCounts();
        double[] refFeatureMax = batchgradient.getFeatureMax();

        for (int i=0; i<size_hash; i++) {
            lastUpdated[i] = gatherGradientIter;
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
    public void train(JavaRDD<String> input) throws IOException {
        String line ="";
        //input.cache();
        //input.persist();

        long sampleSize = input.count();
        strb.append("--- Input instances: " + sampleSize + "\n");
        long numSamples = 0;


        // TODO: fraction should be set addaptively
        //double fraction = 1.0 / ((double) sampleSize);
        double fraction = (1000.0 * batchSize ) / ((double) sampleSize);
        System.out.println("--- Fraction: " + fraction);
        strb.append("--- Fraction: " + fraction + "\n");


        long clusterStartTime = System.currentTimeMillis();

        //input.persist(StorageLevel.MEMORY_AND_DISK());
        JavaRDD<Instance> inputInstances = input.map(new StringToYamallInstance(bitsHash));
        //inputInstances.persist(StorageLevel.MEMORY_AND_DISK());


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // burn in
        strb.append("--- Burn-in starts (" +getBurnInLength() + ")\n");
        saveLog();
        // slow
        //List<Instance> inMemorySamples = inputInstances.takeSample(true, getBurnInLength());
        double burnInFraction = (1.0 * getBurnInLength() ) / (double) sampleSize;
        numSamples += getBurnInLength();
        double burninCumLoss = 0.0;
        List<Instance> inMemorySamples = inputInstances.sample(true, burnInFraction).collect();
        for(Instance sample : inMemorySamples) {
            double score = this.updateBurnIn(sample);
            burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
        }
        strb.append("--- Burn-in is ready, sample size: " + inMemorySamples.size() + "\n--- cummulative loss: " + (burninCumLoss / (double) inMemorySamples.size())  + "\n" );
        saveLog();
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        for (int i = 0; i < sparkIter; i++) {
            line = "--------------------------------------------------------------------\n---> Iter: " + i + "\n";
            strb.append(line);
            System.out.println(line);
            saveLog();
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // compute gradient
            JavaRDD<Instance> subsamp = inputInstances.sample(false, fraction);

            double[] prev_w = this.baseLearner.getDenseWeights();
            BatchGradient.BatchGradientData batchgradient = BatchGradient.computeGradient(subsamp,bitsHash,prev_w);
            endBatchPhase(batchgradient);

            int ind = checkIsInf(batchgradient.getGbatch());
            if (ind >= 0) {
                line = "--- Infinite value in batch grad vector \n";
                strb.append(line);
                saveLog();
                System.exit(0);
            }
            numSamples += batchgradient.getNum();
            line = "--- Gbatch step: " + batchgradient.gatherGradIter + " Cum loss: " + batchgradient.cumLoss + "\n";
            System.out.println(line);
            strb.append(line);
            saveLog();

            double trainLoss = 0.0;
            if (! miniBatchSGD) {
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // grad step
                int batchSize = getSGDPhaseLength();
                strb.append("--- Gradient phase starts (" + batchSize + ")\n");
                //inMemorySamples = inputInstances.takeSample(true, batchSize);
                double sgdFraction = (1.0 * batchSize ) / (double) sampleSize;
                numSamples += batchSize;

                inMemorySamples = inputInstances.sample(true, sgdFraction).collect();
                for (Instance sample : inMemorySamples) {
                    double score = this.updateSGDStep(sample);
                    trainLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
                }
                trainLoss /= (double) inMemorySamples.size();
                strb.append("--- Gradient phase is ready, sample size: " + inMemorySamples.size() + "\n--- cummulative loss: " + trainLoss  + "\n" );
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // info

            long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
            double elapsedTime = clusteringRuntime / 1000.0;
            double elapsedTimeInhours = elapsedTime / 3600.0;

            line = String.format("%d %f %f %f\n", numSamples, trainLoss, batchgradient.cumLoss, elapsedTimeInhours);
            strb.append(line);
            System.out.print(line);

            saveLog();

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
