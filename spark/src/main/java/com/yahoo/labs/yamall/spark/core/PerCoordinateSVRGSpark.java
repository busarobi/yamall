package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.Loss;
import com.yahoo.labs.yamall.ml.PerCoordinateSVRG;
import com.yahoo.labs.yamall.spark.gradient.BatchGradient;
import com.yahoo.labs.yamall.spark.helper.AsyncLocalIterator;
import com.yahoo.labs.yamall.spark.helper.Evaluate;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import com.yahoo.labs.yamall.spark.helper.StringToYamallInstance;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaFutureAction;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Created by busafekete on 8/29/17.
 */
public class PerCoordinateSVRGSpark extends PerCoordinateSVRG implements LearnerSpark {

    protected int bitsHash = 23;
    protected int sparkIter = 1;
    protected String logFile = "log.txt";
    protected StringBuilder strb = new StringBuilder("");
    protected boolean miniBatchSGD = false;
    protected String outputDir = "";
    protected int batchSize = 100000;
    protected int numSGDPartitions = 10;
    protected JavaRDD<String> testRDD = null;

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

    protected JavaRDD<Instance>[] getRDDs(JavaRDD<String> input) {
        long sampleSize = input.count();
        strb.append("--- Input instances: " + sampleSize + "\n");
        JavaRDD<Instance> data = input.map(new StringToYamallInstance(bitsHash));

        // sampleSize \approx batchSize + sparkIter * batchSize + batchSize * ( sparkIter * (sparkIter - 1 ) / 2 )
        // 2 * sampleSize / batchSize \approx 2 +  sparkIter + (sparkIter * sparkIter)
        // 0 \approx ( 2 - 2 * sampleSize / batchSize) +  sparkIter + (sparkIter * sparkIter)
        double solution = (-1.0 + Math.sqrt( 1 - 4 * ( 2 - 2 * sampleSize / batchSize) ) ) / 2.0;
        strb.append( "--- Solution of ( 2 - 2 * sampleSize / batchSize) +  sparkIter + (sparkIter * sparkIter): " + solution + "\n");
        sparkIter = (int) Math.floor(solution);
        strb.append( "--- !!!!!!!!!!! Iter is corrected to " + sparkIter + "\n");
        double[] weights = new double[sparkIter + 1];
        weights[0] = 1.0;
        for(int i = 0; i < sparkIter; i++ ){
            weights[i+1] = i+1.0;
            weights[0] += 1.0;
        }

        JavaRDD<Instance>[] rddArray = data.randomSplit(weights);
        return rddArray;
    }

    class SelectPartition implements Function2<Integer, Iterator<Instance>, Iterator<Instance>>{
        protected int idx = -1;
        public SelectPartition(int idx ){
            this.idx = idx;
        }

        @Override
        public Iterator<Instance> call(Integer ind, Iterator<Instance> iterator) throws Exception {
            if(ind==idx){
                return iterator;
            }else
                return (new ArrayList<Instance>()).iterator();
        }
    };

    @Override
    public void train(JavaRDD<String> input) throws IOException, ExecutionException, InterruptedException {
        String line ="";
        JavaRDD<Instance>[] inputInstancesSplit =  getRDDs(input);
        saveLog();

        long clusterStartTime = System.currentTimeMillis();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // burn in
        strb.append("--- Burn-in starts (" +getBurnInLength() + ")\n");
        int burninSampleSize = 0;
        double burninCumLoss = 0.0;
        List<Instance> inMemorySamples = null;
        JavaRDD<Instance>[] sgdSplit = null;
        JavaFutureAction<List<Instance>> faction = null;
        AsyncLocalIterator sgdIterator = new AsyncLocalIterator(inputInstancesSplit[0], 5);
//        int sgdIter = 0;
        while(sgdIterator.hasNext() && burninSampleSize < this.getBurnInLength()) {
            Instance sample = sgdIterator.next();
            updateFeatureCounts(sample);
            double score = this.updateBurnIn(sample);
            burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
            burninSampleSize++;
        }
//        if (numSGDPartitions<=1) {     //// it simply collects the corresponding RDD
//            inMemorySamples = inputInstancesSplit[0].collect();
//            for (Instance sample : inMemorySamples) {
//                updateFeatureCounts(sample);
//                double score = this.updateBurnIn(sample);
//                burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                burninSampleSize++;
//            }
//        } else {
//            JavaRDD<Instance> currentPartition = inputInstancesSplit[0];
//            double[] weights = new double[numSGDPartitions];
//            for(int ri = 0; ri < numSGDPartitions; ri++ ){
//                weights[ri]=1.0;
//            }
//            sgdSplit = currentPartition.randomSplit(weights);
//            faction = sgdSplit[0].collectAsync();
//            for( int ri = 1; ri < numSGDPartitions; ri ++ ){
//                JavaFutureAction<List<Instance>> nextaction = sgdSplit[ri].collectAsync();
//                inMemorySamples = faction.get();
////                Iterator<Instance> iter = new AsyncLocalIterator(sgdSplit[ri]);
//                strb.append("---+++++++ Number of isntaces: " + inMemorySamples.size() + "\n");
//                saveLog();
//
//                for (Iterator<Instance> iter = new AsyncLocalIterator(sgdSplit[ri]); iter.hasNext();) {
//                    Instance sample = iter.next();
//                    updateFeatureCounts(sample);
//                    double score = this.updateBurnIn(sample);
//                    burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                    burninSampleSize++;
//                }
//                faction = nextaction;
//            }
//
//            inMemorySamples = faction.get();
//            strb.append("---+++++++ Number of isntaces: " + inMemorySamples.size() + "\n");
//            saveLog();
//
//            for (Instance sample : inMemorySamples) {
//                updateFeatureCounts(sample);
//                double score = this.updateBurnIn(sample);
//                burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                burninSampleSize++;
//            }



//            JavaRDD<Instance> currentPartition = inputInstancesSplit[0];
//            List<Partition> lPatitions = currentPartition.partitions();
//            strb.append("---+++ Number of partition in burn-in phase: " + lPatitions.size() + "\n");
//            saveLog();
//            for( Partition p : lPatitions ){
//                // solution 1
//                //int[] idx = {p.index()};
//                //inMemorySamples = currentPartition.collectPartitions(idx)[0];
//                // solution 2
//                //JavaRDD<Instance> partRDD = currentPartition.mapPartitionsWithIndex( new SelectPartition(p.index()), true );
//                //inMemorySamples = partRDD.collect();
//                strb.append("---+++++++ Number of isnaces: " + inMemorySamples.size() + "\n");
//                saveLog();
//
//                for (Instance sample : inMemorySamples) {
//                    updateFeatureCounts(sample);
//                    double score = this.updateBurnIn(sample);
//                    burninCumLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                    burninSampleSize++;
//                }
//            }
//        }
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

            JavaFutureAction<List<Instance>> sgdInmemoryFutureAction = null;
            if (numSGDPartitions<=1) {
                sgdInmemoryFutureAction = inputInstancesSplit[2*i+2].collectAsync();
            } else {
                JavaRDD<Instance> currentPartition = inputInstancesSplit[2*i+2];
                double[] weights = new double[numSGDPartitions];
                for(int ri = 0; ri < numSGDPartitions; ri++ ){
                    weights[ri]=1.0;
                }
                sgdSplit = currentPartition.randomSplit(weights);
                faction = sgdSplit[0].collectAsync();
            }
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // compute gradient
            JavaRDD<Instance> subsamp = inputInstancesSplit[i+1];

            double[] prev_w = this.baseLearner.getDenseWeights();
            BatchGradient.BatchGradientData batchgradient = BatchGradient.computeGradient(subsamp,bitsHash,prev_w);
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
            if (! miniBatchSGD) {
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // grad step

                while(sgdIterator.hasNext() && gradientSampleSize < this.getSGDPhaseLength()) {
                    Instance sample = sgdIterator.next();
                    updateFeatureCounts(sample);
                    double score = this.updateSGDStep(sample);
                    sgdTrainLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
                    gradientSampleSize++;
                }
//                if (numSGDPartitions<=1) {                            //// it simply collects the corresponding RDD
//                    inMemorySamples = sgdInmemoryFutureAction.get();
//                    for (Instance sample : inMemorySamples) {
//                        updateFeatureCounts(sample);
//                        double score = this.updateSGDStep(sample);
//                        sgdTrainLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                        gradientSampleSize++;
//                    }
//                } else {
//                    for( int ri = 1; ri < numSGDPartitions; ri ++ ){
//                        JavaFutureAction<List<Instance>> nextaction = sgdSplit[ri].collectAsync();
//                        inMemorySamples = faction.get();
//                        strb.append("---+++++++ Number of isntaces: " + inMemorySamples.size() + "\n");
//                        saveLog();
//
//                        for (Instance sample : inMemorySamples) {
//                            updateFeatureCounts(sample);
//                            double score = this.updateSGDStep(sample);
//                            sgdTrainLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                            gradientSampleSize++;
//                        }
//                        faction = nextaction;
//                    }
//
//                    inMemorySamples = faction.get();
//                    strb.append("---+++++++ Number of isntaces: " + inMemorySamples.size() + "\n");
//                    saveLog();
//
//                    for (Instance sample : inMemorySamples) {
//                        updateFeatureCounts(sample);
//                        double score = this.updateSGDStep(sample);
//                        sgdTrainLoss += getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//                        gradientSampleSize++;
//                    }
//                }

                endSGDPhase();
                strb.append("--- Gradient phase -- Sample size: " + gradientSampleSize + " Cum. loss: " + (sgdTrainLoss/(double)gradientSampleSize) + "\n");
            }

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
