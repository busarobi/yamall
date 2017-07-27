package com.yahoo.labs.yamall.hadoop.sparkcore;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.IOLearner;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.Loss;
import com.yahoo.labs.yamall.ml.SVRG_FR;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.Partitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Random;

/**
 * Created by busafekete on 7/25/17.
 */
public class DistTrain {
    static class DistTrainRunner implements Serializable {
        protected String inputDir;
        protected String outputDir;

        protected String logFile = "";

        public static int bitsHash = 22;
        StringBuilder strb = new StringBuilder("");

        protected String method = null;
        protected SVRG_FR learner = null;

        public static double minPrediction = -50.0;
        public static double maxPrediction = 50.0;

        protected double cumLoss = 0.0;
        protected long numSamples = 0;
        public static int iter = 100;

        //FileSystem hdfs = null;
        //ArrayList<Path> featureFilePaths = null;
        public DistTrainRunner() {
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");

            this.outputDir = sparkConf.get("spark.myapp.outdir");
            this.inputDir = sparkConf.get("spark.myapp.input");

            this.iter = Integer.parseInt(sparkConf.get("spark.myapp.iter", "10"));

            this.logFile = this.outputDir + "log.txt";

            strb.append("---Input: " + this.inputDir + "\n");
            strb.append("---Output: " + this.outputDir + "\n");
            strb.append("---Iter: " + this.iter + "\n");
        }

        protected void saveLog( int i) throws IOException {
            this.logFile = this.outputDir + "log_" + i + ".txt";
            ResultWriter.writeToHDFS(this.logFile, strb.toString());
        }

        class CustomPartitioner extends Partitioner{

            private int numParts;

            public CustomPartitioner(int i) {
                numParts=i;
            }

            @Override
            public int numPartitions()
            {
                return numParts;
            }

            @Override
            public int getPartition(Object key){

                //partition based on the first character of the key...you can have your logic here !!
                return ((String)key).charAt(0)%numParts;

            }

            @Override
            public boolean equals(Object obj){
                if(obj instanceof CustomPartitioner)
                {
                    CustomPartitioner partitionerObject = (CustomPartitioner)obj;
                    if(partitionerObject.numParts == this.numParts)
                        return true;
                }

                return false;
            }
        }



        class Extractor implements PairFunction<String,Integer,Instance> {
            Random r = new Random();
            protected int partition = 0;
            VWParser vwparser = null;

            public Extractor(int hahsize, int partition){
                vwparser = new VWParser(hahsize, null, false);
            }
            @Override
            public Tuple2<Integer,Instance> call(String arg0) throws Exception {
                //return a tuple ,split[0] contains continent and split[1] contains country
                return new Tuple2<Integer,Instance>(r.nextInt(partition), vwparser.parse(arg0));
            }
        }

        class BatchGradObject implements Serializable {
            protected double[] localGbatch;
            protected int size_hash;
            protected double[] localw;
            protected VWParser vwparser = null;
            protected Loss lossFnc = new LogisticLoss();
            public long gatherGradIter=0;
            public double cumLoss = 0.0;
            protected int bits = 0;

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


            BatchGradObject( int b, double[] weights, VWParser p) {
                bits = b;
                size_hash = 1 << bits;
                localw = new double[size_hash];
                for (int i=0; i < size_hash; i++ ) localw[i] = weights[i];
                localGbatch = new double[size_hash];
                vwparser = p;
            }

            public double accumulateGradient( String sampleString ) {
                gatherGradIter++;
                Instance sample = vwparser.parse(sampleString);

                double pred = predict(sample);

                final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

                if (Math.abs(grad) > 1e-8) {
                    sample.getVector().addScaledSparseVectorToDenseVector(localGbatch, grad);
                }
                cumLoss += lossFnc.lossValue(pred, sample.getLabel()) * sample.getWeight();
                return pred;
            }

            public double predict(Instance sample) {
                return sample.getVector().dot(localw);
            }

            public void aggregate(BatchGradObject obj2){
                for (int i=0; i < size_hash; i++ ) localGbatch[i] += obj2.localGbatch[i];
                gatherGradIter += obj2.gatherGradIter;
                cumLoss += obj2.cumLoss;
            }

            protected void normalizeBatchGradient() {
                for (int i=0; i < size_hash; i++ ) localGbatch[i] /= (double)gatherGradIter;
                cumLoss /= (double) gatherGradIter;
            }
            public double[] getGbatch(){ return localGbatch; }
        }

        class CombOp implements Function2<BatchGradObject,BatchGradObject,BatchGradObject> {

            @Override
            public BatchGradObject call(BatchGradObject v1, BatchGradObject v2) throws Exception {
                v1.aggregate(v2);
                return v1;
            }
        }

        class SeqOp implements Function2<BatchGradObject,String,BatchGradObject> {

            @Override
            public BatchGradObject call(BatchGradObject v1, String v2) throws Exception {
                v1.accumulateGradient(v2);
                return v1;
            }
        }

        public void train() throws IOException {
            SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");
            JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

            // get file list
            //hdfs = FileSystem.get(sparkContext.hadoopConfiguration());

            // Get a list of all the files in the inputPath directory. We will read these files one at a time
            //the second boolean parameter here sets the recursion to true
//            featureFilePaths = new ArrayList<>();
//            RemoteIterator<LocatedFileStatus> fileStatusListIterator = hdfs.listFiles(
//                    new Path(this.outputDir + "/backup" ), true);
//
//            while(fileStatusListIterator.hasNext()){
//                LocatedFileStatus fileStatus = fileStatusListIterator.next();
//                String fileName = fileStatus.getPath().getName();
//                if ( fileName.contains(".gz") || fileName.contains(".txt") )
//                    featureFilePaths.add(fileStatus.getPath());
//            }
//
//            // pick some file for testing
//            Collections.shuffle(featureFilePaths);
//
//            strb.append("---Number of files: " + featureFilePaths.size() + "\n");
//            ArrayList<Path> featureFilePathsTest = new ArrayList<>();
//            for(int i =0; i < 3; i++ ){
//                featureFilePathsTest.add(featureFilePaths.remove(featureFilePaths.size()-1));
//            }



            JavaRDD<String> input = sparkContext.textFile(inputDir);
            long lineNum = input.count();

            String line = "--- Number of training instance: " + lineNum + "\n";
            System.out.println(line);
            strb.append(line);

            double fraction = 1.0 /(iter+1.0);
            System.out.println("--Fraction: " + fraction );

            //input.cache();
            //input.persist();


            // save example to hdfs
            JavaRDD<String> subsampTrain = input.sample(false,fraction);
            //JavaRDD<String> subsampTest = input.sample(false,fraction);
            //DataFrame wordsDataFrame = spark.createDataFrame(subsamp, String.class);
            long lineNumGrad = subsampTrain.count();
            line = "--- Number of instances for the gradient step: " + lineNumGrad + "\n";
            strb.append( line );

//            String dirForGradStep = this.outputDir + "grad";
//            hdfs.delete(new Path(dirForGradStep), true);
//            subsamp.saveAsTextFile( dirForGradStep, GzipCodec.class );

            // save data for test
//            subsamp = input.sample(false,fraction);
//            lineNumGrad = subsamp.count();
//            line = "--- Number of test instance: " + lineNumGrad + "\n";
//            strb.append( line );

//            String testDir = this.outputDir + "test";
//            hdfs.delete(new Path(testDir), true);
//            subsamp.saveAsTextFile( testDir, GzipCodec.class );


            long clusterStartTime = System.currentTimeMillis();



            // create learner
            double learningRate = Double.parseDouble(sparkConf.get("spark.myapp.lr", "0.05"));
            double regPar = Double.parseDouble(sparkConf.get("spark.myapp.reg", "0.0"));
            int step = Integer.parseInt(sparkConf.get("spark.myapp.step", "500"));

            //int lineNumGrad = (int)(lineNum / (iter + 2.0));
            //int stepPerGrad = (int)(lineNumGrad/(double)iter);
            //int sqrtbathcsize =(int) Math.sqrt((double)lineNumGrad);
            //stepPerGrad = Math.min( stepPerGrad, sqrtbathcsize);
            //int stepPerGrad =(int) Math.sqrt((double)lineNumGrad);
            int stepPerGrad =(int) (lineNumGrad/ (10.0));

            strb.append( "---SVRG_FR learning rate: " + learningRate + "\n");
            strb.append( "---SVRG_FR regularization param: " + regPar + "\n");
            strb.append( "---SVRG_FR step: " + stepPerGrad + "\n");

            learner = new SVRG_FR(bitsHash);
            learner.setLearningRate(learningRate);
            learner.setRegularizationParameter(regPar);
            learner.setStep(stepPerGrad);
            Loss lossFnc = new LogisticLoss();
            learner.setLoss(lossFnc);

            saveLog(0);


            VWParser vwparser = new VWParser(bitsHash, null, false);
            //open();

            // optimization
            for( int i=0; i < iter; i++) {
                line = "---> Iter: " + i + "\n";
                strb.append( line );

                double samplingRatio = (stepPerGrad / (double) lineNumGrad);
                line = "--- Inner sampling ratio: " + samplingRatio + "\n";
                strb.append( line );

                JavaRDD<String> s = subsampTrain.sample(false, samplingRatio);
                List<String> samples = s.collect();
                line = "--- Inner training size: " + samples.size() + "\n";
                strb.append( line );

                saveLog(0);

                for(String strInstance : samples) {
                    //String strInstance = getLine();
                    //System.out.println(strInstance);
                    Instance instance = vwparser.parse(strInstance);
                    double score;
                    if (i==0) {
                        score = learner.freeRexUpdate(instance);
                    } else {
                        score = learner.gradStep(instance);
                    }
                    score = Math.min(Math.max(score, minPrediction), maxPrediction);

                    cumLoss += learner.getLoss().lossValue(score, instance.getLabel()) * instance.getWeight();
                    numSamples++;

                }

                learner.initGatherState();
                //
                double trainLoss = cumLoss / (double) numSamples;
                //double testLoss = eval(subsampTest, vwparser);
                //double testLoss = 0.0;
                String modelFile = "model_" + i;
                saveModel(this.outputDir, modelFile);

                saveLog(0);

                // compute gradient
                JavaRDD<String> subsamp  = input.sample(false, fraction);

                // compute batch gradient
                double[] prev_w = learner.getDenseWeights();
                BatchGradObject batchgradient = subsamp.treeAggregate( new BatchGradObject(bitsHash, prev_w, vwparser), new SeqOp(), new CombOp());
                batchgradient.normalizeBatchGradient();

                // set Gbatch to learner
                learner.setGBatch(batchgradient.getGbatch());

                line = "--- Gbatch step: " + batchgradient.gatherGradIter + " Cum loss: " + batchgradient.cumLoss + "\n";
                strb.append( line );

                long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
                double elapsedTime = clusteringRuntime/1000.0;
                double elapsedTimeInhours = elapsedTime/3600.0;

                line = String.format("%d %f %f %f\n", numSamples, trainLoss, batchgradient.cumLoss, elapsedTimeInhours);
                strb.append(line);
                System.out.print(this.method + " " + line);


                saveLog(0);
                saveLog(i);
            }
//            Extractor ext = new Extractor(bitsHash,iter+1);
//            JavaPairRDD<Integer,Instance> pairRDD= input.mapToPair(ext);
//            pairRDD.groupByKey();

            //pairRDD=pairRDD.partitionBy(new CustomPartitioner(iter+1));
        }

        protected void saveModel( String dir, String fname ) throws IOException {
            FileDeleter.delete(new File(dir + fname));
            IOLearner.saveLearner(learner, fname);

            // copy output to HDFS
            FileSystem fileSystem = FileSystem.get(new Configuration());
            fileSystem.moveFromLocalFile(new Path(fname), new Path(dir));

        }

        public double eval( JavaRDD<String> samples, VWParser vwparser ) throws  IOException {
            int numSamples = 0;
            double score;
            double cumLoss = 0.0;

            for (String strLine : samples.take(300000)) {

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

            return cumLoss / (double) numSamples;
        }


//        public double eval( ArrayList<Path> files, FileSystem hdfs, VWParser vwparser ) throws  IOException {
//            int numSamples = 0;
//            double score;
//            double cumLoss = 0.0;
//
//            for (Path featureFile : files) {
//                BufferedReader br = null;
//                if (featureFile.getName().contains(".gz"))
//                    br = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
//                else
//                    br = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));
//
//                for(;;) { // forever
//                    String strLine = br.readLine();
//
//                    Instance sample;
//
//                    if (strLine != null) {
//                        sample = vwparser.parse(strLine);
//                    } else
//                        break;
//
//
//                    score = learner.predict(sample);
//                    score = Math.min(Math.max(score, minPrediction), maxPrediction);
//
//                    cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//
//                    numSamples++;
//                }
//            }
//
//            return cumLoss / (double) numSamples;
//        }

//        BufferedReader trainbr = null;
//        public void open()throws IOException {
//            if (trainbr != null) trainbr.close();
//
//            Collections.shuffle(featureFilePaths);
//            Path featureFile = featureFilePaths.get(0);
//            System.out.println("----- Starting file " + featureFile.toString() + " -----");
//
//            if (featureFile.getName().contains(".gz"))
//                trainbr = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
//            else
//                trainbr = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));
//        }
//
//
//        public String getLine() throws IOException{
//            String strLine = trainbr.readLine();
//            if (strLine == null) open();
//            strLine = trainbr.readLine();
//            return strLine;
//        }

    }



    public static void main(String[] args) throws IOException {
        DistTrainRunner sl = new DistTrainRunner();
        sl.train();
    }

}
