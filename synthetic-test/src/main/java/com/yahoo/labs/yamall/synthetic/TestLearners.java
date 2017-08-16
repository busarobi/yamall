package com.yahoo.labs.yamall.synthetic;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.Learner;
import com.yahoo.labs.yamall.parser.VWParser;
import com.yahoo.labs.yamall.synthetic.data.DataGenerator;
import com.yahoo.labs.yamall.synthetic.data.DataGeneratorFactory;
import com.yahoo.labs.yamall.synthetic.helper.JobParallelLauncher;
import com.yahoo.labs.yamall.synthetic.helper.LearnerFactory;
import com.yahoo.labs.yamall.synthetic.helper.ReadProperty;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

/**
 * Created by busafekete on 8/15/17.
 */
public class TestLearners implements JobParallelLauncher.JobRunner {
    //static protected Logger logger = Logger.getLogger(TestLearners.class);

    protected DataGenerator train = null;
    protected DataGenerator test = null;
    protected String outputFile = null;
    protected String postFix = null;

    protected String method = null;
    protected BufferedWriter results = null;
    protected Learner learner = null;
    protected Properties properties = null;
    public static int bitsHash = 22;
    public static int evalPeriod = 1000;

    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;


    public TestLearners ( DataGenerator train, DataGenerator test, Properties properties, String outputFile ) throws IOException {
        this.train = train;
        this.test = test;
        this.properties = properties;
        this.postFix = postFix;

        this.bitsHash = Integer.parseInt(properties.getProperty("b", "22" ));
        this.method = properties.getProperty("method", null);

        this.evalPeriod = Integer.parseInt(properties.getProperty("evalPeriod", "1000"));
        this.learner = LearnerFactory.getLearner(properties);
        this.outputFile = outputFile;


        System.out.println( "Result file: " + this.outputFile );
        System.out.println( "Eval period: " + this.evalPeriod );
  }


    public void run() {
        try {
            train();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void train() throws IOException {
        long start = System.nanoTime();

        results = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile )));

        VWParser vwparser = new VWParser(bitsHash, null, false);

        int numSamples = 0;
        double score;
        double cumLoss = 0.0;

        do {
            Instance sample;
            String strLine = train.getNextInstance();

            if (strLine != null)
                sample = vwparser.parse(strLine);
            else
                break;

            score = learner.update(sample);
            score = Math.min(Math.max(score, minPrediction), maxPrediction);

            cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();

            numSamples++;

            if (numSamples % evalPeriod == 0 ){
                double trainLoss = cumLoss / (double) numSamples;
                double testLoss = eval();

                double l2norm = learner.getWeights().squaredL2Norm();
                String line = String.format("%d %f %f %f\n", numSamples, trainLoss, testLoss, l2norm );
                results.write(line);
                results.flush();

                System.out.print(this.method + " " + line);

            }
        }
        while (true);

        train.close();
        results.close();

        long millis = System.nanoTime() - start;
        System.out.println(String.format("Elapsed time: %d min, %d sec\n", TimeUnit.NANOSECONDS.toMinutes(millis),
                TimeUnit.NANOSECONDS.toSeconds(millis) - 60 * TimeUnit.NANOSECONDS.toMinutes(millis)));
    }

    public double eval() throws  IOException {

        VWParser vwparser = new VWParser(bitsHash, null, false);

        int numSamples = 0;
        double score;
        double cumLoss = 0.0;

        do {
            Instance sample;
            String strLine = test.getNextInstance();
            if (strLine != null)
                sample = vwparser.parse(strLine);
            else
                break;

            score = learner.predict(sample);
            score = Math.min(Math.max(score, minPrediction), maxPrediction);

            cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();

            numSamples++;
        }
        while (true);

        test.close();

        return cumLoss / (double) numSamples;
    }


    public static void process(Properties properties) throws IOException {
        int numOfThreads = Integer.parseInt(properties.getProperty("threads", "4"));
        int numOfRepetitions = Integer.parseInt(properties.getProperty("repetitions", "10"));
        String method = properties.getProperty("method", null);
        String fname = properties.getProperty("output", "./");
        String learningRate = properties.getProperty("lr", "1.0");


        JobParallelLauncher launcher = new JobParallelLauncher(numOfThreads);


        for( int i=0; i < numOfRepetitions; i++ ){
            String dataType = properties.getProperty("data_type", "default");

            String postFix = String.format("r_%04d_%s_lr_%s", i, dataType, learningRate);
            String outputFile = fname + "/" + method + "_" + postFix + ".txt";
            System.out.println("Output: " + outputFile);

            properties.setProperty("seed", Integer.toString(i));
            DataGeneratorFactory.DataPair pair = DataGeneratorFactory.getDataGenerator(properties);

            TestLearners cp = new TestLearners(pair.train.copy(), pair.test.copy(), properties, outputFile);
            launcher.addJob(cp);
        }

        launcher.run();
    }


    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: java -classpath yamall-examples-jar-with-dependencies.jar com.yahoo.labs.yamall.synthetic.TestLearners properyfile");
            System.exit(0);
        }

        Properties properties = ReadProperty.readProperty(args[0]);
        process(properties);
    }

}