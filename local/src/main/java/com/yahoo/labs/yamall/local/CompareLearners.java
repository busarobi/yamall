package com.yahoo.labs.yamall.local;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.*;
import com.yahoo.labs.yamall.parser.VWParser;

import java.io.*;
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.TimeUnit;

/**
 * Created by busafekete on 7/11/17.
 */
public class CompareLearners extends Thread {

    protected String trainFile = null;
    protected String testFile = null;
    protected String outputFile = null;

    protected String method = null;
    protected BufferedWriter results = null;
    protected Learner learner = null;

    public static int bitsHash = 22;
    public static int evalPeriod = 1000;

    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;


    public CompareLearners ( String trainFile, String testFile, String outputFile, String method ) throws IOException {
        this.trainFile = trainFile;
        this.testFile = testFile;
        this.method = method;
        this.outputFile = outputFile + "_" + this.method + ".txt";

        System.out.println( "Result file: " + this.outputFile );
        this.learner = getLearner();

    }

    @Override
    public void run() {
        try {
            train();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void train() throws IOException {
        long start = System.nanoTime();

        FileInputStream fstream = new FileInputStream(this.trainFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

        results = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile )));

        VWParser vwparser = new VWParser(bitsHash, null, false);

        int numSamples = 0;
        double score;
        double cumLoss = 0.0;

        do {
            Instance sample;
            String strLine = br.readLine();
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


                String line = String.format("%d %f %f\n", numSamples, trainLoss, testLoss );
                results.write(line );
                results.flush();

                System.out.print(this.method + " " + line);

            }
        }
        while (true);

        br.close();
        results.close();

        long millis = System.nanoTime() - start;
        System.out.printf("Elapsed time: %d min, %d sec\n", TimeUnit.NANOSECONDS.toMinutes(millis),
                TimeUnit.NANOSECONDS.toSeconds(millis) - 60 * TimeUnit.NANOSECONDS.toMinutes(millis));
    }

    public double eval() throws  IOException {
        FileInputStream fstream = new FileInputStream(testFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

        VWParser vwparser = new VWParser(bitsHash, null, false);

        int numSamples = 0;
        double score;
        double cumLoss = 0.0;

        do {
            Instance sample;
            String strLine = br.readLine();
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

        br.close();

        return cumLoss / (double) numSamples;
    }


    public Learner getLearner(){
        Learner learner = null;
        Loss lossFnc = new LogisticLoss();

        System.out.println( "Method: " + this.method );

        if ( this.method.compareToIgnoreCase("SGD_VW") == 0) {
            double learningRate = 1.0;
            learner = new SGD_VW(bitsHash);
            learner.setLoss(lossFnc);
            learner.setLearningRate(learningRate);
        } else if ( this.method.compareToIgnoreCase("SVRG") == 0) {
            SVRG svrg = new SVRG(bitsHash);
            svrg.setLoss(lossFnc);

            double learningRate = 0.05;
            svrg.setLearningRate(learningRate);
            svrg.setRegularizationParameter(0.0);
            svrg.setStep(50);
            //svrg.doAveraging();

            learner = svrg;
        } else if ( this.method .compareToIgnoreCase("SGD") == 0) {
            double learningRate = 1.0;
            learner = new SGD(bitsHash);
            learner.setLoss(lossFnc);
            learner.setLearningRate(learningRate);
        } else if ( this.method .compareToIgnoreCase("MB_SGD") == 0) {
            MiniBatchSGD mbsgd = new MiniBatchSGD(bitsHash);
            mbsgd.setLoss(lossFnc);

            double learningRate = 0.5;
            mbsgd.setLearningRate(learningRate);
            mbsgd.setRegularizationParameter(0.001);
            mbsgd.setStep(50);

            learner = mbsgd;
        }

        return learner;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println(
                    "Usage: java -classpath yamall-examples-jar-with-dependencies.jar com.yahoo.labs.yamall.examples.StatisticsVWFile vw_filename_train vw_filename_test output");
            System.exit(0);
        }

        String trainFile = args[0];
        String testFile = args[1];
        String resultFile = args[2];

        System.out.println( "Result file: " + trainFile );
        System.out.println( "Result file: " + testFile );

        //String[] methodNames = {"SGD_VW", "SVRG", "SGD", "MB_SGD"};
        String[] methodNames = {"SVRG", "MB_SGD"};
        Queue<CompareLearners> queue = new LinkedList<CompareLearners>();

        for( String m : methodNames ){
            CompareLearners cp = new CompareLearners(trainFile,testFile,resultFile,m);
            cp.start();
            queue.add(cp);
        }

        try {
            for( CompareLearners cp : queue ){
                cp.join();
            }
        } catch (InterruptedException e) {
            System.out.println("Main thread Interrupted");
        }

    }

}
