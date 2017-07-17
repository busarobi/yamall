package com.yahoo.labs.yamall.local;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.*;
import com.yahoo.labs.yamall.parser.VWParser;
import javafx.scene.chart.PieChart;

import java.io.*;
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.TimeUnit;

/**
 * Created by busafekete on 7/11/17.
 */
public class CompareLearners extends Thread {
    protected DataGenerator train = null;
    protected DataGenerator test = null;
    protected String outputFile = null;

    protected String method = null;
    protected BufferedWriter results = null;
    protected Learner learner = null;

    public static int bitsHash = 22;
    public static int evalPeriod = 1000;

    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;


    public CompareLearners ( DataGenerator train, DataGenerator test, String outputFile, String method ) throws IOException {
        this.train = train;
        this.test = test;
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


                String line = String.format("%d %f %f\n", numSamples, trainLoss, testLoss );
                results.write(line );
                results.flush();

                System.out.print(this.method + " " + line);

            }
        }
        while (true);

        train.close();
        results.close();

        long millis = System.nanoTime() - start;
        System.out.printf("Elapsed time: %d min, %d sec\n", TimeUnit.NANOSECONDS.toMinutes(millis),
                TimeUnit.NANOSECONDS.toSeconds(millis) - 60 * TimeUnit.NANOSECONDS.toMinutes(millis));
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
            svrg.setStep(500);
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

            double learningRate = 0.05;
            mbsgd.setLearningRate(learningRate);
            mbsgd.setRegularizationParameter(0.0);
            mbsgd.setStep(500);

            learner = mbsgd;
        }

        return learner;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println(
                    "Usage: java -classpath yamall-examples-jar-with-dependencies.jar com.yahoo.labs.yamall.examples.StatisticsVWFile output");
            System.exit(0);
        }

        int trainSize = 5000000;
        int testSize = 50000;
        String resultFile = args[0];

        DataGenerator train = new DataGeneratorNormal(trainSize, 1000, 100);
        DataGenerator test = train.copy();
        ((DataGeneratorNormal)test).setNum(testSize);

        System.out.println( "Train size: " + trainSize );
        System.out.println( "Test file: " + testSize );

        String[] methodNames = {"SGD_VW", "SVRG", "SGD", "MB_SGD"};
        //String[] methodNames = {"SVRG", "MB_SGD"};
        Queue<CompareLearners> queue = new LinkedList<CompareLearners>();

        for( String m : methodNames ){
            CompareLearners cp = new CompareLearners(train.copy(), test.copy(), resultFile,m);
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
