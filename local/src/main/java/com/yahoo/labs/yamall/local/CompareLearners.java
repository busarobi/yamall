package com.yahoo.labs.yamall.local;


import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.*;
import com.yahoo.labs.yamall.parser.VWParser;

import java.io.*;
import java.util.LinkedList;
import java.util.Properties;
import java.util.Queue;
import java.util.concurrent.TimeUnit;

/**
 * Created by busafekete on 7/11/17.
 */
public class CompareLearners extends Thread {
    protected DataGenerator train = null;
    protected DataGenerator test = null;
    protected String outputFile = null;
    protected String postFix = null;

    protected String method = null;
    protected BufferedWriter results = null;
    protected Learner learner = null;
    protected Properties properties = null;
    public static int bitsHash = 22;
    public static int evalPeriod = 10000;

    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;


    public CompareLearners ( DataGenerator train, DataGenerator test, Properties properties, String postFix ) throws IOException {
        this.train = train;
        this.test = test;
        this.properties = properties;
        this.postFix = postFix;
        this.method = properties.getProperty("method", null);

        this.learner = getLearner();

        String fname = this.properties.getProperty("output");
        this.outputFile = fname + "_" + this.method + "_" + this.postFix + ".txt";

        System.out.println( "Result file: " + this.outputFile );


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

        System.out.println( "----> Method: " + this.method );

        if ( this.method.compareToIgnoreCase("SGD_VW") == 0) {

            double learningRate = Double.parseDouble(this.properties.getProperty("sgd_vw_lr", "1.0"));

            this.postFix = String.format("lr_%f_", learningRate) + this.postFix;

            System.out.println( "SGD_VW learning rate: " + learningRate);

            learner = new SGD_VW(bitsHash);
            learner.setLearningRate(learningRate);
        } else if ( this.method.compareToIgnoreCase("SVRG") == 0) {
            double learningRate = Double.parseDouble(this.properties.getProperty("svrg_lr", "0.05"));
            double regPar = Double.parseDouble(this.properties.getProperty("svrg_reg", "0.0"));
            int step = Integer.parseInt(this.properties.getProperty("svrg_step", "50"));

            this.postFix = String.format("lr_%f_reg_%f_step_%d_", learningRate, regPar, step) + this.postFix;

            System.out.println( "SVRG learning rate: " + learningRate);
            System.out.println( "SVRG regularization param: " + regPar);
            System.out.println( "SVRG step: " + step);

            SVRG svrg = new SVRG(bitsHash);
            svrg.setLearningRate(learningRate);
            svrg.setRegularizationParameter(regPar);
            svrg.setStep(step);
            //svrg.doAveraging();

            learner = svrg;
        } else if ( this.method.compareToIgnoreCase("SVRG_ADA") == 0) {
            double learningRate = Double.parseDouble(this.properties.getProperty("svrg_ada_lr", "0.05"));
            double regPar = Double.parseDouble(this.properties.getProperty("svrg_ada_reg", "0.0"));
            int step = Integer.parseInt(this.properties.getProperty("svrg_ada_step", "500"));

            this.postFix = String.format("lr_%f_reg_%f_step_%d_", learningRate, regPar, step) + this.postFix;

            System.out.println( "SVRG learning rate: " + learningRate);
            System.out.println( "SVRG regularization param: " + regPar);
            System.out.println( "SVRG step: " + step);

            SVRG_ADA svrg = new SVRG_ADA(bitsHash);
            svrg.setLearningRate(learningRate);
            svrg.setRegularizationParameter(regPar);
            svrg.setStep(step);
            //svrg.doAveraging();

            learner = svrg;
        } else if ( this.method.compareToIgnoreCase("SVRG_FR") == 0) {
            double learningRate = Double.parseDouble(this.properties.getProperty("svrg_fr_lr", "0.05"));
            double regPar = Double.parseDouble(this.properties.getProperty("svrg_fr_reg", "0.0"));
            int step = Integer.parseInt(this.properties.getProperty("svrg_fr_step", "500"));

            this.postFix = String.format("lr_%f_reg_%f_step_%d_", learningRate, regPar, step) + this.postFix;

            System.out.println( "SVRG_FR learning rate: " + learningRate);
            System.out.println( "SVRG_FR regularization param: " + regPar);
            System.out.println( "SVRG_FR step: " + step);

            SVRG_FR svrg = new SVRG_FR(bitsHash);
            svrg.setLearningRate(learningRate);
            svrg.setRegularizationParameter(regPar);
            svrg.setStep(step);
            //svrg.doAveraging();

            learner = svrg;
        } else if ( this.method .compareToIgnoreCase("SGD") == 0) {
            double learningRate = Double.parseDouble(this.properties.getProperty("sgd_lr", "1.0"));

            this.postFix = String.format("lr_%f_", learningRate) + this.postFix;

            System.out.println( "SGD learning rate: " + learningRate);

            learner = new SGD(bitsHash);
            learner.setLearningRate(learningRate);
        } else if ( this.method .compareToIgnoreCase("FREE_REX") == 0) {
            double learningRate = Double.parseDouble(this.properties.getProperty("free_rex_lr", "0.01"));

            this.postFix = String.format("lr_%f_", learningRate) + this.postFix;

            System.out.println( "FREE REX learning rate: " + learningRate);

            learner = new PerCoordinateFreeRex(bitsHash);
            learner.setLearningRate(learningRate);
        } else if ( this.method .compareToIgnoreCase("SOLO") == 0) {
            double learningRate = Double.parseDouble(this.properties.getProperty("solo_lr", "0.1"));

            this.postFix = String.format("lr_%f_", learningRate) + this.postFix;

            System.out.println( "SOLO learning rate: " + learningRate);

            learner = new PerCoordinateSOLO(bitsHash);
            learner.setLearningRate(learningRate);
        } else if ( this.method .compareToIgnoreCase("MB_SGD") == 0) {
            MiniBatchSGD mbsgd = new MiniBatchSGD(bitsHash);

            double learningRate = Double.parseDouble(this.properties.getProperty("mb_sgd_lr", "0.05"));
            double regPar = Double.parseDouble(this.properties.getProperty("mb_sgd_reg", "0.0"));
            int step = Integer.parseInt(this.properties.getProperty("mb_sgd_step", "50"));

            this.postFix = String.format("lr_%f_reg_%f_step_%d_", learningRate, regPar, step) + this.postFix;

            System.out.println( "MB SGD learning rate: " + learningRate);
            System.out.println( "MB SGD regularization param: " + regPar);
            System.out.println( "MB SGD step: " + step);

            mbsgd.setLearningRate(learningRate);
            mbsgd.setRegularizationParameter(regPar);
            mbsgd.setStep(step);
            learner = mbsgd;
        }

        Loss lossFnc = new LogisticLoss();
        learner.setLoss(lossFnc);

        return learner;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println(
                    "Usage: java -classpath yamall-examples-jar-with-dependencies.jar com.yahoo.labs.yamall.examples.StatisticsVWFile output");
            System.exit(0);
        }
        System.out.println("Wed Jul 19 15:13:33 UTC 2017");

        Properties properties = ReadProperty.readProperty(args[0]);

        int trainSize = Integer.parseInt(properties.getProperty("tain_size", "5000000"));
        int testSize = Integer.parseInt(properties.getProperty("test_size", "50000"));
        int dim = 1000;
        int sparsity = 100;

        String dataType = properties.getProperty("data_type", "default");
        DataGenerator train = null;
        DataGenerator test = null;
        if (dataType.compareToIgnoreCase("default") == 0) {
            train = new DataGeneratorNormal(trainSize, dim, sparsity);
            test = train.copy();
            ((DataGeneratorNormal) test).setNum(testSize);
        } else if (dataType.compareToIgnoreCase("file") == 0) {
            String trainFile = properties.getProperty("train_file");
            train = new DataGeneratorFromFile(trainFile);

            String testFile = properties.getProperty("test_file");
            test = new DataGeneratorFromFile(testFile);
        } else if (dataType.compareToIgnoreCase("gzfile") == 0) {
            String trainFile = properties.getProperty("train_file");
            train = new DataGeneratorFromZippedFile(trainFile);

            String testFile = properties.getProperty("test_file");
            test = new DataGeneratorFromZippedFile(testFile);
        }


        System.out.println( "Train size: " + trainSize );
        System.out.println( "Test file: " + testSize );

        System.out.println( "Dim: " + dim );
        System.out.println( "Sparsity: " + sparsity );


        int numOfThreads = Integer.parseInt(properties.getProperty("threads", "10"));
        //String[] methodNames = {"SVRG", "MB_SGD"};
        Queue<CompareLearners> queue = new LinkedList<CompareLearners>();

        for( int i=0; i < numOfThreads; i++ ){
            String postFix = String.format("%04d_%s", i, dataType);
            CompareLearners cp = new CompareLearners(train.copy(), test.copy(), properties, postFix);
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
