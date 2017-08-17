package com.yahoo.labs.yamall.synthetic.helper;

import com.yahoo.labs.yamall.ml.*;

import java.util.Properties;

/**
 * Created by busafekete on 8/15/17.
 */
public class LearnerFactory {
    public static Learner getLearner(Properties properties ){
        Learner learner = null;
        String method = properties.getProperty("method", null);
        int bitsHash = Integer.parseInt(properties.getProperty("b", "22" ));
        double learningRate = Double.parseDouble(properties.getProperty("lr", "1.0"));
        System.out.println( "----> Method: " + method );

        if ( method.compareToIgnoreCase("SGD_VW") == 0) {
            System.out.println( "SGD_VW learning rate: " + learningRate);

            learner = new SGD_VW(bitsHash);
            learner.setLearningRate(learningRate);
        } else if ( method .compareToIgnoreCase("Pistol") == 0) {
            System.out.println( "SGD learning rate: " + learningRate);

            learner = new PerCoordinatePiSTOL(bitsHash);
            learner.setLearningRate(learningRate);
        } else if ( method .compareToIgnoreCase("FREE_REX") == 0) {
            boolean scaling = Boolean.parseBoolean(properties.getProperty("scaling", "false"));
            boolean wscaling = Boolean.parseBoolean(properties.getProperty("wscaling", "true"));

            System.out.println( "FREE REX learning rate: " + learningRate);

            PerCoordinateFreeRex l = new PerCoordinateFreeRex(bitsHash);
            l.useScaling(scaling);
            l.useWeightScaling(wscaling);
            l.setLearningRate(learningRate);

            learner = l;
        } else if ( method .compareToIgnoreCase("DENSE_FREE_REX") == 0) {
            boolean scaling = Boolean.parseBoolean(properties.getProperty("scaling", "false"));
            boolean wscaling = Boolean.parseBoolean(properties.getProperty("wscaling", "true"));

            System.out.println( "DENSE FREE REX learning rate: " + learningRate);

            FreeRex l = new FreeRex(bitsHash);
            l.useScaling(scaling);
            l.useWeightScaling(wscaling);
            l.setLearningRate(learningRate);

            learner = l;
        } else if ( method .compareToIgnoreCase("SVRG") == 0) {
            int batchSize = Integer.parseInt(properties.getProperty("batchsize", "1000"));

            System.out.println( "SVRG learning rate: " + learningRate);

            SVRG l = new SVRG(bitsHash);
            l.setBatchSize(batchSize);
            l.setLearningRate(learningRate);

            learner = l;
        } else if ( method .compareToIgnoreCase("SVRG_FR") == 0) {
            int batchSize = Integer.parseInt(properties.getProperty("batchsize", "1000"));

            System.out.println( "SVRG_FR learning rate: " + learningRate);

            PerCoordinateSVRG l = new PerCoordinateSVRG(bitsHash);
            l.setBatchSize(batchSize);
            l.setLearningRate(learningRate);

            learner = l;
        } else if ( method .compareToIgnoreCase("DENSE_SVRG") == 0) {
            int batchSize = Integer.parseInt(properties.getProperty("batchsize", "1000"));

            System.out.println( "DENSE_SVRG learning rate: " + learningRate);

            DenseSVRG l = new DenseSVRG(bitsHash);
            l.setBatchSize(batchSize);
            l.setLearningRate(learningRate);

            learner = l;
        } else if ( method .compareToIgnoreCase("SOLO") == 0) {
            System.out.println( "SOLO learning rate: " + learningRate);

            learner = new PerCoordinateSOLO(bitsHash);
            learner.setLearningRate(learningRate);
        }
        Loss lossFnc = new LogisticLoss();
        learner.setLoss(lossFnc);

        return learner;
    }

}
