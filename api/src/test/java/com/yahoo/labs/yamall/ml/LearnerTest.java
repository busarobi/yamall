// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.unitils.reflectionassert.ReflectionAssert;

import java.io.*;

public class LearnerTest {

    public double learnerLogisticLossTest(Learner l, int iter) throws IOException, ClassNotFoundException {
        Instance pos = new Instance(1.0);
        Instance neg = new Instance(-1.0);
        pos.getVector().put(0, 1);
        neg.getVector().put(0, 1);

        Loss loss = new LogisticLoss();
        l.setLoss(loss);

        double sPlus = 0.0;
        double sMinus = 0.0;
        for (int i = 0; i < iter; i++) {
            if (i % 5 == 0) {
                l.update(pos);
                sPlus = sPlus + pos.getWeight();
            }
            else {
                l.update(neg);
                sMinus = sMinus + neg.getWeight();
            }
        }
        System.out.println(l);
        System.out.println("Weights: " + l.getWeights());
        System.out.println("Constant:" + Math.log(sPlus / sMinus));
        double lossPos = loss.lossValue(l.predict(pos), 1.0);
        double lossNeg = loss.lossValue(l.predict(neg), -1.0);
        double avLoss = (lossPos * sPlus + lossNeg * sMinus) / (sPlus + sMinus);
        double exactLoss = loss.lossConstantBinaryLabels(sPlus, sMinus);
        System.out.println("Average loss algo: " + avLoss);
        System.out.println("Average loss exact: " + exactLoss);
        System.out.println("Error: " + (avLoss - exactLoss));
        System.out.println();

        byte[] tmp = pickle(l);
        Learner l2 = unpickle(tmp, Learner.class);
        ReflectionAssert.assertReflectionEquals(l, l2);

        return avLoss - exactLoss - 1 / Math.sqrt(iter);
    }

    private static <T extends Serializable> byte[] pickle(T obj) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(obj);
        oos.close();
        return baos.toByteArray();
    }

    private static <T extends Serializable> T unpickle(byte[] b, Class<T> cl)
            throws IOException, ClassNotFoundException {
        ByteArrayInputStream bais = new ByteArrayInputStream(b);
        ObjectInputStream ois = new ObjectInputStream(bais);
        Object o = ois.readObject();
        return cl.cast(o);
    }

    @Test
    public void testLearners() throws ClassNotFoundException, IOException {
        int iter = 1000000;

        Assert.assertTrue(learnerLogisticLossTest(new SOLO(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new SGD_VW(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new KT(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new PerCoordinatePiSTOL(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new PerCoordinateSOLO(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new COCOB(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new PerCoordinateCOCOB(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new PerCoordinateKT(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new SGD_FM(1, 8), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new PerCoordinateFreeRex(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new SVRG(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new PerCoordinateSVRG(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new FreeRex(1), iter) < 0);
        Assert.assertTrue(learnerLogisticLossTest(new DenseSVRG(1), iter) < 0);
    }

}
