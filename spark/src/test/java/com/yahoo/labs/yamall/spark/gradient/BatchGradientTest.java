package com.yahoo.labs.yamall.spark.gradient;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.PerCoordinateSVRG;
import com.yahoo.labs.yamall.spark.helper.HDFSDirectoryReader;
import com.yahoo.labs.yamall.spark.helper.StringToYamallInstance;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Random;

/**
 * Created by busafekete on 9/15/17.
 */
public class BatchGradientTest {
    protected static Random random = new Random(0);
    protected static int bits = 5;
    protected static int size_hash = 1 << bits;
    protected static String inputDir = "/Users/busafekete/work/DistOpt/Clkb_small_data/";

    @BeforeMethod
    public void setUp() throws Exception {
        System.out.println( "--> Input dit: " + inputDir );

    }

    @AfterMethod
    public void tearDown() throws Exception {

    }

    public static double[] computeSequential( double[] w ) throws NoSuchMethodException, IOException, NoSuchFieldException, InvocationTargetException, IllegalAccessException {
        // compute the batch gradient by using the sequential algorithm
        PerCoordinateSVRG learner = new PerCoordinateSVRG(bits);
        learner.setLoss(new LogisticLoss());

        // we don't want to change the visibility of updateBatchGradient() therefore we change its visibility temporarily
        Method updateBatchGradient = PerCoordinateSVRG.class.getDeclaredMethod("updateBatchGradient", Instance.class);
        updateBatchGradient.setAccessible(true);
        Method endBatchPhase = PerCoordinateSVRG.class.getDeclaredMethod("endBatchPhase");
        endBatchPhase.setAccessible(true);

        Field field = PerCoordinateSVRG.class.getDeclaredField("w_previous");
        field.setAccessible(true);
        field.set(learner,w);


        HDFSDirectoryReader hdfsreader = new HDFSDirectoryReader(inputDir, bits);

        while (hdfsreader.hasNext()) {
            Instance instance = hdfsreader.getNextInstance();
            updateBatchGradient.invoke(learner, instance);  // equivalent learner.updateBatchGradient(instance);
        }
        // end batch phase because of lazy update
        endBatchPhase.invoke(learner);

        //compare them
        field = PerCoordinateSVRG.class.getDeclaredField("negativeBatchGradient");
        field.setAccessible(true);
        double[] batchGradient_seq = (double[]) field.get(learner);

        return batchGradient_seq;
    }


    @Test
    public void testComputeGradient() throws Exception {
        // weight
        double[] w = new double[size_hash];
        for( int i = 0; i < size_hash; i++ ) w[i] = (2.0 * random.nextDouble()) - 1.0;

        // spark version
        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");


        JavaRDD<String> input = sparkContext.textFile(inputDir);
        JavaRDD<Instance> data = input.map(new StringToYamallInstance(bits));


        BatchGradient.BatchGradientData result = BatchGradient.computeGradient(data, bits, w);
        double[] batchGradient = result.getGbatch();

        double[] batchGradient_seq = computeSequential(w);

        double avgDifference = 0.0;
        for( int i = 0; i < result.size_hash; i++) {
            avgDifference = batchGradient[i] - batchGradient_seq[i];
            System.out.println( i + " " + String.format("%.15f", batchGradient[i] - batchGradient_seq[i]));

        }

        avgDifference /= (double) result.size_hash;
        System.out.println( String.format("Average difference: %.15f", avgDifference ) );

        Assert.assertEquals(avgDifference, 0.0, 0.0000001);
    }

}