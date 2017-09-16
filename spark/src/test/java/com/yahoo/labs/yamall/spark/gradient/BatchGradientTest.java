package com.yahoo.labs.yamall.spark.gradient;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.PerCoordinateSVRG;
import com.yahoo.labs.yamall.spark.helper.HDFSDirectoryReader;
import com.yahoo.labs.yamall.spark.helper.StringToYamallInstance;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Random;

/**
 * Created by busafekete on 9/15/17.
 */
public class BatchGradientTest {
    protected Random random = new Random(0);
    protected int bits = 5;
    protected int size_hash = 1 << bits;
    protected String inputDir = "/Users/busafekete/work/DistOpt/spark/datagz/";

    @BeforeMethod
    public void setUp() throws Exception {
        System.out.println( "--> Input dit: " + inputDir );

    }

    @AfterMethod
    public void tearDown() throws Exception {

    }

    @Test
    public void testComputeGradient() throws Exception {
        // compute the batch gradient by using the sequential algorithm
        PerCoordinateSVRG learner = new PerCoordinateSVRG(bits);
        learner.setLoss(new LogisticLoss());

        // we don't want to change the visibility of updateBatchGradient() therefore we change its visibility temporarily
        Method updateBatchGradient = PerCoordinateSVRG.class.getDeclaredMethod("updateBatchGradient", Instance.class);
        updateBatchGradient.setAccessible(true);
        Method endBatchPhase = PerCoordinateSVRG.class.getDeclaredMethod("endBatchPhase");
        endBatchPhase.setAccessible(true);

        HDFSDirectoryReader hdfsreader = new HDFSDirectoryReader(inputDir, bits);

        while (hdfsreader.hasNext()) {
            Instance instance = hdfsreader.getNextInstance();
            updateBatchGradient.invoke(learner, instance);  // equivalent learner.updateBatchGradient(instance);
        }
        // end batch phase because of lazy update
        endBatchPhase.invoke(learner);

        //compare them
        Field field = PerCoordinateSVRG.class.getDeclaredField("negativeBatchGradient");
        field.setAccessible(true);
        double[] batchGradient_seq = (double[]) field.get(learner);

        // spark version
        SparkConf sparkConf = new SparkConf().setAppName("batch gradient test unit").setMaster("local");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);


        JavaRDD<String> input = sparkContext.textFile(inputDir);
        JavaRDD<Instance> data = input.map(new StringToYamallInstance(bits));

        double[] w = new double[size_hash];
        for( int i = 0; i < size_hash; i++ ) w[i] = (2.0 * random.nextDouble()) - 1.0;

        BatchGradient.BatchGradientData result = BatchGradient.computeGradient(data, bits, w);
        double[] batchGradient = result.getGbatch();


        for( int i = 0; i < result.size_hash; i++) {
            System.out.println( i + " " + batchGradient[i] + " " + batchGradient_seq[i]);
        }

    }

}