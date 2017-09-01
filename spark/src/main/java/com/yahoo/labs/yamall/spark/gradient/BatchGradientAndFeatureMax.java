//package com.yahoo.labs.yamall.spark.gradient;
//
//import com.yahoo.labs.yamall.core.Instance;
//import org.apache.spark.api.java.JavaRDD;
//
///**
// * Created by busafekete on 8/30/17.
// */
//public class BatchGradientAndFeatureMax {
//    public static class BatchGradientDataAndFeatureMax extends BatchGradient.BatchGradientData {
//        BatchGradientDataAndFeatureMax(int b, double[] weights) {
//            super(b,weights);
//        }
//        //TODO:
//        //TODO:
//    }
//
//    public static class CombOpAndFeatureMax extends BatchGradient.CombOp {
//        //TODO:
//    }
//
//    public static class SeqOpAndFeatureMax extends BatchGradient.SeqOp {
//        //TODO:
//    }
//
//    public static BatchGradientAndFeatureMax computeGradient(JavaRDD<Instance> data, int bit, double[] w ){
//        BatchGradientAndFeatureMax batchgradient = data.treeAggregate(new BatchGradientAndFeatureMax(bit, w), new SeqOp(), new CombOp(), 11);
//        batchgradient.normalizeBatchGradient();
//        return batchgradient;
//    }
//
//
//}
