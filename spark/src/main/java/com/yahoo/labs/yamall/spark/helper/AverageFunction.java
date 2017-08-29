package com.yahoo.labs.yamall.spark.helper;

import org.apache.spark.api.java.function.Function2;

import java.io.Serializable;

/**
 * Created by busafekete on 8/29/17.
 */
public class AverageFunction {
    public static class AvgCount implements Serializable {
        public AvgCount(double val, int num) {
            val_ = val;
            num_ = num;
        }
        public double val_;
        public int num_;
        public double avg() {
            return val_ / (double) num_;
        }
    }

    public static class AddandCount implements Function2<AvgCount, Double, AvgCount> {
        @Override
        public AvgCount call(AvgCount a, Double x) {
            a.val_ += x;
            a.num_ += 1;
            return a;
        }
    }

    public static class Combine implements Function2<AvgCount, AvgCount, AvgCount> {
        @Override
        public AvgCount call(AvgCount a, AvgCount b) {
            a.val_ += b.val_;
            a.num_ += b.num_;
            return a;
        }
    }

}
