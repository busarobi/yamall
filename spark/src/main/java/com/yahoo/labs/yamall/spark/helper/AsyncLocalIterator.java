package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import org.apache.spark.Partition;
import org.apache.spark.SimpleFutureAction;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import scala.collection.mutable.ArraySeq;
import scala.runtime.AbstractFunction0;
import scala.runtime.AbstractFunction1;
import scala.runtime.AbstractFunction2;
import scala.runtime.BoxedUnit;

import java.io.Serializable;
import java.util.*;

public class AsyncLocalIterator implements Iterator<Instance>, Serializable {
    public static abstract class SerializableFunction2<T1, T2, R> extends AbstractFunction2<T1, T2, R> implements Serializable {
    }
    public static abstract class SerializableFunction1<T1, R> extends AbstractFunction1<T1, R> implements Serializable {
    }
    public static abstract class SerializableFunction0<R> extends AbstractFunction0<R> implements Serializable {
    }



    public static SimpleFutureAction<ArrayList<Instance>> collectPartition(JavaRDD<Instance> javaRDD, int p) {

        ArrayList<ArrayList<Instance>> results = new ArrayList<ArrayList<Instance>>(1);


        SerializableFunction2<Object, ArrayList<Instance>, BoxedUnit> handler = new SerializableFunction2<Object, ArrayList<Instance>, BoxedUnit>() {
            public BoxedUnit apply(Object index, ArrayList<Instance> result) {
                results.add(0, result);
                return BoxedUnit.UNIT;
            }
        };


        SerializableFunction1<scala.collection.Iterator<Instance>, ArrayList<Instance>> collector = new SerializableFunction1<scala.collection.Iterator<Instance>, ArrayList<Instance>>() {
            public ArrayList<Instance> apply(scala.collection.Iterator<Instance> iter) {
                ArrayList<Instance> instances = new ArrayList<Instance>();
                while (iter.hasNext()) {
                    instances.add(iter.next());
                }
                return instances;
            }
        };


        SerializableFunction0<ArrayList<Instance>> getresults = new SerializableFunction0<ArrayList<Instance>>() {
            public ArrayList<Instance> apply() {
                return results.get(0);
            }
        };

        RDD<Instance> scalaRDD = (RDD<Instance>) javaRDD.rdd();
        ArraySeq<Object> seq = new ArraySeq<Object>(1);
        seq.update(0, p);
        return scalaRDD.context().submitJob(scalaRDD, collector, seq, handler, getresults);
    }


    JavaRDD<Instance> rdd;
    int nextPartition = 0;
    int numPartitions;
    int numBufferedPartitions = 1;
    Queue<Integer> partitionIndices;
    Queue<SimpleFutureAction<ArrayList<Instance>>> futures;
    Queue<ArrayList<Instance>> arrays;
    Iterator<Instance> currentIterator;

    @Override
    public boolean hasNext() {

        boolean currentHasNext = false;
        if (currentIterator != null)
            currentHasNext = currentIterator.hasNext();

        //if the array currently in memory still has elements, we definitely can return true
        if(currentHasNext)
            return true;

        //else, load the next iterator, then call hasNext again.
        try {
            this.loadIterator();
        } catch ( org.apache.spark.SparkException e ) {
            throw new RuntimeException(e);
        }

        if(currentIterator == null) {
            return false;
        }
        return this.hasNext();
    }

    private void loadPartition() throws org.apache.spark.SparkException {
        SimpleFutureAction<ArrayList<Instance>> nextFuture = this.futures.poll();
        if(nextFuture != null) {
            this.arrays.offer(nextFuture.get());
            this.enqueuePartition();
        }
    }

    private void loadIterator() throws org.apache.spark.SparkException {
        ArrayList<Instance> nextArray = this.arrays.poll();
        if(nextArray == null) {
            this.loadPartition();
            nextArray = this.arrays.poll();
        }

        if(nextArray != null) {
            this.currentIterator = nextArray.iterator();
        } else {
            this.currentIterator = null;
        }

    }

    @Override
    public Instance next() {
        try {
            if (this.currentIterator == null)
                this.loadIterator();

            if (this.currentIterator.hasNext() == false)
                this.loadIterator();
        } catch ( org.apache.spark.SparkException e ) {
            throw new RuntimeException(e);
        }

        return this.currentIterator.next();
    }

    private void enqueuePartition() {
        if(this.nextPartition < this.numPartitions) {
            int index = this.partitionIndices.poll();
            this.futures.offer(AsyncLocalIterator.collectPartition(rdd, index));
        }
        this.nextPartition++;

    }

    public AsyncLocalIterator(JavaRDD<Instance> rdd, int numBufferedPartitions) {
        this.numBufferedPartitions = numBufferedPartitions;
        this.initialize(rdd);
    }

    public AsyncLocalIterator(JavaRDD<Instance> rdd) {
        this.initialize(rdd);
    }

    private void initialize(JavaRDD<Instance> rdd) {
        this.rdd = rdd;
        this.numPartitions = rdd.getNumPartitions();
        this.partitionIndices = new LinkedList<Integer>();
        for(Partition p: this.rdd.partitions()) {
            this.partitionIndices.offer(p.index());
        }
        this.currentIterator = null;

        this.futures = new LinkedList<SimpleFutureAction<ArrayList<Instance>>>();
        this.arrays = new LinkedList<ArrayList<Instance>>();

        for(int i=0; i < this.numBufferedPartitions; i++) {
            this.enqueuePartition();
        }

    }


}
