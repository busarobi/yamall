package com.yahoo.labs.yamall.local;

/**
 * Created by busafekete on 7/17/17.
 */
public interface DataGenerator {
    public String getNextInstance();
    public void close();
    public DataGenerator copy();
}
