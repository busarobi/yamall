package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.ml.IOLearner;
import com.yahoo.labs.yamall.ml.Learner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.File;
import java.io.IOException;

/**
 * Created by busafekete on 8/29/17.
 */
public class ModelSerializationToHDFS {
    private static final String MODEL_BIN = "model.bin";

    public static void saveModel(String dir, Learner learner) throws IOException {
        FileDeleter.delete(new File(MODEL_BIN));
        IOLearner.saveLearner(learner, MODEL_BIN);

        // copy output to HDFS
        FileSystem fileSystem = FileSystem.get(new Configuration());
        FileDeleter.delete(new File(dir + "/" + MODEL_BIN));
        fileSystem.moveFromLocalFile(new Path(MODEL_BIN), new Path(dir));

    }


    public static Learner loadModel(String dir, String fname) throws IOException {
        // move model to the node
        FileSystem fileSystem = FileSystem.get(new Configuration());
        fileSystem.copyToLocalFile(new Path(dir + fname), new Path(MODEL_BIN));

        Learner learner = IOLearner.loadLearner(MODEL_BIN);
        return learner;
    }


}
