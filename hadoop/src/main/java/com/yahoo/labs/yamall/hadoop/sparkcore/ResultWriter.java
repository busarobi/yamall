package com.yahoo.labs.yamall.hadoop.sparkcore;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.*;
import java.net.URI;

/**
 * Created by busafekete on 7/23/17.
 */
public class ResultWriter {
    public static void writeToHDFS( String fileName, String content ) throws IOException {
        Configuration conf = new Configuration();
        System.out.println("Connecting to -- "+conf.get("fs.defaultFS"));

        //Destination file in HDFS
        FileSystem fs = FileSystem.get(URI.create(fileName), conf);
        Path path = new Path(fileName);
        FileDeleter.delete(new File(fileName));

        OutputStream out = fs.create(path);
        OutputStreamWriter ow = new OutputStreamWriter(out);
        BufferedWriter writer = new BufferedWriter(ow);
        writer.write(content);

        writer.close();
    }
}
