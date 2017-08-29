package com.yahoo.labs.yamall.spark.helper;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


import java.io.File;
import java.io.IOException;

/**
 * Created by busafekete on 7/23/17.
 */
public class FileDeleter {

    public static void delete(File fileToDelete) {

        if (fileToDelete.exists()) {

            boolean status = fileToDelete.delete();
            if (status)
                System.out.println("Deleted successfully: " + fileToDelete.getAbsolutePath());
            else
                System.out.println("Failed to delete: " + fileToDelete.getAbsolutePath());
        }
//        else
//            System.out.println("File doesn't exist: " + fileToDelete.getAbsolutePath());
    }

    public static void directoryDeleter(String dirname) {
        // Delete old file here
        try {
            FileSystem  hdfs = FileSystem.get(new Configuration());
            Path newFolderPath = new Path(dirname);

            if(hdfs.exists(newFolderPath)){
                System.out.println("EXISTS");
                hdfs.delete(newFolderPath, true);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}