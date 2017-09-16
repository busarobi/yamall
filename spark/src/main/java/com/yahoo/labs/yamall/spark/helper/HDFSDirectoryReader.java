package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.zip.GZIPInputStream;

/**
 * Created by busafekete on 9/15/17.
 */
public class HDFSDirectoryReader {
    protected ReaderThread  readerthread = null;
    protected String dirname = null;
    protected BlockingQueue<Instance> blockingQueue = null;
    protected int bufferSize = 16384;
    protected int bits = 22;
    protected Thread rthread = null;
    protected int processedItem = 0;
    protected int nFeatures = 0;
    protected int nLabels = 0;

    public HDFSDirectoryReader(String dirname, int bits) throws IOException {
        this.dirname = dirname;
        this.bits = bits;

        this.blockingQueue = new ArrayBlockingQueue<>(this.bufferSize);
        this.readerthread = new ReaderThread(this.blockingQueue, this.dirname, this.bits);

        this.rthread = new Thread(this.readerthread);
        this.rthread.start();
    }

    public boolean hasNext() throws IOException {
        if (this.blockingQueue.size() > 0) return true;
        if ((! this.readerthread.isEndOfFile() )) { // not end of line, but we do not know whether new instance will come
            int waiting_time = 0;
            while (true) {
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                if (this.blockingQueue.size() > 0) return true;
                if (this.readerthread.isEndOfFile()) return false;
                waiting_time++;

                if (waiting_time>100) throw new IOException("Reading time out");
            }
        }
        return false;
    }

    public Instance getNextInstance() {
        Instance instance = null;
        try {
            instance = this.blockingQueue.take();
            this.processedItem++;
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return instance;
    }


    public void reset() throws IOException {
        this.close();
        this.blockingQueue = new ArrayBlockingQueue<Instance>(this.bufferSize);
        this.readerthread = new ReaderThread(this.blockingQueue, this.dirname, this.bits);
        this.rthread = new Thread(this.readerthread);
        this.rthread.start();
    }

    // for reading data
    public static class ReaderThread implements Runnable{
        protected BlockingQueue<Instance> blockingQueue = null;
        protected String dirname = null;

        protected volatile boolean endOfFile = false;
        public volatile boolean shutdown = true;
        public volatile ArrayList<Path> featureFilePaths = null;
        protected VWParser vwparser = null;
        protected BufferedReader br = null;
        protected int pathi = 0;
        protected FileSystem hdfs = null;

        public ReaderThread(BlockingQueue<Instance> blockingQueue, String dirname, int bits) throws IOException {
            this.blockingQueue = blockingQueue;
            this.dirname = dirname;

            // Get a list of all the files in the inputPath directory. We will read these files one at a time
            //the second boolean parameter here sets the recursion to true

            hdfs = FileSystem.get(new Configuration());

            featureFilePaths = new ArrayList<>();
            RemoteIterator<LocatedFileStatus> fileStatusListIterator = hdfs.listFiles(
                    new Path(this.dirname), true);

            while (fileStatusListIterator.hasNext()) {
                LocatedFileStatus fileStatus = fileStatusListIterator.next();
                String fileName = fileStatus.getPath().getName();
                if (fileName.contains(".gz") || fileName.contains(".bz2") || fileName.startsWith("part"))
                    featureFilePaths.add(fileStatus.getPath());
            }

            // pick some file for testing
            Collections.shuffle(featureFilePaths);

            System.out.println("--- Number of train files: " + featureFilePaths.size());

            this.vwparser = new VWParser(bits, null, false);
            this.br = openNext();
        }

        protected BufferedReader openNext() throws IOException {
            if (pathi>=featureFilePaths.size()) return null;

            Path featureFile = featureFilePaths.get(pathi++);
            System.out.println("--- Open: " + featureFile.toString());

            BufferedReader brc = null;
            if (featureFile.getName().contains(".gz"))
                brc = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
            else if (featureFile.getName().contains(".bz2"))
                brc = new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(hdfs.open(featureFile))));
            else
                brc = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));

            return brc;
        }


        @Override
        public void run() {
            try {
                String buffer = br.readLine();
                if (buffer == null ) {
                    this.endOfFile = true;
                }

                if (this.endOfFile == false ) {
                    while(true){
                        Instance instance = vwparser.parse(buffer);
                        blockingQueue.put(instance);

                        buffer = br.readLine();

                        if ( buffer == null ) {
                            br = openNext();
                            if (br == null) {
                                this.endOfFile = true;
                                break;
                            } else {
                                buffer = br.readLine();  //assumes that there is no empty file
                            }
                        }
                        //if (endOfFile ==true) break;
                        if (shutdown ==false) return;
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        synchronized public boolean isEndOfFile() {
            return this.endOfFile;
        }

    }

    protected void finalize() {
        this.close();
    }

    public void close() {
        System.out.println( "--- Closing reader thread");
        if ( ( this.readerthread != null ) && (this.readerthread.endOfFile==false) ){
            this.readerthread.shutdown = false;
            this.getNextInstance();
        }
    }
}
