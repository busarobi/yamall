package com.yahoo.labs.yamall.synthetic.helper;

import java.util.concurrent.LinkedBlockingQueue;

/**
 * Created by busafekete on 8/15/17.
 */
public class JobParallelLauncher {

    public interface JobRunner {
        void run();
    }

    private static class JobRunnerThread extends Thread {

        private LinkedBlockingQueue<JobRunner> queue;

        public JobRunnerThread(LinkedBlockingQueue<JobRunner> queue) {
            this.queue = queue;
        }

        @Override
        public void run() {
            try {
                JobRunner job = queue.poll();
                while (job != null) {
                    job.run();
                    job = queue.poll();
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private int jobsInParallel;
    private LinkedBlockingQueue<JobRunner> queue;

    public JobParallelLauncher(int jobsInParallel) {
        this.queue = new LinkedBlockingQueue<JobRunner>();
        this.jobsInParallel = jobsInParallel;
    }

    public void addJob(JobRunner job) {
        queue.add(job);
    }

    public void run() {

        JobRunnerThread[] threads = new JobRunnerThread[jobsInParallel];
        for (int i = 0; i < threads.length; i++) {
            threads[i] = new JobRunnerThread(queue);
            threads[i].start();
        }

        for (JobRunnerThread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
