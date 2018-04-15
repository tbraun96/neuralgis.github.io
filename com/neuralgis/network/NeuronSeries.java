package com.neuralgis.network;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.locks.LockSupport;

public class NeuronSeries {

    public static int INPUT_LAYER = 0;
    public static int HIDDEN_LAYER = 1;
    public static int OUTPUT_LAYER = 2;

    private int SERIES_TYPE;
    private int COLUMN_INDEX;
    private Brain brain;
    private Neuron[] NEURONS;
    private float[] WEIGHTS_IN;
    private int DEPTH;
    private NeuronState STATE;
    private ArrayList<Float> VALUES;
    private float[] inputs;

    public NeuronSeries(float[] data, int SERIES_TYPE, int COLUMN_INDEX, Brain brain) throws IOException {
        this.inputs = data;
        this.SERIES_TYPE = SERIES_TYPE;
        this.COLUMN_INDEX = COLUMN_INDEX;
        this.brain = brain;
        this.setDepth(data.length);

        if (brain.getNetworkType() == NetworkType.SIMPLE_BACKPROPAGATION) {
            if (SERIES_TYPE == INPUT_LAYER) this.NEURONS = new Neuron[data.length];
            else if (SERIES_TYPE == HIDDEN_LAYER) this.NEURONS = new Neuron[data.length];
            else if (SERIES_TYPE == OUTPUT_LAYER) this.NEURONS = new Neuron[data.length];
            this.WEIGHTS_IN = new float[brain.getNeuronSeries(COLUMN_INDEX - 1).getDepth()];
        }

        this.MAX_THREADS = Runtime.getRuntime().availableProcessors();
        this.THREADS = new ParallelWorker[this.MAX_THREADS];

        int stride = (int) Math.floor(DEPTH / MAX_THREADS);
        int prevEndIdx = 0;
        for (int i = 0; i < MAX_THREADS -1; i++){
            this.THREADS[i] = new ParallelWorker(prevEndIdx,prevEndIdx + stride -1);
            prevEndIdx += stride;
        }
        this.THREADS[MAX_THREADS -1] = new ParallelWorker(prevEndIdx, this.DEPTH -1);
    }

    private void printInitDebug(){

    }

    
    private void forward(){
        if (COLUMN_INDEX == HIDDEN_LAYER){
            //this.setStartIndex(brain.getNeuronSeries(COLUMN_INDEX - 1).DEPTH * this.DEPTH); //the number of previous connections

        }
        if (COLUMN_INDEX == INPUT_LAYER){
            
        }
        if (COLUMN_INDEX == OUTPUT_LAYER){
            
        }
    }

    private void backward(){

    }


    private int MAX_THREADS;
    private ParallelWorker[] THREADS;

    protected ParallelWorker[] getThreads(){
        return this.THREADS;
    }

    //protected int[] getIndicesConsideringThreadCount(){ }

    //if run() is called, it is always called by the Brain. The Brain will also set the STATE of this NeuronSeries prior to executing this function

    public void fire() {

        if (this.brain.getNetworkType() == NetworkType.SIMPLE_BACKPROPAGATION) {
            if (this.SERIES_TYPE == OUTPUT_LAYER) {
                fireSerial();
                computeError();
                //now, compute error. Brain does the rest. We are done here
            } else if (this.SERIES_TYPE == HIDDEN_LAYER) {
                fireParallel();
            }
        }
    }

    private void fireParallel(){
        if (MAX_THREADS == 1){
            fireSerial();
            return;
        }

        //execute each thread
        for (int i = 0; i < THREADS.length; i++){
            new Thread(THREADS[i]).run();
        }

        //wait for threads to finish
        do {
            LockSupport.parkNanos(100000); //wait 100 microseconds
        } while(isRunning());

    }

    private boolean isRunning(){
        for (int i = 0; i < THREADS.length; i++){
            if (THREADS[i].isAlive()) return true;
        }
        return false;
    }

    private void fireSerial(){
        for (int i = 0; i < DEPTH; i++){
            NEURONS[i].fire();
        }
    }

    /**
     * Only executed on output neurons, and only executed after output neurons fired.
     */
    private void computeError(){

    }

    protected NeuronState getState() {
        return STATE;
    }

    protected void setState(NeuronState STATE) {
        this.STATE = STATE;
    }

    protected int getMaxThreads() {
        return MAX_THREADS;
    }

    protected void setMaxThreads(int MAX_THREADS) {
        this.MAX_THREADS = MAX_THREADS;
    }

    protected ArrayList<Float> getValues() {
        return VALUES;
    }

    protected void setValues(ArrayList<Float> VALUES) {
        this.VALUES = VALUES;
    }

    protected int getDepth() {
        return DEPTH;
    }

    protected void setDepth(int DEPTH) {
        this.DEPTH = DEPTH;
    }

    protected class ParallelWorker extends Thread {

        private int startIdx;
        private int endIdx;

        public ParallelWorker(int startIdx, int endIdx){
            System.out.println("Starting parallel worker[" + startIdx + "," + endIdx + "]");
            this.startIdx = startIdx;
            this.endIdx = endIdx;
        }

        @Override
        public void run(){
            for (int i = startIdx; i <= endIdx; i++) NEURONS[i].fire();
        }

    }

}
