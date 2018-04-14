package com.neuralgis.network;

import java.io.IOException;
import java.util.ArrayList;

public class NeuronSeries implements Runnable{

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
    
    public NeuronSeries(float[] data, int SERIES_TYPE, int COLUMN_INDEX, Brain brain) throws IOException {
        this.data = data;
        this.SERIES_TYPE = SERIES_TYPE;
        this.COLUMN_INDEX = COLUMN_INDEX;
        this.brain = brain;
        this.setDepth(data.length);
        if (brain.getNetworkType() == NetworkType.SIMPLE_BACKPROPAGATION) {
            if (SERIES_TYPE == INPUT_LAYER) this.NEURONS = new float[data.length];
            else if (SERIES_TYPE == HIDDEN_LAYER) this.NEURONS = new float[data.length];
            else if (SERIES_TYPE == OUTPUT_LAYER) this.NEURONS = new float[data.length];
            this.WEIGHTS_IN = new float[brain.getNeuronSeries(COLUMN_INDEX - 1).getDepth()];
        }

    }


    
    private int forward(){
        if (COLUMN_INDEX == HIDDEN_LAYER){
            this.setStartIndex(brain.getNeuronSeries(COLUMN_INDEX - 1).DEPTH * this.DEPTH); //the number of previous connections

        }
        if (COLUMN_INDEX == INPUT_LAYER){
            
        }
        if (COLUMN_INDEX == OUTPUT_LAYER){
            
        }
    }

    private int backward(){

    }


    private int MAX_THREADS;
    private Thread[] THREADS;

    protected Thread[] getThreads(){
        return this.THREADS;
    }

    protected int[] getIndicesConsideringThreadCount(){

    }

    @Override
    public void run() {

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

}
