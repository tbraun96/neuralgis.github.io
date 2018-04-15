package com.neuralgis.network;

import org.apache.commons.lang.ArrayUtils;

import java.util.ArrayList;
import java.util.concurrent.locks.LockSupport;

public class Neuron {

    private int COL_INDEX;
    private int ROW_INDEX;
    private int NEURON_TYPE;
    private NetworkType NETWORK_TYPE;
    private float VALUE;
    private Brain brain;
    private int THREAD_INDEX;
    private ArrayList<Float> INBOUND_WEIGHTS;

    private int[] ranges = new int[2];

    protected Neuron(int COL_INDEX, int ROW_INDEX, int NEURON_TYPE, NetworkType NETWORK_TYPE, int THREAD_INDEX, Brain brain) {
        this.COL_INDEX = COL_INDEX;
        this.ROW_INDEX = ROW_INDEX;
        this.NEURON_TYPE = NEURON_TYPE;
        this.NETWORK_TYPE = NETWORK_TYPE;
        this.THREAD_INDEX = THREAD_INDEX;
        this.brain = brain;
        getRangesForward();
        initInboundWeights();
    }

    private void initInboundWeights() {
        if (NEURON_TYPE == NeuronSeries.HIDDEN_LAYER || NEURON_TYPE == NeuronSeries.OUTPUT_LAYER) {
            int prevDepth = this.brain.getNeuronSeries(COL_INDEX - 1).getDepth();
            this.INBOUND_WEIGHTS = new ArrayList<Float>();
            for (int i = 0; i < prevDepth; i++) {
                this.INBOUND_WEIGHTS.add(i, this.brain.getRandomSeed());
            }
        }

        if (NEURON_TYPE == NeuronSeries.INPUT_LAYER) {
            this.setValue(this.brain.getInput()[ROW_INDEX]);
        }
    }

    protected void setValue(float VALUE) {
        this.VALUE = VALUE;
    }

    protected float getValue() {
        return this.VALUE;
    }


    public void fire() {
        if (NETWORK_TYPE == NetworkType.SIMPLE_BACKPROPAGATION) {
            if (NEURON_TYPE == NeuronSeries.HIDDEN_LAYER) {
                if (brain.getNeuronSeries(COL_INDEX).getState() == NeuronState.FEED_FORWARD) {
                    ArrayList<Float> previousLayerOutputs = brain.getNeuronSeries(COL_INDEX - 1).getValues();
                    INBOUND_WEIGHTS = (ArrayList<Float>) brain.getNeuronSeries(COL_INDEX - 1).getValues().subList(this.ranges[0], this.ranges[1]);
                    //addAllIntoSingleFloatValue(multiply previousLayerOutput[i] * INBOUND_WEIGHTS[i] -> use kernel.mult()) -> input into TransferFunction and set this neuron equal to TF's output.
                    this.VALUE = TransferFunction.sigmoid(this.brain, ArrayUtils.toPrimitive((Float[]) INBOUND_WEIGHTS.toArray()), ArrayUtils.toPrimitive((Float[]) previousLayerOutputs.toArray()));

                } else if (brain.getNeuronSeries(COL_INDEX).getState() == NeuronState.BACKPROPAGATION) {

                }

            }

            else if(NEURON_TYPE == NeuronSeries.OUTPUT_LAYER){
                if (brain.getNeuronSeries(COL_INDEX).getState() == NeuronState.FEED_FORWARD){
                    ArrayList<Float> previousLayerOutputs = brain.getNeuronSeries(COL_INDEX - 1).getValues();
                    INBOUND_WEIGHTS = (ArrayList<Float>) brain.getNeuronSeries(COL_INDEX - 1).getValues().subList(this.ranges[0], this.ranges[1]);
                    this.VALUE = TransferFunction.sigmoid(this.brain, ArrayUtils.toPrimitive((Float[]) INBOUND_WEIGHTS.toArray()), ArrayUtils.toPrimitive((Float[]) previousLayerOutputs.toArray()));
                }
            }
        }
    }

    private int[] getRangesForward() {
        int end;
        int start = 0;
        for (int i = COL_INDEX - 1; i > 0; i--) { //through hidden
            start += brain.getNeuronSeries(i).getDepth() * brain.getNeuronSeries(i - 1).getDepth();
        }

        //now factor ROW_INDEX
        start += brain.getNeuronSeries(COL_INDEX - 1).getDepth() * (ROW_INDEX - 1); //if COL_INDEX=1, this happens without above loop
        end = start + brain.getNeuronSeries(COL_INDEX - 1).getDepth();
        return new int[]{start, end};
    }
}
