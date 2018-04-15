package com.neuralgis.network;

public class TransferFunction {

    public static float sigmoid(Brain brain, float[] inboundWeights, float[] inputColumn){
        //1/(1+e^(-x))
        return (float) (1/(1 + Math.exp((double) -brain.getKernel().sumAll(brain.getKernel().mult(inboundWeights, inputColumn)))));
    }

}
