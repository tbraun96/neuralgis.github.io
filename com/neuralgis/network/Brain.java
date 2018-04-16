package com.neuralgis.network;

import com.neuralgis.gpu.GPUMathKernel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Brain {

    private float[] input;
    private NeuronSeries[] columns;
    private NetworkType networkType;
    private GPUMathKernel kernel;
    private float[] expectedOutputs;

    public Brain(String fileInput, NetworkType type, float[] expectedOutputs, int ...params) throws IOException {
        BufferedImage image = ImageIO.read(new File(System.getProperty("user.dir") + "/" + fileInput));
        this.setInput(RasterLoader.rasterToNetworkInput(image));
        this.networkType = type;
        this.expectedOutputs = expectedOutputs;
        this.kernel = new GPUMathKernel();
        if (type == NetworkType.SIMPLE_BACKPROPAGATION){
            int hiddenLayers = params[0];
            this.columns = new NeuronSeries[1 + hiddenLayers + 1];
            //initialize NeuralNetworks
            //this.columns[0] = new NeuronSeries();
        }
    }

    public NetworkType getNetworkType(){
        return this.networkType;
    }

    public NeuronSeries getNeuronSeries(int column){
        return columns[column];
    }

    private Random random = new Random();
    protected float getRandomSeed(){
        return random.nextFloat();
    }

    protected float[] getInput() {
        return input;
    }

    protected void setInput(float[] input) {
        this.input = input;
    }

    protected GPUMathKernel getKernel() {
        return kernel;
    }

    protected float[] getExpectedOutputs() {
        return expectedOutputs;
    }
}
