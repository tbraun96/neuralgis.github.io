package com.neuralgis.network;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Brain {

    private float[] input;
    private NeuronSeries[] columns;
    private NetworkType networkType;
    public Brain(String fileInput, NetworkType type, int ...params) throws IOException {
        BufferedImage image = ImageIO.read(new File(System.getProperty("user.dir") + "/" + fileInput));
        this.input = RasterLoader.rasterToNetworkInput(image);
        this.networkType = type;
        if (type == NetworkType.SIMPLE_BACKPROPAGATION){
            int hiddenLayers = params[0];
            this.columns = new NeuronSeries[1 + hiddenLayers + 1];
            //initialize NeuralNetworks
            this.columns[0] = new NeuronSeries();
        }
    }

    public NetworkType getNetworkType(){
        return this.networkType;
    }

    public NeuronSeries getNeuronSeries(int column){
        return columns[column];
    }

}
