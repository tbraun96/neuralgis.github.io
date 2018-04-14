package com.neuralgis.network;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class RasterLoader {

    public static float[] rasterToNetworkInput(BufferedImage image){
        int dim = image.getHeight() * image.getWidth();
        float[] array = new float[dim];
        int index = 0;
        int max = 0;
        for (int y = 0; y < image.getHeight(); y++){
            for (int x = 0; x < image.getWidth(); x++){
                Color c = new Color(image.getRGB(x,y));
                int unnormed = c.getRed() + c.getGreen() + c.getBlue();
                if (unnormed > max) max = unnormed;
                array[index] = unnormed;
                index = index + 1;
            }
        }
        return arrayNorm(max,array);
    }

    private static float[] arrayNorm(int max, float[] array){
        System.out.println(max);
        for (int i = 0; i <array.length; i++) array[i] = array[i]/max;
        return array;
    }

    public static void main(String ...args) throws IOException {
        BufferedImage image = ImageIO.read(new File(System.getProperty("user.dir") + "/einstein.jpg"));
        float[] input = rasterToNetworkInput(image);
        for (int i = 0; i < 10; i++) System.out.println(i + " = " + input[i]);
    }

}
